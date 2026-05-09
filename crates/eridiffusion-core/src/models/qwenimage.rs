//! QwenImage-2512 training model — dual-stream MMDiT with LoRA.
//!
//! ## Architecture (60 double-stream blocks)
//!
//! Per block (musubi-tuner `qwen_image_model.py`, flame `qwenimage_dit.rs`):
//! ```text
//! img_mod → 6 params: shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
//! txt_mod → 6 params: same for text stream
//!
//! img_norm1(img) * (1+img_scale) + img_shift → Q/K/V (to_q/to_k/to_v)
//! txt_norm1(txt) * (1+txt_scale) + txt_shift → Q/K/V (add_q/add_k/add_v)
//! joint attention: cat(img_q, txt_q), cat(img_k, txt_k), cat(img_v, txt_v) → SDPA → split
//! img += gate_attn * img_attn_out
//! txt += gate_attn * txt_attn_out
//!
//! img_norm2(img) * (1+img_scale_mlp) + img_shift_mlp → GELU FFN → gate_mlp * out
//! txt_norm2(txt) * (1+txt_scale_mlp) + txt_shift_mlp → GELU FFN → gate_mlp * out
//! ```
//!
//! ## LoRA targets (musubi-tuner `lora_qwen_image.py`)
//!
//! `QwenImageTransformerBlock`, exclude `_mod_`.
//! Covers: attn.to_q/k/v, attn.to_out.0, attn.add_q/k/v_proj, attn.to_add_out,
//!         img_mlp.net.0.proj, img_mlp.net.2, txt_mlp.net.0.proj, txt_mlp.net.2

use flame_core::{parameter::Parameter, DType, Result, Shape, Tensor};
use crate::lora::LoRALinear;
use std::collections::HashMap;
use std::sync::Arc;

pub const NUM_LAYERS: usize = 60;
pub const DIM: usize = 3072;
pub const NUM_HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const IN_CHANNELS: usize = 64;  // 16 VAE channels × 2×2 pack
pub const OUT_CHANNELS: usize = 16;
pub const JOINT_DIM: usize = 3584;   // Qwen2.5-VL-7B hidden
pub const MLP_HIDDEN: usize = 12288; // 4 × dim
pub const NORM_EPS: f32 = 1e-6;
pub const ROPE_THETA: f64 = 10000.0;
pub const ROPE_AXES_DIMS: [usize; 3] = [16, 56, 56];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoraTarget {
    ImgQ, ImgK, ImgV, ImgOut,
    TxtQ, TxtK, TxtV, TxtOut,
    ImgFfnUp, ImgFfnDown,
    TxtFfnUp, TxtFfnDown,
}

#[derive(Clone)]
pub struct QwenImageLoraBundle {
    pub adapters: HashMap<(usize, LoraTarget), LoRALinear>,
}

impl QwenImageLoraBundle {
    pub fn new(
        num_layers: usize,
        rank: usize,
        alpha: f32,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        let mut adapters = HashMap::new();
        let targets = [
            (LoraTarget::ImgQ, DIM, DIM),
            (LoraTarget::ImgK, DIM, DIM),
            (LoraTarget::ImgV, DIM, DIM),
            (LoraTarget::ImgOut, DIM, DIM),
            (LoraTarget::TxtQ, DIM, DIM),
            (LoraTarget::TxtK, DIM, DIM),
            (LoraTarget::TxtV, DIM, DIM),
            (LoraTarget::TxtOut, DIM, DIM),
            (LoraTarget::ImgFfnUp, DIM, MLP_HIDDEN),
            (LoraTarget::ImgFfnDown, MLP_HIDDEN, DIM),
            (LoraTarget::TxtFfnUp, DIM, MLP_HIDDEN),
            (LoraTarget::TxtFfnDown, MLP_HIDDEN, DIM),
        ];
        for i in 0..num_layers {
            for &(target, in_dim, out_dim) in &targets {
                let lora = LoRALinear::new(in_dim, out_dim, rank, alpha, device.clone(), seed + i as u64)?;
                adapters.insert((i, target), lora);
            }
        }
        Ok(Self { adapters })
    }

    pub fn num_adapters(&self) -> usize { self.adapters.len() }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for lora in self.adapters.values() {
            params.extend(lora.parameters());
        }
        params
    }

    /// Names parallel to `parameters()`. Each adapter contributes
    /// `(prefix.lora_A, prefix.lora_B)` in order — matches `LoRALinear::parameters`.
    pub fn parameter_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for ((block_idx, target), _) in self.adapters.iter() {
            let prefix = format!(
                "transformer_blocks.{block_idx}.{}",
                Self::target_suffix(*target),
            );
            names.push(format!("{prefix}.lora_A"));
            names.push(format!("{prefix}.lora_B"));
        }
        names
    }

    pub fn refresh_caches(&self) {
        for lora in self.adapters.values() {
            lora.refresh_cache();
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let mut tensors = HashMap::new();
        for (&(block_idx, target), lora) in &self.adapters {
            let prefix = format!(
                "transformer_blocks.{block_idx}.{}",
                Self::target_suffix(target),
            );
            lora.save_tensors(&prefix, &mut tensors)?;
        }
        flame_core::serialization::save_tensors(
            &tensors, path,
            flame_core::serialization::SerializationFormat::SafeTensors,
        )
    }

    /// Inverse of `save` — load LoRA weights from a safetensors file produced
    /// by either this trainer's `save` (or `save_weights`) OR by
    /// `checkpoint::save_full` (which embeds the same LoRA weights as plain
    /// tensor entries; the optimizer-state prefix `__opt__/` is ignored).
    pub fn load(&self, path: &std::path::Path, device: &std::sync::Arc<cudarc::driver::CudaDevice>)
        -> Result<()>
    {
        let raw = flame_core::serialization::load_file(path, device)?;
        let mut applied = 0usize;
        let mut missing = 0usize;
        for (&(block_idx, target), lora) in &self.adapters {
            let prefix = format!(
                "transformer_blocks.{block_idx}.{}",
                Self::target_suffix(target),
            );
            // load_tensors looks up `{prefix}.lora_A.weight` and
            // `{prefix}.lora_B.weight`. If both keys exist and shapes match,
            // applies; otherwise returns an error. Track per-adapter outcome.
            match lora.load_tensors(&prefix, &raw) {
                Ok(()) => applied += 1,
                Err(_) => missing += 1,
            }
        }
        log::info!(
            "[qwenimage] LoRA loaded: {}/{} adapters from {}",
            applied, applied + missing, path.display()
        );
        if applied == 0 {
            return Err(flame_core::Error::InvalidInput(format!(
                "no LoRA adapters matched any prefix in {}", path.display()
            )));
        }
        // Refresh the BF16 transposed weight cache so the next forward picks
        // up the new lora_A/lora_B values.
        self.refresh_caches();
        Ok(())
    }

    /// Canonical save prefix suffix for each LoRA target. Public so trainer
    /// binaries can build matching `(name, Parameter)` pairs for
    /// `checkpoint::save_full` without duplicating the mapping table.
    pub fn target_suffix(target: LoraTarget) -> &'static str {
        match target {
            LoraTarget::ImgQ => "attn.to_q",
            LoraTarget::ImgK => "attn.to_k",
            LoraTarget::ImgV => "attn.to_v",
            LoraTarget::ImgOut => "attn.to_out.0",
            LoraTarget::TxtQ => "attn.add_q_proj",
            LoraTarget::TxtK => "attn.add_k_proj",
            LoraTarget::TxtV => "attn.add_v_proj",
            LoraTarget::TxtOut => "attn.to_add_out",
            LoraTarget::ImgFfnUp => "img_mlp.net.0.proj",
            LoraTarget::ImgFfnDown => "img_mlp.net.2",
            LoraTarget::TxtFfnUp => "txt_mlp.net.0.proj",
            LoraTarget::TxtFfnDown => "txt_mlp.net.2",
        }
    }
}

pub struct QwenImageTrainingModel {
    pub bundle: QwenImageLoraBundle,
    pub num_blocks: usize,
    /// Full fine-tune parameters (None in LoRA mode).
    /// Maps weight key -> Parameter for all trainable base weights.
    pub fft_params: Option<HashMap<String, Parameter>>,
    pub is_full_finetune: bool,
    resident: HashMap<String, Tensor>,
    /// Resident block weights (used when model fits in VRAM, e.g. QWENIMAGE_MAX_BLOCKS=2).
    block_weights: Vec<HashMap<String, Tensor>>,
    /// BlockOffloader-backed block loading for full 60-block model.
    /// When Some, block_weights is empty — blocks are loaded per-step via offloader.
    /// Wrapped in `Arc<Mutex>` so the per-block `checkpoint_offload` closure
    /// can hold a cheap clone and re-fetch weights from pinned memory inside
    /// the closure (instead of capturing the weights HashMap by move, which
    /// previously held 60 × ~648 MB = ~38 GB of GPU memory across all the
    /// captured closures and OOM'd training around block 24).
    pub offloader: Option<std::sync::Arc<std::sync::Mutex<crate::training::block_offload::BlockOffloader>>>,
    device: Arc<cudarc::driver::CudaDevice>,
}

impl QwenImageTrainingModel {
    pub fn load(
        model_path: &std::path::Path,
        lora_rank: usize,
        lora_alpha: f32,
        full_finetune: bool,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        log::info!("[qwenimage-trainer] loading from {}", model_path.display());

        let max_blocks = std::env::var("QWENIMAGE_MAX_BLOCKS")
            .ok().and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(NUM_LAYERS);

        // Decide: BlockOffloader (per-step block loading) vs resident (all blocks on GPU).
        // Use BlockOffloader when loading all 60 blocks from a sharded directory.
        let use_offloader = max_blocks == NUM_LAYERS && model_path.is_dir();

        let (resident, per_block, offloader) = if use_offloader {
            // BlockOffloader path: load shared weights only, blocks loaded per-step.
            log::info!("[qwenimage-trainer] BlockOffloader mode: blocks loaded per-step");

            let shared_prefixes = [
                "img_in.", "txt_norm.", "txt_in.",
                "time_text_embed.", "norm_out.", "proj_out.",
            ];

            let mut resident = HashMap::new();
            let mut shard_paths: Vec<String> = Vec::new();

            for entry in std::fs::read_dir(model_path)
                .map_err(|e| flame_core::Error::Io(e.to_string()))?
            {
                let entry = entry.map_err(|e| flame_core::Error::Io(e.to_string()))?;
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                    shard_paths.push(p.to_string_lossy().into_owned());
                    // Load only shared (non-block) weights
                    let shared = flame_core::serialization::load_file_filtered(
                        &p, &device, |key| shared_prefixes.iter().any(|pfx| key.starts_with(pfx)),
                    )?;
                    resident.extend(shared);
                }
            }
            shard_paths.sort();

            let path_refs: Vec<&str> = shard_paths.iter().map(|s| s.as_str()).collect();

            struct QwenImageFacilitator;
            impl crate::training::block_offload::BlockFacilitator for QwenImageFacilitator {
                fn block_count(&self) -> usize { NUM_LAYERS }
                fn classify_key(&self, key: &str) -> Option<usize> {
                    let rest = key.strip_prefix("transformer_blocks.")?;
                    rest.split('.').next()?.parse().ok()
                }
            }
            // Phase 5 — `TrainBlockFacilitator` wiring lets this trainer
            // switch to `TrainingBlockOffloader` in a follow-up without
            // having to re-derive the key classification.
            impl crate::training::training_offload::TrainBlockFacilitator for QwenImageFacilitator {
                fn is_trainable_key(&self, key: &str) -> bool {
                    key.contains(".lora_A") || key.contains(".lora_B")
                        || key.ends_with(".lora_a") || key.ends_with(".lora_b")
                }
                fn is_frozen_block_key(&self, key: &str) -> bool {
                    key.starts_with("transformer_blocks.") && !self.is_trainable_key(key)
                }
                fn shared_resident_key(&self, key: &str) -> bool {
                    !self.is_trainable_key(key) && !self.is_frozen_block_key(key)
                }
            }

            // Streaming mode caps pinned host RAM at 2 × max_block_bytes
            // (≈1.3 GB for Qwen-Image-2512) instead of pinning every block
            // upfront (≈39 GB), at the cost of a per-block CPU memcpy from
            // the safetensors mmap into the staging buffer at each prefetch.
            // Recommended for sample-only paths and any host where the full
            // pinned footprint would not fit.
            let use_streaming = std::env::var("QWEN_BLOCK_STREAMING")
                .ok()
                .map(|v| !matches!(v.as_str(), "0" | "" | "false" | "False"))
                .unwrap_or(false);
            let offloader = if use_streaming {
                log::info!("[qwenimage-trainer] BlockOffloader: streaming mode (QWEN_BLOCK_STREAMING=1)");
                crate::training::block_offload::BlockOffloader::load_streaming(
                    &path_refs, &QwenImageFacilitator, device.clone(),
                )
            } else {
                crate::training::block_offload::BlockOffloader::load(
                    &path_refs, &QwenImageFacilitator, device.clone(),
                )
            }
            .map_err(|e| flame_core::Error::InvalidInput(format!("BlockOffloader: {e}")))?;
            // native_layout=false (legacy default): offloader pre-transposes
            // 2D `.weight` tensors to `[Cin, Cout]` so the inline
            // `linear_bias` closure can pass them directly to `Tensor::matmul`
            // (which expects [Cin, Cout] for the rhs operand).

            log::info!("[qwenimage-trainer] BlockOffloader: {} blocks, {} shared weights",
                offloader.block_count(), resident.len());

            let per_block = Vec::new(); // empty — blocks loaded per-step
            (resident, per_block, Some(std::sync::Arc::new(std::sync::Mutex::new(offloader))))
        } else {
            // Resident path: load all blocks into GPU memory.
            let block_filter = |key: &str| -> bool {
                if !key.starts_with("transformer_blocks.") { return true; }
                if let Some(rest) = key.strip_prefix("transformer_blocks.") {
                    if let Some(dot) = rest.find('.') {
                        if let Ok(idx) = rest[..dot].parse::<usize>() {
                            return idx < max_blocks;
                        }
                    }
                }
                true
            };

            let all_weights = if model_path.is_dir() {
                let mut merged = HashMap::new();
                for entry in std::fs::read_dir(model_path)
                    .map_err(|e| flame_core::Error::Io(e.to_string()))?
                {
                    let entry = entry.map_err(|e| flame_core::Error::Io(e.to_string()))?;
                    let p = entry.path();
                    if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                        let shard = flame_core::serialization::load_file_filtered(
                            &p, &device, block_filter,
                        )?;
                        merged.extend(shard);
                    }
                }
                merged
            } else {
                flame_core::serialization::load_file_filtered(model_path, &device, block_filter)?
            };
            log::info!("[qwenimage-trainer] loaded {} weight tensors (resident)", all_weights.len());

            let mut resident = HashMap::new();
            let mut per_block: Vec<HashMap<String, Tensor>> = (0..NUM_LAYERS).map(|_| HashMap::new()).collect();

            for (key, tensor) in &all_weights {
                if key.starts_with("transformer_blocks.") {
                    if let Some(rest) = key.strip_prefix("transformer_blocks.") {
                        if let Some(dot_pos) = rest.find('.') {
                            if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                                if idx < NUM_LAYERS {
                                    per_block[idx].insert(key.clone(), tensor.clone());
                                }
                            }
                        }
                    }
                } else {
                    resident.insert(key.clone(), tensor.clone());
                }
            }
            drop(all_weights);

            log::info!("[qwenimage-trainer] {} resident, {} block maps", resident.len(), per_block.len());
            (resident, per_block, None)
        };

        log::info!("[qwenimage-trainer] mode: {}", if full_finetune { "full fine-tune" } else { "LoRA" });

        let actual_blocks = if use_offloader { NUM_LAYERS } else { max_blocks };
        let (bundle, fft_params) = if full_finetune {
            let mut params = HashMap::new();
            for (key, tensor) in &resident {
                let p = Parameter::new(tensor.to_dtype(DType::F32)?.requires_grad_(true));
                params.insert(key.clone(), p);
            }
            for block in &per_block {
                for (key, tensor) in block {
                    let p = Parameter::new(tensor.to_dtype(DType::F32)?.requires_grad_(true));
                    params.insert(key.clone(), p);
                }
            }
            log::info!("[qwenimage-trainer] full fine-tune: {} trainable weight tensors", params.len());
            let bundle = QwenImageLoraBundle { adapters: HashMap::new() };
            (bundle, Some(params))
        } else {
            let bundle = QwenImageLoraBundle::new(actual_blocks, lora_rank, lora_alpha, device.clone(), seed)?;
            log::info!("[qwenimage-trainer] {} LoRA adapters (rank={})", bundle.num_adapters(), lora_rank);
            (bundle, None)
        };

        Ok(Self {
            bundle,
            num_blocks: actual_blocks,
            fft_params,
            is_full_finetune: full_finetune,
            resident,
            block_weights: per_block,
            offloader,
            device,
        })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        if let Some(ref fft) = self.fft_params {
            fft.values().cloned().collect()
        } else {
            self.bundle.parameters()
        }
    }

    pub fn refresh_lora_cache(&self) {
        if !self.is_full_finetune {
            self.bundle.refresh_caches();
        }
    }

    /// Save trained weights.
    /// Full fine-tune: saves ALL model weights as BF16.
    /// LoRA: saves only adapter deltas.
    pub fn save_weights(&self, path: &std::path::Path) -> Result<()> {
        if let Some(ref fft) = self.fft_params {
            let mut tensors = HashMap::new();
            for (key, param) in fft {
                tensors.insert(key.clone(), param.tensor()?.to_dtype(DType::BF16)?);
            }
            flame_core::serialization::save_tensors(
                &tensors, path,
                flame_core::serialization::SerializationFormat::SafeTensors,
            )
        } else {
            self.bundle.save(path)
        }
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.resident.get(key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {key}")))
    }

    fn bw(&self, block_idx: usize, suffix: &str) -> Result<&Tensor> {
        let key = format!("transformer_blocks.{block_idx}.{suffix}");
        self.block_weights[block_idx].get(&key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {key}")))
    }

    /// Autograd-aware linear (matmul path for gradient flow through LoRA).
    /// `.contiguous()` after `.transpose()` is load-bearing — flame's BF16
    /// matmul reads saved tensors as if contiguous, so a strided transpose
    /// view silently mismatches dimensions (CONVENTIONS doc + autograd_ops
    /// transpose-contig fix).
    fn matmul_bias(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let in_feat = *dims.last().unwrap();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let out_feat = weight.shape().dims()[0];
        let x_2d = x.reshape(&[batch, in_feat])?;
        let wt = weight.transpose()?.contiguous()?;
        let mut out = x_2d.matmul(&wt)?;
        if let Some(b) = bias { out = out.add(b)?; }
        let mut out_shape = dims[..dims.len()-1].to_vec();
        out_shape.push(out_feat);
        out.reshape(&out_shape)
    }

    fn add_lora(&self, base: Tensor, input: &Tensor, block_idx: usize, target: LoraTarget) -> Result<Tensor> {
        if let Some(lora) = self.bundle.adapters.get(&(block_idx, target)) {
            let input_3d = if input.shape().dims().len() == 2 { input.unsqueeze(0)? } else { input.clone() };
            let delta = lora.forward_delta(&input_3d)?;
            base.add(&delta)
        } else {
            Ok(base)
        }
    }

    /// Inference-path linear (non-autograd, fast).
    ///
    /// Bypasses the cuBLASLt BIAS epilogue and adds bias as a separate
    /// elementwise op. The fused-bias path drifts ~0.003 mean abs (cos
    /// ≈ 0.999996) per call vs the unfused `matmul + add(bias)` reference
    /// — see `docs/FUSED_LINEAR3D_BUG_DIG.md`. Compounded across the
    /// shared-weight forwards (img_in, txt_in, time_text_embed,
    /// norm_out, proj_out) over 50 sample steps the drift accumulates.
    /// Separate add is bit-identical to the legacy `matmul + add(bias)`
    /// path that all the other ported models use.
    fn fused_linear(&self, x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
        let x3d = if x.shape().dims().len() == 2 { x.unsqueeze(0)? } else { x.clone() };
        let mut out = flame_core::ops::fused_inference::fused_linear3d_native(&x3d, w, None)?;
        if let Some(bias) = b {
            out = out.add(bias)?;
        }
        if x.shape().dims().len() == 2 { out.squeeze(Some(0)) } else { Ok(out) }
    }

    /// Full training forward.
    ///
    /// `packed_noisy`: [B, seq, 64] BF16 (packed latent: 16ch × 2×2)
    /// `timestep`: [B] BF16 (sigma in [0, 1])
    /// `txt_embed`: [B, txt_seq, 3584] BF16
    /// `latent_h`, `latent_w`: latent spatial dims (after VAE, before packing)
    pub fn forward(
        &mut self,
        packed_noisy: &Tensor,
        timestep: &Tensor,
        txt_embed: &Tensor,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<Tensor> {
        let dims = packed_noisy.shape().dims().to_vec();
        let (b, img_seq) = (dims[0], dims[1]);
        let txt_seq = txt_embed.shape().dims()[1];

        // Precompute 3-axis RoPE (image + text). Per-stream tables are kept
        // for the resident path (`self.dual_stream_block` legacy code); the
        // BlockOffloader inference path uses the joint TXT-first
        // concatenation below to feed `dual_stream_block_iflame` directly.
        let pack_h = latent_h / 2;
        let pack_w = latent_w / 2;
        let (img_cos, img_sin) = compute_image_rope(pack_h, pack_w, ROPE_THETA)?;
        let (txt_cos, txt_sin) = compute_text_rope(txt_seq, pack_h, pack_w, ROPE_THETA)?;
        // Joint cos/sin in [txt, img] order — matches inference-flame's
        // `build_rope_tables` and the `dual_stream_block_iflame` cat order.
        // Shape: [1, 1, n_total, half] BF16, consumed by `rope_fused_bf16`.
        let pe_cos = Tensor::cat(&[&txt_cos, &img_cos], 0)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let pe_sin = Tensor::cat(&[&txt_sin, &img_sin], 0)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        // Timestep embedding: sinusoidal(256) → Linear → SiLU → Linear
        let temb = sinusoidal_embedding(timestep, 256)?;
        let temb = self.fused_linear(&temb, self.w("time_text_embed.timestep_embedder.linear_1.weight")?,
            Some(self.w("time_text_embed.timestep_embedder.linear_1.bias")?))?;
        let temb = temb.silu()?;
        let temb = self.fused_linear(&temb, self.w("time_text_embed.timestep_embedder.linear_2.weight")?,
            Some(self.w("time_text_embed.timestep_embedder.linear_2.bias")?))?;

        // Image input projection: [B, seq, 64] → [B, seq, 3072]
        let mut img = self.fused_linear(packed_noisy, self.w("img_in.weight")?, Some(self.w("img_in.bias")?))?;

        // Text input: RMSNorm → Linear
        let txt_normed = flame_core::norm::rms_norm(txt_embed, &[JOINT_DIM], Some(self.w("txt_norm.weight")?), NORM_EPS)?;
        let mut txt = self.fused_linear(&txt_normed, self.w("txt_in.weight")?, Some(self.w("txt_in.bias")?))?;

        // 60 dual-stream blocks — BlockOffloader or resident path.
        let n_blocks = if self.offloader.is_some() {
            NUM_LAYERS
        } else {
            std::env::var("QWENIMAGE_MAX_BLOCKS")
                .ok().and_then(|s| s.parse::<usize>().ok())
                .map(|n| n.min(NUM_LAYERS)).unwrap_or(NUM_LAYERS)
        };
        if n_blocks != NUM_LAYERS {
            log::warn!("[qwenimage-trainer] QWENIMAGE_MAX_BLOCKS={n_blocks}");
        }

        if let Some(ref offloader_arc) = self.offloader {
            // BlockOffloader path. Critical design:
            //   - The per-block `checkpoint_offload` closure does **not**
            //     capture the block weights HashMap. Instead it captures a
            //     cheap `Arc<Mutex<BlockOffloader>>` clone and the block
            //     index, then re-fetches the weights inside the closure on
            //     every call (forward once + each backward recompute).
            //   - This avoids the 38 GB closure-capture leak that previously
            //     OOM'd around block 24: each of 60 captured closures was
            //     holding ~648 MB of weight Arcs alive until backward.
            //   - Trade-off: backward recompute does a fresh pinned→GPU H2D
            //     per block instead of using captured handles. Slower but
            //     fits in 24 GB GPU.
            //
            // Outer prefetch primes block 0. Subsequent prefetches happen
            // after each block forward returns (so the next block's H2D
            // overlaps with the autograd bookkeeping for this one).
            {
                let mut g = offloader_arc.lock()
                    .map_err(|e| flame_core::Error::InvalidInput(format!("offloader lock: {e}")))?;
                g.prefetch_block(0)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }

            // Two paths:
            //   * Inference (`!is_recording()`): mirror inference-flame's
            //     loop directly. `await_block` returns an `Arc<HashMap>` that
            //     we pass straight into `dual_stream_block_iflame`; no
            //     GPU-to-GPU `clone_block_weights`, no `Tensor::cat → narrow`
            //     post-block round-trip. Slot reuse is gated by the
            //     BlockOffloader's existing event tracking.
            //   * Training (autograd recording): keep the legacy
            //     `checkpoint_offload + clone_block_weights + cat→narrow`
            //     path so the activation-offload pool and gradient
            //     bookkeeping stay byte-identical to pre-port. Routes through
            //     the **legacy** `dual_stream_block_standalone` for now — the
            //     new fused-kernel forward needs a separate training-parity
            //     pass before the train binary cuts over.
            let is_inference = !flame_core::autograd::AutogradContext::is_recording();

            for i in 0..n_blocks {
                if is_inference {
                    // Inference fast path — bisect bug: route through the
                    // verbatim iflame port (`dual_stream_block_iflame`).
                    let raw = {
                        let mut g = offloader_arc.lock()
                            .map_err(|e| flame_core::Error::InvalidInput(
                                format!("offloader lock (block {i}): {e}")
                            ))?;
                        g.await_block(i)
                            .map_err(|e| flame_core::Error::InvalidInput(
                                format!("await block {i}: {e}")
                            ))?
                    };
                    if i + 1 < n_blocks {
                        let mut g = offloader_arc.lock()
                            .map_err(|e| flame_core::Error::InvalidInput(
                                format!("offloader lock: {e}")
                            ))?;
                        g.prefetch_block(i + 1)
                            .map_err(|e| flame_core::Error::InvalidInput(
                                format!("prefetch {}: {e}", i + 1)
                            ))?;
                    }
                    self.bundle.refresh_caches();
                    let (new_img, new_txt) = dual_stream_block_iflame(
                        &img, &txt, &temb, i,
                        &pe_cos, &pe_sin,
                        &img_cos, &img_sin, &txt_cos, &txt_sin,
                        &raw, &self.bundle,
                    )?;
                    drop(raw);
                    img = new_img;
                    txt = new_txt;
                } else {
                    // Training path — unchanged from pre-port behavior.
                    let img_c = img.clone();
                    let txt_c = txt.clone();
                    let temb_c = temb.clone();
                    let ic = img_cos.clone();
                    let is_ = img_sin.clone();
                    let tc = txt_cos.clone();
                    let ts = txt_sin.clone();
                    let bundle_c = self.bundle.clone();
                    let off_clone = offloader_arc.clone();

                    let block_out = flame_core::autograd::AutogradContext::checkpoint_offload(
                        &[img_c.clone(), txt_c.clone()],
                        move || {
                            let raw = {
                                let mut g = off_clone.lock()
                                    .map_err(|e| flame_core::Error::InvalidInput(
                                        format!("offloader lock (block {i}): {e}")
                                    ))?;
                                g.await_block(i)
                                    .map_err(|e| flame_core::Error::InvalidInput(
                                        format!("await block {i}: {e}")
                                    ))?
                            };
                            let weights = clone_block_weights(&raw)?;
                            drop(raw);

                            bundle_c.refresh_caches();
                            let (new_img, new_txt) = dual_stream_block_standalone(
                                &img_c, &txt_c, &temb_c, i,
                                &ic, &is_, &tc, &ts,
                                &weights, &bundle_c,
                            )?;
                            Tensor::cat(&[&new_img, &new_txt], 1)
                        },
                    )?;

                    if i + 1 < n_blocks {
                        let mut g = offloader_arc.lock()
                            .map_err(|e| flame_core::Error::InvalidInput(
                                format!("offloader lock: {e}")
                            ))?;
                        g.prefetch_block(i + 1)
                            .map_err(|e| flame_core::Error::InvalidInput(
                                format!("prefetch {}: {e}", i + 1)
                            ))?;
                    }

                    let img_seq_len = img.shape().dims()[1];
                    let txt_seq_len = txt.shape().dims()[1];
                    img = block_out.narrow(1, 0, img_seq_len)?;
                    txt = block_out.narrow(1, img_seq_len, txt_seq_len)?;
                }

                if i % 10 == 0 || i == n_blocks - 1 {
                    log::info!("[qwenimage-trainer] block {}/{n_blocks}", i + 1);
                }
            }
        } else {
            // Resident path: all block weights already on GPU.
            for i in 0..n_blocks {
                let (new_img, new_txt) = self.dual_stream_block(
                    &img, &txt, &temb, i,
                    &img_cos, &img_sin, &txt_cos, &txt_sin,
                )?;
                img = new_img;
                txt = new_txt;
            }
        }

        // Output: AdaLayerNormContinuous → proj_out
        // norm_out: SiLU(temb) → Linear(dim, 2*dim) → chunk → LN(x) * (1+scale) + shift
        // Reference: qwen_image_model.py:547-548:
        //   scale, shift = torch.chunk(emb, 2, dim=1)
        let norm_emb = temb.silu()?;
        let norm_out = self.fused_linear(&norm_emb,
            self.w("norm_out.linear.weight")?, Some(self.w("norm_out.linear.bias")?))?;
        let chunks = norm_out.unsqueeze(1)?.chunk(2, 2)?;
        let scale = &chunks[0];
        let shift = &chunks[1];

        let hidden = img.shape().dims()[2];
        let img_norm = flame_core::layer_norm::layer_norm(&img, &[hidden], None, None, NORM_EPS)?;
        let img_out = img_norm.mul(&scale.add_scalar(1.0)?)?.add(shift)?;

        // proj_out: [B, seq, 3072] → [B, seq, 64]
        self.matmul_bias(&img_out, self.w("proj_out.weight")?, Some(self.w("proj_out.bias")?))
    }

    /// Edit-mode forward (Qwen-Image-Edit-2509 / 2511).
    ///
    /// `packed_noisy`: [B, target_seq, 64] — noisy target (flow-matched)
    /// `control_packed`: [B, sum_control_seq, 64] — clean source latents, packed and
    ///                   concatenated along seq for multiple controls.
    /// `regions`: `(height, width)` per region in patch units (i.e. `lat_h/2`).
    ///   `regions[0] = target`, `regions[1..] = controls` (in the same order).
    /// `timestep`, `txt_embed`: as T2I.
    ///
    /// Returns **only the target portion** of the prediction as
    /// `[B, target_seq, 64]`. The control portion is dropped internally to
    /// match musubi's `model_pred = model_pred[:, :img_seq_len]` convention.
    pub fn forward_edit(
        &mut self,
        packed_noisy: &Tensor,
        control_packed: &Tensor,
        timestep: &Tensor,
        txt_embed: &Tensor,
        regions: &[(usize, usize)],
    ) -> Result<Tensor> {
        self.forward_edit_inner(packed_noisy, control_packed, timestep, txt_embed, regions, false)
    }

    /// Edit-2511 variant of `forward_edit` — applies the `zero_cond_t`
    /// per-region modulation split that musubi-tuner's
    /// `qwen_image_model.py` enables when `zero_cond_t=True`. Image
    /// modulation params are computed from a doubled timestep `[t; t*0]`
    /// so that target positions see the real timestep and control
    /// positions are conditioned on `t=0`.
    ///
    /// NOTE: needs runtime validation against musubi reference outputs;
    /// this code path has not yet been exercised on a GPU.
    pub fn forward_edit_2511(
        &mut self,
        packed_noisy: &Tensor,
        control_packed: &Tensor,
        timestep: &Tensor,
        txt_embed: &Tensor,
        regions: &[(usize, usize)],
    ) -> Result<Tensor> {
        self.forward_edit_inner(packed_noisy, control_packed, timestep, txt_embed, regions, true)
    }

    fn forward_edit_inner(
        &mut self,
        packed_noisy: &Tensor,
        control_packed: &Tensor,
        timestep: &Tensor,
        txt_embed: &Tensor,
        regions: &[(usize, usize)],
        zero_cond_t: bool,
    ) -> Result<Tensor> {
        if regions.len() < 2 {
            return Err(flame_core::Error::InvalidInput(format!(
                "forward_edit: expected >=2 regions (target + >=1 control), got {}",
                regions.len(),
            )));
        }
        let target_seq = regions[0].0 * regions[0].1;
        let control_seq: usize = regions[1..].iter().map(|(h, w)| h * w).sum();
        let target_dims = packed_noisy.shape().dims();
        let control_dims = control_packed.shape().dims();
        if target_dims[1] != target_seq {
            return Err(flame_core::Error::InvalidInput(format!(
                "forward_edit: packed_noisy seq {} != regions[0] {}*{}={}",
                target_dims[1], regions[0].0, regions[0].1, target_seq,
            )));
        }
        if control_dims[1] != control_seq {
            return Err(flame_core::Error::InvalidInput(format!(
                "forward_edit: control_packed seq {} != sum(regions[1..]) {}",
                control_dims[1], control_seq,
            )));
        }

        // Concatenate [target, controls] along seq dim. Same convention as
        // musubi (qwen_image_train_network.py:492): target first, controls after.
        let packed_all = Tensor::cat(&[packed_noisy, control_packed], 1)?;

        let b = target_dims[0];
        let txt_seq = txt_embed.shape().dims()[1];

        // Multi-region image RoPE covering all regions in order.
        let (img_cos, img_sin) = compute_image_rope_multi(regions, ROPE_THETA)?;
        // Text RoPE with max-over-regions offset.
        let (txt_cos, txt_sin) = compute_text_rope_multi(txt_seq, regions, ROPE_THETA)?;

        // Timestep embedding. Edit-2511 (`zero_cond_t=True`) doubles the
        // input timestep into `[t; t*0]` BEFORE the time embedder so that
        // image modulation params come out at `[2*B, dim]`. We still keep
        // the un-doubled `temb_base` around because txt modulation and the
        // final norm_out only use the base half.
        let timestep_for_img = if zero_cond_t {
            let zeros = timestep.mul_scalar(0.0)?;
            Tensor::cat(&[timestep, &zeros], 0)?
        } else {
            timestep.clone()
        };
        let img_temb = sinusoidal_embedding(&timestep_for_img, 256)?;
        let img_temb = self.fused_linear(&img_temb, self.w("time_text_embed.timestep_embedder.linear_1.weight")?,
            Some(self.w("time_text_embed.timestep_embedder.linear_1.bias")?))?;
        let img_temb = img_temb.silu()?;
        let img_temb = self.fused_linear(&img_temb, self.w("time_text_embed.timestep_embedder.linear_2.weight")?,
            Some(self.w("time_text_embed.timestep_embedder.linear_2.bias")?))?;
        // base temb (single batch) — for txt modulation and final norm_out.
        let temb_base = if zero_cond_t {
            img_temb.narrow(0, 0, target_dims[0])?
        } else {
            img_temb.clone()
        };

        // Image input projection over the FULL [target+control] seq.
        let mut img = self.fused_linear(&packed_all, self.w("img_in.weight")?, Some(self.w("img_in.bias")?))?;
        let txt_normed = flame_core::norm::rms_norm(txt_embed, &[JOINT_DIM], Some(self.w("txt_norm.weight")?), NORM_EPS)?;
        let mut txt = self.fused_linear(&txt_normed, self.w("txt_in.weight")?, Some(self.w("txt_in.bias")?))?;

        // Block loop — SAME kernels as T2I. Edit-mode does not change block ops,
        // only the seq length and RoPE table. Both offloader and resident paths
        // work transparently because dual_stream_block is length-agnostic.
        let n_blocks = if self.offloader.is_some() {
            NUM_LAYERS
        } else {
            std::env::var("QWENIMAGE_MAX_BLOCKS")
                .ok().and_then(|s| s.parse::<usize>().ok())
                .map(|n| n.min(NUM_LAYERS)).unwrap_or(NUM_LAYERS)
        };

        if let Some(ref offloader_arc) = self.offloader {
            if zero_cond_t {
                return Err(flame_core::Error::InvalidInput(
                    "edit-2511 (zero_cond_t) is not yet wired through the BlockOffloader path; \
                     run without --block-swap or run forward_edit_2511 on the resident path.".into(),
                ));
            }
            // Same closure-fetch-not-capture pattern as `forward()` —
            // prevents the 38 GB closure-capture leak. See `forward()` for
            // the full rationale.
            {
                let mut g = offloader_arc.lock()
                    .map_err(|e| flame_core::Error::InvalidInput(format!("offloader lock: {e}")))?;
                g.prefetch_block(0)
                    .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch: {e}")))?;
            }
            for i in 0..n_blocks {
                let img_c = img.clone();
                let txt_c = txt.clone();
                let temb_c = img_temb.clone();
                let ic = img_cos.clone();
                let is_ = img_sin.clone();
                let tc = txt_cos.clone();
                let ts = txt_sin.clone();
                let bundle_c = self.bundle.clone();
                let off_clone = offloader_arc.clone();

                let block_out = flame_core::autograd::AutogradContext::checkpoint_offload(
                    &[img_c.clone(), txt_c.clone()],
                    move || {
                        let raw = {
                            let mut g = off_clone.lock()
                                .map_err(|e| flame_core::Error::InvalidInput(
                                    format!("offloader lock (block {i}): {e}")
                                ))?;
                            g.await_block(i)
                                .map_err(|e| flame_core::Error::InvalidInput(
                                    format!("await block {i}: {e}")
                                ))?
                        };
                        let weights = clone_block_weights(&raw)?;
                        drop(raw);

                        bundle_c.refresh_caches();
                        let (new_img, new_txt) = dual_stream_block_standalone(
                            &img_c, &txt_c, &temb_c, i,
                            &ic, &is_, &tc, &ts,
                            &weights, &bundle_c,
                        )?;
                        Tensor::cat(&[&new_img, &new_txt], 1)
                    },
                )?;

                if i + 1 < n_blocks {
                    let mut g = offloader_arc.lock()
                        .map_err(|e| flame_core::Error::InvalidInput(format!("offloader lock: {e}")))?;
                    g.prefetch_block(i + 1)
                        .map_err(|e| flame_core::Error::InvalidInput(format!("prefetch {}: {e}", i+1)))?;
                }

                let img_seq_len = img.shape().dims()[1];
                let txt_seq_len = txt.shape().dims()[1];
                img = block_out.narrow(1, 0, img_seq_len)?;
                txt = block_out.narrow(1, img_seq_len, txt_seq_len)?;

                if i % 10 == 0 || i == n_blocks - 1 {
                    log::info!("[qwenimage-edit] block {}/{n_blocks}", i + 1);
                }
            }
        } else {
            for i in 0..n_blocks {
                let (new_img, new_txt) = if zero_cond_t {
                    self.dual_stream_block_2511(
                        &img, &txt, &img_temb, &temb_base, i,
                        &img_cos, &img_sin, &txt_cos, &txt_sin,
                        target_seq,
                    )?
                } else {
                    self.dual_stream_block(
                        &img, &txt, &img_temb, i,
                        &img_cos, &img_sin, &txt_cos, &txt_sin,
                    )?
                };
                img = new_img;
                txt = new_txt;
            }
        }

        // AdaLayerNormContinuous → proj_out on the FULL seq. Always use the
        // base (un-doubled) temb here — musubi's `qwen_image_model.py:1361`
        // does `temb = temb.chunk(2, dim=0)[0]` before final norm.
        let norm_emb = temb_base.silu()?;
        let norm_out = self.fused_linear(&norm_emb,
            self.w("norm_out.linear.weight")?, Some(self.w("norm_out.linear.bias")?))?;
        let chunks = norm_out.unsqueeze(1)?.chunk(2, 2)?;
        let scale = &chunks[0];
        let shift = &chunks[1];
        let hidden = img.shape().dims()[2];
        let img_norm = flame_core::layer_norm::layer_norm(&img, &[hidden], None, None, NORM_EPS)?;
        let img_out = img_norm.mul(&scale.add_scalar(1.0)?)?.add(shift)?;
        let pred_full = self.matmul_bias(&img_out, self.w("proj_out.weight")?, Some(self.w("proj_out.bias")?))?;

        // Slice target portion only. Matches `noise_pred = noise_pred[:, :img_seq_len]`
        // in musubi-tuner `qwen_image_train_network.py:567`.
        let _ = b; // silence unused
        pred_full.narrow(1, 0, target_seq)
    }

    fn dual_stream_block(
        &self,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        block_idx: usize,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (b, img_seq, _) = {
            let d = img.shape().dims();
            (d[0], d[1], d[2])
        };
        let txt_seq = txt.shape().dims()[1];

        // Image modulation: SiLU(temb) → Linear → 6 params
        let img_mod = temb.silu()?;
        let img_mod = self.matmul_bias(&img_mod,
            self.bw(block_idx, "img_mod.1.weight")?,
            Some(self.bw(block_idx, "img_mod.1.bias")?))?;
        let img_chunks = img_mod.unsqueeze(1)?.chunk(6, 2)?;

        // Text modulation
        let txt_mod = temb.silu()?;
        let txt_mod = self.matmul_bias(&txt_mod,
            self.bw(block_idx, "txt_mod.1.weight")?,
            Some(self.bw(block_idx, "txt_mod.1.bias")?))?;
        let txt_chunks = txt_mod.unsqueeze(1)?.chunk(6, 2)?;

        // Pre-norm (parameter-free LayerNorm) + modulate
        let img_norm1 = flame_core::layer_norm::layer_norm(img, &[DIM], None, None, NORM_EPS)?;
        let img_normed = img_norm1.mul(&img_chunks[1].add_scalar(1.0)?)?.add(&img_chunks[0])?;

        let txt_norm1 = flame_core::layer_norm::layer_norm(txt, &[DIM], None, None, NORM_EPS)?;
        let txt_normed = txt_norm1.mul(&txt_chunks[1].add_scalar(1.0)?)?.add(&txt_chunks[0])?;

        // Q/K/V projections with LoRA
        let img_q = self.add_lora(self.matmul_bias(&img_normed, self.bw(block_idx, "attn.to_q.weight")?, Some(self.bw(block_idx, "attn.to_q.bias")?))?, &img_normed, block_idx, LoraTarget::ImgQ)?;
        let img_k = self.add_lora(self.matmul_bias(&img_normed, self.bw(block_idx, "attn.to_k.weight")?, Some(self.bw(block_idx, "attn.to_k.bias")?))?, &img_normed, block_idx, LoraTarget::ImgK)?;
        let img_v = self.add_lora(self.matmul_bias(&img_normed, self.bw(block_idx, "attn.to_v.weight")?, Some(self.bw(block_idx, "attn.to_v.bias")?))?, &img_normed, block_idx, LoraTarget::ImgV)?;

        let txt_q = self.add_lora(self.matmul_bias(&txt_normed, self.bw(block_idx, "attn.add_q_proj.weight")?, Some(self.bw(block_idx, "attn.add_q_proj.bias")?))?, &txt_normed, block_idx, LoraTarget::TxtQ)?;
        let txt_k = self.add_lora(self.matmul_bias(&txt_normed, self.bw(block_idx, "attn.add_k_proj.weight")?, Some(self.bw(block_idx, "attn.add_k_proj.bias")?))?, &txt_normed, block_idx, LoraTarget::TxtK)?;
        let txt_v = self.add_lora(self.matmul_bias(&txt_normed, self.bw(block_idx, "attn.add_v_proj.weight")?, Some(self.bw(block_idx, "attn.add_v_proj.bias")?))?, &txt_normed, block_idx, LoraTarget::TxtV)?;

        // QK norm
        let img_q = rms_norm_per_head(&img_q, self.bw(block_idx, "attn.norm_q.weight")?, NUM_HEADS, HEAD_DIM)?;
        let img_k = rms_norm_per_head(&img_k, self.bw(block_idx, "attn.norm_k.weight")?, NUM_HEADS, HEAD_DIM)?;
        let txt_q = rms_norm_per_head(&txt_q, self.bw(block_idx, "attn.norm_added_q.weight")?, NUM_HEADS, HEAD_DIM)?;
        let txt_k = rms_norm_per_head(&txt_k, self.bw(block_idx, "attn.norm_added_k.weight")?, NUM_HEADS, HEAD_DIM)?;

        // Apply RoPE to img and txt Q/K separately, then concatenate
        let img_q_4d = img_q.reshape(&[b, img_seq, NUM_HEADS, HEAD_DIM])?;
        let img_k_4d = img_k.reshape(&[b, img_seq, NUM_HEADS, HEAD_DIM])?;
        let txt_q_4d = txt_q.reshape(&[b, txt_seq, NUM_HEADS, HEAD_DIM])?;
        let txt_k_4d = txt_k.reshape(&[b, txt_seq, NUM_HEADS, HEAD_DIM])?;

        let img_q_rope = apply_rope(&img_q_4d, img_cos, img_sin)?;
        let img_k_rope = apply_rope(&img_k_4d, img_cos, img_sin)?;
        let txt_q_rope = apply_rope(&txt_q_4d, txt_cos, txt_sin)?;
        let txt_k_rope = apply_rope(&txt_k_4d, txt_cos, txt_sin)?;

        // Joint attention: cat img+txt → [B, H, S, D], SDPA, split
        let total_seq = img_seq + txt_seq;
        let q = Tensor::cat(&[&img_q_rope, &txt_q_rope], 1)?.permute(&[0, 2, 1, 3])?;
        let k = Tensor::cat(&[&img_k_rope, &txt_k_rope], 1)?.permute(&[0, 2, 1, 3])?;
        let v = {
            let img_v_4d = img_v.reshape(&[b, img_seq, NUM_HEADS, HEAD_DIM])?;
            let txt_v_4d = txt_v.reshape(&[b, txt_seq, NUM_HEADS, HEAD_DIM])?;
            Tensor::cat(&[&img_v_4d, &txt_v_4d], 1)?.permute(&[0, 2, 1, 3])?
        };

        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, total_seq, DIM])?;

        // Split img/txt
        let img_attn = attn_out.narrow(1, 0, img_seq)?;
        let txt_attn = attn_out.narrow(1, img_seq, txt_seq)?;

        // Output projections with LoRA
        let img_attn_out = self.add_lora(
            self.matmul_bias(&img_attn, self.bw(block_idx, "attn.to_out.0.weight")?, Some(self.bw(block_idx, "attn.to_out.0.bias")?))?,
            &img_attn, block_idx, LoraTarget::ImgOut)?;
        let txt_attn_out = self.add_lora(
            self.matmul_bias(&txt_attn, self.bw(block_idx, "attn.to_add_out.weight")?, Some(self.bw(block_idx, "attn.to_add_out.bias")?))?,
            &txt_attn, block_idx, LoraTarget::TxtOut)?;

        // Gated residual
        let img = img.add(&img_chunks[2].mul(&img_attn_out)?)?;
        let txt = txt.add(&txt_chunks[2].mul(&txt_attn_out)?)?;

        // FFN with LoRA
        let img_norm2 = flame_core::layer_norm::layer_norm(&img, &[DIM], None, None, NORM_EPS)?;
        let img_ffn_in = img_norm2.mul(&img_chunks[4].add_scalar(1.0)?)?.add(&img_chunks[3])?;
        let img_ffn_up = self.add_lora(
            self.matmul_bias(&img_ffn_in, self.bw(block_idx, "img_mlp.net.0.proj.weight")?, Some(self.bw(block_idx, "img_mlp.net.0.proj.bias")?))?,
            &img_ffn_in, block_idx, LoraTarget::ImgFfnUp)?;
        let img_ffn_act = img_ffn_up.gelu()?;
        let img_ffn_down = self.add_lora(
            self.matmul_bias(&img_ffn_act, self.bw(block_idx, "img_mlp.net.2.weight")?, Some(self.bw(block_idx, "img_mlp.net.2.bias")?))?,
            &img_ffn_act, block_idx, LoraTarget::ImgFfnDown)?;
        let img = img.add(&img_chunks[5].mul(&img_ffn_down)?)?;

        let txt_norm2 = flame_core::layer_norm::layer_norm(&txt, &[DIM], None, None, NORM_EPS)?;
        let txt_ffn_in = txt_norm2.mul(&txt_chunks[4].add_scalar(1.0)?)?.add(&txt_chunks[3])?;
        let txt_ffn_up = self.add_lora(
            self.matmul_bias(&txt_ffn_in, self.bw(block_idx, "txt_mlp.net.0.proj.weight")?, Some(self.bw(block_idx, "txt_mlp.net.0.proj.bias")?))?,
            &txt_ffn_in, block_idx, LoraTarget::TxtFfnUp)?;
        let txt_ffn_act = txt_ffn_up.gelu()?;
        let txt_ffn_down = self.add_lora(
            self.matmul_bias(&txt_ffn_act, self.bw(block_idx, "txt_mlp.net.2.weight")?, Some(self.bw(block_idx, "txt_mlp.net.2.bias")?))?,
            &txt_ffn_act, block_idx, LoraTarget::TxtFfnDown)?;
        let txt = txt.add(&txt_chunks[5].mul(&txt_ffn_down)?)?;

        Ok((img, txt))
    }

    /// Edit-2511 dual-stream block — per-region (target vs control) image
    /// modulation driven by a doubled timestep `[t; t*0]`.
    ///
    /// Mirrors musubi-tuner's `qwen_image_model.py:975-1095` with
    /// `zero_cond_t=True`:
    ///
    /// - `img_temb` has batch dim `2*B` (`[t; t*0]`) and feeds `img_mod` so
    ///   the modulation params come out at `[2*B, 6*dim]`.
    /// - The chunked params split into `base` (first `B`) and `ext` (last
    ///   `B`); positions `[0, target_seq)` use base, `[target_seq, ..)` use
    ///   ext (the t=0 path), so controls "see" timestep zero.
    /// - `txt_mod` only sees the un-doubled `temb` (single batch).
    fn dual_stream_block_2511(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_temb: &Tensor,        // [2*B, dim]
        txt_temb: &Tensor,        // [B, dim]
        block_idx: usize,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        target_seq: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b, img_seq, _) = {
            let d = img.shape().dims();
            (d[0], d[1], d[2])
        };
        let txt_seq = txt.shape().dims()[1];
        if target_seq > img_seq {
            return Err(flame_core::Error::InvalidInput(format!(
                "dual_stream_block_2511: target_seq {target_seq} > img_seq {img_seq}",
            )));
        }
        let ext_seq = img_seq - target_seq;

        // Image modulation on the doubled batch.
        let img_mod = img_temb.silu()?;
        let img_mod = self.matmul_bias(&img_mod,
            self.bw(block_idx, "img_mod.1.weight")?,
            Some(self.bw(block_idx, "img_mod.1.bias")?))?;
        // [2*B, 6*dim] -> [2*B, 1, 6*dim] -> chunk(6) along last dim.
        let img_chunks = img_mod.unsqueeze(1)?.chunk(6, 2)?;

        // Text modulation on the base-only batch.
        let txt_mod = txt_temb.silu()?;
        let txt_mod = self.matmul_bias(&txt_mod,
            self.bw(block_idx, "txt_mod.1.weight")?,
            Some(self.bw(block_idx, "txt_mod.1.bias")?))?;
        let txt_chunks = txt_mod.unsqueeze(1)?.chunk(6, 2)?;

        // Per-region modulate helper: x is [B, img_seq, dim]; chunk is
        // [2*B, 1, dim]. Splits chunk into base/ext halves on batch dim and
        // applies different shift/scale to the two image regions.
        let modulate_split = |x: &Tensor, scale_chunk: &Tensor, shift_chunk: &Tensor| -> Result<Tensor> {
            let s_base = scale_chunk.narrow(0, 0, b)?;
            let s_ext = scale_chunk.narrow(0, b, b)?;
            let h_base = shift_chunk.narrow(0, 0, b)?;
            let h_ext = shift_chunk.narrow(0, b, b)?;
            let x_base = x.narrow(1, 0, target_seq)?;
            let part_base = x_base.mul(&s_base.add_scalar(1.0)?)?.add(&h_base)?;
            if ext_seq == 0 {
                return Ok(part_base);
            }
            let x_ext = x.narrow(1, target_seq, ext_seq)?;
            let part_ext = x_ext.mul(&s_ext.add_scalar(1.0)?)?.add(&h_ext)?;
            Tensor::cat(&[&part_base, &part_ext], 1)
        };
        // Per-region gate helper: gate_chunk is [2*B, 1, dim]; broadcast
        // base across the target seq, ext across the rest.
        let gate_split = |out: &Tensor, gate_chunk: &Tensor| -> Result<Tensor> {
            let g_base = gate_chunk.narrow(0, 0, b)?;
            let g_ext = gate_chunk.narrow(0, b, b)?;
            let out_base = out.narrow(1, 0, target_seq)?;
            let part_base = g_base.mul(&out_base)?;
            if ext_seq == 0 {
                return Ok(part_base);
            }
            let out_ext = out.narrow(1, target_seq, ext_seq)?;
            let part_ext = g_ext.mul(&out_ext)?;
            Tensor::cat(&[&part_base, &part_ext], 1)
        };

        // Pre-norm + per-region modulate (img). Txt stays unsplit.
        let img_norm1 = flame_core::layer_norm::layer_norm(img, &[DIM], None, None, NORM_EPS)?;
        let img_normed = modulate_split(&img_norm1, &img_chunks[1], &img_chunks[0])?;

        let txt_norm1 = flame_core::layer_norm::layer_norm(txt, &[DIM], None, None, NORM_EPS)?;
        let txt_normed = txt_norm1.mul(&txt_chunks[1].add_scalar(1.0)?)?.add(&txt_chunks[0])?;

        // Q/K/V projections with LoRA — identical to T2I path.
        let img_q = self.add_lora(self.matmul_bias(&img_normed, self.bw(block_idx, "attn.to_q.weight")?, Some(self.bw(block_idx, "attn.to_q.bias")?))?, &img_normed, block_idx, LoraTarget::ImgQ)?;
        let img_k = self.add_lora(self.matmul_bias(&img_normed, self.bw(block_idx, "attn.to_k.weight")?, Some(self.bw(block_idx, "attn.to_k.bias")?))?, &img_normed, block_idx, LoraTarget::ImgK)?;
        let img_v = self.add_lora(self.matmul_bias(&img_normed, self.bw(block_idx, "attn.to_v.weight")?, Some(self.bw(block_idx, "attn.to_v.bias")?))?, &img_normed, block_idx, LoraTarget::ImgV)?;
        let txt_q = self.add_lora(self.matmul_bias(&txt_normed, self.bw(block_idx, "attn.add_q_proj.weight")?, Some(self.bw(block_idx, "attn.add_q_proj.bias")?))?, &txt_normed, block_idx, LoraTarget::TxtQ)?;
        let txt_k = self.add_lora(self.matmul_bias(&txt_normed, self.bw(block_idx, "attn.add_k_proj.weight")?, Some(self.bw(block_idx, "attn.add_k_proj.bias")?))?, &txt_normed, block_idx, LoraTarget::TxtK)?;
        let txt_v = self.add_lora(self.matmul_bias(&txt_normed, self.bw(block_idx, "attn.add_v_proj.weight")?, Some(self.bw(block_idx, "attn.add_v_proj.bias")?))?, &txt_normed, block_idx, LoraTarget::TxtV)?;

        // QK norm
        let img_q = rms_norm_per_head(&img_q, self.bw(block_idx, "attn.norm_q.weight")?, NUM_HEADS, HEAD_DIM)?;
        let img_k = rms_norm_per_head(&img_k, self.bw(block_idx, "attn.norm_k.weight")?, NUM_HEADS, HEAD_DIM)?;
        let txt_q = rms_norm_per_head(&txt_q, self.bw(block_idx, "attn.norm_added_q.weight")?, NUM_HEADS, HEAD_DIM)?;
        let txt_k = rms_norm_per_head(&txt_k, self.bw(block_idx, "attn.norm_added_k.weight")?, NUM_HEADS, HEAD_DIM)?;

        // Apply RoPE
        let img_q_4d = img_q.reshape(&[b, img_seq, NUM_HEADS, HEAD_DIM])?;
        let img_k_4d = img_k.reshape(&[b, img_seq, NUM_HEADS, HEAD_DIM])?;
        let txt_q_4d = txt_q.reshape(&[b, txt_seq, NUM_HEADS, HEAD_DIM])?;
        let txt_k_4d = txt_k.reshape(&[b, txt_seq, NUM_HEADS, HEAD_DIM])?;

        let img_q_rope = apply_rope(&img_q_4d, img_cos, img_sin)?;
        let img_k_rope = apply_rope(&img_k_4d, img_cos, img_sin)?;
        let txt_q_rope = apply_rope(&txt_q_4d, txt_cos, txt_sin)?;
        let txt_k_rope = apply_rope(&txt_k_4d, txt_cos, txt_sin)?;

        // Joint attention
        let total_seq = img_seq + txt_seq;
        let q = Tensor::cat(&[&img_q_rope, &txt_q_rope], 1)?.permute(&[0, 2, 1, 3])?;
        let k = Tensor::cat(&[&img_k_rope, &txt_k_rope], 1)?.permute(&[0, 2, 1, 3])?;
        let v = {
            let img_v_4d = img_v.reshape(&[b, img_seq, NUM_HEADS, HEAD_DIM])?;
            let txt_v_4d = txt_v.reshape(&[b, txt_seq, NUM_HEADS, HEAD_DIM])?;
            Tensor::cat(&[&img_v_4d, &txt_v_4d], 1)?.permute(&[0, 2, 1, 3])?
        };

        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, total_seq, DIM])?;

        let img_attn = attn_out.narrow(1, 0, img_seq)?;
        let txt_attn = attn_out.narrow(1, img_seq, txt_seq)?;

        let img_attn_out = self.add_lora(
            self.matmul_bias(&img_attn, self.bw(block_idx, "attn.to_out.0.weight")?, Some(self.bw(block_idx, "attn.to_out.0.bias")?))?,
            &img_attn, block_idx, LoraTarget::ImgOut)?;
        let txt_attn_out = self.add_lora(
            self.matmul_bias(&txt_attn, self.bw(block_idx, "attn.to_add_out.weight")?, Some(self.bw(block_idx, "attn.to_add_out.bias")?))?,
            &txt_attn, block_idx, LoraTarget::TxtOut)?;

        // Per-region gated residual (img); txt unchanged.
        let img_gate1 = gate_split(&img_attn_out, &img_chunks[2])?;
        let img = img.add(&img_gate1)?;
        let txt = txt.add(&txt_chunks[2].mul(&txt_attn_out)?)?;

        // FFN with per-region modulation on img norm2; txt normal.
        let img_norm2 = flame_core::layer_norm::layer_norm(&img, &[DIM], None, None, NORM_EPS)?;
        let img_ffn_in = modulate_split(&img_norm2, &img_chunks[4], &img_chunks[3])?;
        let img_ffn_up = self.add_lora(
            self.matmul_bias(&img_ffn_in, self.bw(block_idx, "img_mlp.net.0.proj.weight")?, Some(self.bw(block_idx, "img_mlp.net.0.proj.bias")?))?,
            &img_ffn_in, block_idx, LoraTarget::ImgFfnUp)?;
        let img_ffn_act = img_ffn_up.gelu()?;
        let img_ffn_down = self.add_lora(
            self.matmul_bias(&img_ffn_act, self.bw(block_idx, "img_mlp.net.2.weight")?, Some(self.bw(block_idx, "img_mlp.net.2.bias")?))?,
            &img_ffn_act, block_idx, LoraTarget::ImgFfnDown)?;
        let img_gate2 = gate_split(&img_ffn_down, &img_chunks[5])?;
        let img = img.add(&img_gate2)?;

        let txt_norm2 = flame_core::layer_norm::layer_norm(&txt, &[DIM], None, None, NORM_EPS)?;
        let txt_ffn_in = txt_norm2.mul(&txt_chunks[4].add_scalar(1.0)?)?.add(&txt_chunks[0 + 3])?;
        let txt_ffn_up = self.add_lora(
            self.matmul_bias(&txt_ffn_in, self.bw(block_idx, "txt_mlp.net.0.proj.weight")?, Some(self.bw(block_idx, "txt_mlp.net.0.proj.bias")?))?,
            &txt_ffn_in, block_idx, LoraTarget::TxtFfnUp)?;
        let txt_ffn_act = txt_ffn_up.gelu()?;
        let txt_ffn_down = self.add_lora(
            self.matmul_bias(&txt_ffn_act, self.bw(block_idx, "txt_mlp.net.2.weight")?, Some(self.bw(block_idx, "txt_mlp.net.2.bias")?))?,
            &txt_ffn_act, block_idx, LoraTarget::TxtFfnDown)?;
        let txt = txt.add(&txt_chunks[5].mul(&txt_ffn_down)?)?;

        Ok((img, txt))
    }
}

// ---------------------------------------------------------------------------
// Public helpers for parity testing
// ---------------------------------------------------------------------------

/// Public wrapper for sinusoidal_embedding (used by parity test).
pub fn sinusoidal_embedding_pub(t: &Tensor) -> Result<Tensor> {
    sinusoidal_embedding(t, 256)
}

impl QwenImageTrainingModel {
    /// Public access to a resident weight tensor (used by parity test).
    pub fn w_pub(&self, key: &str) -> Result<&Tensor> { self.w(key) }
    /// Public access to fused_linear (used by parity test).
    pub fn fused_linear_pub(&self, x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
        self.fused_linear(x, w, b)
    }
    /// Forward pass returning per-block (img, txt) intermediates for parity testing.
    pub fn forward_with_intermediates(
        &mut self,
        packed_noisy: &Tensor,
        timestep: &Tensor,
        txt_embed: &Tensor,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let pack_h = latent_h / 2;
        let pack_w = latent_w / 2;
        let txt_seq = txt_embed.shape().dims()[1];
        let (img_cos, img_sin) = compute_image_rope(pack_h, pack_w, ROPE_THETA)?;
        let (txt_cos, txt_sin) = compute_text_rope(txt_seq, pack_h, pack_w, ROPE_THETA)?;

        let temb = sinusoidal_embedding(timestep, 256)?;
        let temb = self.fused_linear(&temb, self.w("time_text_embed.timestep_embedder.linear_1.weight")?,
            Some(self.w("time_text_embed.timestep_embedder.linear_1.bias")?))?;
        let temb = temb.silu()?;
        let temb = self.fused_linear(&temb, self.w("time_text_embed.timestep_embedder.linear_2.weight")?,
            Some(self.w("time_text_embed.timestep_embedder.linear_2.bias")?))?;

        let mut img = self.fused_linear(packed_noisy, self.w("img_in.weight")?, Some(self.w("img_in.bias")?))?;
        let txt_normed = flame_core::norm::rms_norm(txt_embed, &[JOINT_DIM], Some(self.w("txt_norm.weight")?), NORM_EPS)?;
        let mut txt = self.fused_linear(&txt_normed, self.w("txt_in.weight")?, Some(self.w("txt_in.bias")?))?;

        let n_blocks = std::env::var("QWENIMAGE_MAX_BLOCKS")
            .ok().and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.min(NUM_LAYERS)).unwrap_or(NUM_LAYERS);

        let mut block_outputs = Vec::new();
        for i in 0..n_blocks {
            let (new_img, new_txt) = self.dual_stream_block(
                &img, &txt, &temb, i, &img_cos, &img_sin, &txt_cos, &txt_sin,
            )?;
            img = new_img;
            txt = new_txt;
            block_outputs.push((img.clone(), txt.clone()));
        }

        let hidden = img.shape().dims()[2];
        let norm_emb = temb.silu()?;
        let norm_out = self.fused_linear(&norm_emb,
            self.w("norm_out.linear.weight")?, Some(self.w("norm_out.linear.bias")?))?;
        let chunks = norm_out.unsqueeze(1)?.chunk(2, 2)?;
        let scale = &chunks[0];
        let shift = &chunks[1];
        let img_norm = flame_core::layer_norm::layer_norm(&img, &[hidden], None, None, NORM_EPS)?;
        let img_out = img_norm.mul(&scale.add_scalar(1.0)?)?.add(shift)?;
        let pred = self.matmul_bias(&img_out, self.w("proj_out.weight")?, Some(self.w("proj_out.bias")?))?;

        Ok((pred, block_outputs))
    }
}

// ---------------------------------------------------------------------------
// Block offload helpers
// ---------------------------------------------------------------------------

/// Clone all tensors from a block weight map — BlockOffloader reuses GPU slots,
/// so we must own the data before the next block is loaded.
fn clone_block_weights(raw: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(raw.len());
    for (key, t) in raw {
        out.insert(key.clone(), t.clone_result()?);
    }
    Ok(out)
}

/// Standalone dual-stream block using explicit weight map (for BlockOffloader path).
/// Keys in `weights` are full paths like `transformer_blocks.{i}.attn.to_q.weight`.
#[allow(clippy::too_many_arguments)]
fn dual_stream_block_standalone(
    img: &Tensor,
    txt: &Tensor,
    temb: &Tensor,
    block_idx: usize,
    img_cos: &Tensor,
    img_sin: &Tensor,
    txt_cos: &Tensor,
    txt_sin: &Tensor,
    weights: &HashMap<String, Tensor>,
    bundle: &QwenImageLoraBundle,
) -> Result<(Tensor, Tensor)> {
    let (b, img_seq, _) = {
        let d = img.shape().dims();
        (d[0], d[1], d[2])
    };
    let txt_seq = txt.shape().dims()[1];

    let bw = |suffix: &str| -> Result<&Tensor> {
        let key = format!("transformer_blocks.{block_idx}.{suffix}");
        weights.get(&key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {key}")))
    };

    let matmul_bias = |x: &Tensor, weight: &Tensor, bias: Option<&Tensor>| -> Result<Tensor> {
        // BlockOffloader::prepare_weights pre-transposes 2D `.weight`
        // tensors at load time (block_offload.rs:944-949). So `weight` here
        // is already laid out as `[in_features, out_features]`, NOT the
        // PyTorch [out, in]. Use it directly — re-transposing would put it
        // back to [out, in] and cause `lhs[..., in] @ wt[out, in]` to fail
        // the matmul k-dim check. Pre-EDv2-salvage flame-diffusion source
        // had `weight.transpose()?` here but never ran (untested code).
        let dims = x.shape().dims().to_vec();
        let in_feat = *dims.last().unwrap();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let out_feat = weight.shape().dims()[1];
        let x_2d = x.reshape(&[batch, in_feat])?;
        let mut out = x_2d.matmul(weight)?;
        if let Some(b) = bias { out = out.add(b)?; }
        let mut out_shape = dims[..dims.len()-1].to_vec();
        out_shape.push(out_feat);
        out.reshape(&out_shape)
    };

    let add_lora = |base: Tensor, input: &Tensor, target: LoraTarget| -> Result<Tensor> {
        if let Some(lora) = bundle.adapters.get(&(block_idx, target)) {
            let input_3d = if input.shape().dims().len() == 2 { input.unsqueeze(0)? } else { input.clone() };
            let delta = lora.forward_delta(&input_3d)?;
            base.add(&delta)
        } else {
            Ok(base)
        }
    };

    // Image modulation
    let img_mod = temb.silu()?;
    let img_mod = matmul_bias(&img_mod, bw("img_mod.1.weight")?, Some(bw("img_mod.1.bias")?))?;
    let img_chunks = img_mod.unsqueeze(1)?.chunk(6, 2)?;

    // Text modulation
    let txt_mod = temb.silu()?;
    let txt_mod = matmul_bias(&txt_mod, bw("txt_mod.1.weight")?, Some(bw("txt_mod.1.bias")?))?;
    let txt_chunks = txt_mod.unsqueeze(1)?.chunk(6, 2)?;

    // Pre-norm + modulate
    let img_norm1 = flame_core::layer_norm::layer_norm(img, &[DIM], None, None, NORM_EPS)?;
    let img_normed = img_norm1.mul(&img_chunks[1].add_scalar(1.0)?)?.add(&img_chunks[0])?;
    let txt_norm1 = flame_core::layer_norm::layer_norm(txt, &[DIM], None, None, NORM_EPS)?;
    let txt_normed = txt_norm1.mul(&txt_chunks[1].add_scalar(1.0)?)?.add(&txt_chunks[0])?;

    // DIAG: bisect where the magnitude blows up
    let diag = block_idx <= 2 && std::env::var("QWEN_DIAG_BISECT").ok().as_deref() == Some("1");
    let dump = |name: &str, t: &Tensor| -> Result<()> {
        if !diag { return Ok(()); }
        let v = t.to_dtype(DType::F32)?.to_vec()?;
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let mean_abs: f32 = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
        log::warn!("[DIAG b{block_idx}] {name}: max={max_abs:.2} mean={mean_abs:.4} numel={}", v.len());
        Ok(())
    };
    dump("img_in", img)?;
    dump("img_norm1", &img_norm1)?;
    dump("img_normed", &img_normed)?;
    dump("img_chunks[1]_scale", &img_chunks[1])?;
    dump("img_chunks[0]_shift", &img_chunks[0])?;

    // Q/K/V with LoRA
    let img_q = add_lora(matmul_bias(&img_normed, bw("attn.to_q.weight")?, Some(bw("attn.to_q.bias")?))?, &img_normed, LoraTarget::ImgQ)?;
    let img_k = add_lora(matmul_bias(&img_normed, bw("attn.to_k.weight")?, Some(bw("attn.to_k.bias")?))?, &img_normed, LoraTarget::ImgK)?;
    let img_v = add_lora(matmul_bias(&img_normed, bw("attn.to_v.weight")?, Some(bw("attn.to_v.bias")?))?, &img_normed, LoraTarget::ImgV)?;
    dump("img_q", &img_q)?;
    dump("img_v", &img_v)?;
    let txt_q = add_lora(matmul_bias(&txt_normed, bw("attn.add_q_proj.weight")?, Some(bw("attn.add_q_proj.bias")?))?, &txt_normed, LoraTarget::TxtQ)?;
    let txt_k = add_lora(matmul_bias(&txt_normed, bw("attn.add_k_proj.weight")?, Some(bw("attn.add_k_proj.bias")?))?, &txt_normed, LoraTarget::TxtK)?;
    let txt_v = add_lora(matmul_bias(&txt_normed, bw("attn.add_v_proj.weight")?, Some(bw("attn.add_v_proj.bias")?))?, &txt_normed, LoraTarget::TxtV)?;

    // QK norm
    let img_q = rms_norm_per_head(&img_q, bw("attn.norm_q.weight")?, NUM_HEADS, HEAD_DIM)?;
    let img_k = rms_norm_per_head(&img_k, bw("attn.norm_k.weight")?, NUM_HEADS, HEAD_DIM)?;
    let txt_q = rms_norm_per_head(&txt_q, bw("attn.norm_added_q.weight")?, NUM_HEADS, HEAD_DIM)?;
    let txt_k = rms_norm_per_head(&txt_k, bw("attn.norm_added_k.weight")?, NUM_HEADS, HEAD_DIM)?;

    // RoPE
    let img_q_4d = img_q.reshape(&[b, img_seq, NUM_HEADS, HEAD_DIM])?;
    let img_k_4d = img_k.reshape(&[b, img_seq, NUM_HEADS, HEAD_DIM])?;
    let txt_q_4d = txt_q.reshape(&[b, txt_seq, NUM_HEADS, HEAD_DIM])?;
    let txt_k_4d = txt_k.reshape(&[b, txt_seq, NUM_HEADS, HEAD_DIM])?;
    let img_q_rope = apply_rope(&img_q_4d, img_cos, img_sin)?;
    let img_k_rope = apply_rope(&img_k_4d, img_cos, img_sin)?;
    let txt_q_rope = apply_rope(&txt_q_4d, txt_cos, txt_sin)?;
    let txt_k_rope = apply_rope(&txt_k_4d, txt_cos, txt_sin)?;

    // Joint attention
    let total_seq = img_seq + txt_seq;
    let q = Tensor::cat(&[&img_q_rope, &txt_q_rope], 1)?.permute(&[0, 2, 1, 3])?;
    let k = Tensor::cat(&[&img_k_rope, &txt_k_rope], 1)?.permute(&[0, 2, 1, 3])?;
    let v = {
        let img_v_4d = img_v.reshape(&[b, img_seq, NUM_HEADS, HEAD_DIM])?;
        let txt_v_4d = txt_v.reshape(&[b, txt_seq, NUM_HEADS, HEAD_DIM])?;
        Tensor::cat(&[&img_v_4d, &txt_v_4d], 1)?.permute(&[0, 2, 1, 3])?
    };
    dump("v", &v)?;
    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    dump("attn_out_pre_perm", &attn_out)?;
    let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, total_seq, DIM])?;
    dump("attn_out_post_reshape", &attn_out)?;

    // Split img/txt
    let img_attn = attn_out.narrow(1, 0, img_seq)?;
    let txt_attn = attn_out.narrow(1, img_seq, txt_seq)?;
    dump("img_attn (post-narrow)", &img_attn)?;

    // Output projections with LoRA
    let img_attn_out = add_lora(
        matmul_bias(&img_attn, bw("attn.to_out.0.weight")?, Some(bw("attn.to_out.0.bias")?))?,
        &img_attn, LoraTarget::ImgOut)?;
    let txt_attn_out = add_lora(
        matmul_bias(&txt_attn, bw("attn.to_add_out.weight")?, Some(bw("attn.to_add_out.bias")?))?,
        &txt_attn, LoraTarget::TxtOut)?;

    // Gated residual
    let img = img.add(&img_chunks[2].mul(&img_attn_out)?)?;
    let txt = txt.add(&txt_chunks[2].mul(&txt_attn_out)?)?;

    // FFN with LoRA
    let img_norm2 = flame_core::layer_norm::layer_norm(&img, &[DIM], None, None, NORM_EPS)?;
    let img_ffn_in = img_norm2.mul(&img_chunks[4].add_scalar(1.0)?)?.add(&img_chunks[3])?;
    let img_ffn_up = add_lora(
        matmul_bias(&img_ffn_in, bw("img_mlp.net.0.proj.weight")?, Some(bw("img_mlp.net.0.proj.bias")?))?,
        &img_ffn_in, LoraTarget::ImgFfnUp)?;
    let img_ffn_act = img_ffn_up.gelu()?;
    let img_ffn_down = add_lora(
        matmul_bias(&img_ffn_act, bw("img_mlp.net.2.weight")?, Some(bw("img_mlp.net.2.bias")?))?,
        &img_ffn_act, LoraTarget::ImgFfnDown)?;
    let img = img.add(&img_chunks[5].mul(&img_ffn_down)?)?;

    let txt_norm2 = flame_core::layer_norm::layer_norm(&txt, &[DIM], None, None, NORM_EPS)?;
    let txt_ffn_in = txt_norm2.mul(&txt_chunks[4].add_scalar(1.0)?)?.add(&txt_chunks[3])?;
    let txt_ffn_up = add_lora(
        matmul_bias(&txt_ffn_in, bw("txt_mlp.net.0.proj.weight")?, Some(bw("txt_mlp.net.0.proj.bias")?))?,
        &txt_ffn_in, LoraTarget::TxtFfnUp)?;
    let txt_ffn_act = txt_ffn_up.gelu()?;
    let txt_ffn_down = add_lora(
        matmul_bias(&txt_ffn_act, bw("txt_mlp.net.2.weight")?, Some(bw("txt_mlp.net.2.bias")?))?,
        &txt_ffn_act, LoraTarget::TxtFfnDown)?;
    let txt = txt.add(&txt_chunks[5].mul(&txt_ffn_down)?)?;

    Ok((img, txt))
}

// ---------------------------------------------------------------------------
// Verbatim port of `inference-flame::qwenimage_dit::block_forward`
// (qwenimage_dit.rs:1003-1197). Replaces `dual_stream_block_standalone`
// when the BlockOffloader is constructed with `.with_native_layout(true)`.
//
// Differences from the legacy `dual_stream_block_standalone`:
//   * Linear ops use `flame_core::ops::fused_inference::fused_linear3d_native`,
//     which expects PyTorch-native `[Cout, Cin]` layout (caller must
//     opt-in via `BlockOffloader::with_native_layout(true)`).
//   * Joint q/k/v are concatenated `[txt, img]` (txt first), matching
//     diffusers + inference-flame. `attn_out` is split with the same order.
//   * RoPE is applied AFTER the joint concat using `rope_fused_bf16` over
//     the joint `pe_cos / pe_sin` (shape `[1, 1, n_total, total_half]`
//     BF16). The caller is responsible for supplying joint cos/sin in
//     txt-first order (use `compute_joint_rope_pe` below).
//   * RMS norm and LayerNorm-no-affine route through
//     `cuda_ops_bf16::rms_norm_bf16` / `layer_norm_bf16` directly, matching
//     inference-flame's `Self::rms_norm` / `Self::layer_norm_no_affine`.
//
// LoRA adaptation: inference-flame uses a `LoraStack::apply(full_key, ...)`
// scheme keyed by full weight path. EDv2 uses `QwenImageLoraBundle.adapters`
// keyed by `(block_idx, LoraTarget)`. This port translates each
// `lin_lora` call site's weight suffix into the matching `LoraTarget` and
// applies the bundle's `LoRALinear::forward_delta` exactly as the legacy
// `add_lora` closure did.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn dual_stream_block_iflame(
    img: &Tensor,
    txt: &Tensor,
    temb: &Tensor,
    block_idx: usize,
    pe_cos: &Tensor,
    pe_sin: &Tensor,
    img_cos: &Tensor,
    img_sin: &Tensor,
    txt_cos: &Tensor,
    txt_sin: &Tensor,
    weights: &HashMap<String, Tensor>,
    bundle: &QwenImageLoraBundle,
) -> Result<(Tensor, Tensor)> {
    use flame_core::ops::fused_inference::fused_linear3d_native;

    let h = NUM_HEADS;
    let d = HEAD_DIM;
    let dim = DIM;
    let prefix = format!("transformer_blocks.{block_idx}");

    let w = |suffix: &str| -> Result<&Tensor> {
        let key = format!("{prefix}.{suffix}");
        weights.get(&key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing: {key}")))
    };

    // Legacy explicit transpose + matmul + add. Slower than
    // `fused_linear3d_native` but produces correct output — the fused
    // kernel has a numerical bug under EDv2's call pattern that I haven't
    // root-caused yet (tested both `native_layout=true` contiguous
    // [Cout, Cin] AND inference-flame's pre-transpose-then-untranspose
    // view pattern; both produce stylized garbage). With this legacy
    // path, output matches inference-flame visually. Perf is ~35% faster
    // than full-legacy (other iflame port pieces help) but ~5× slower
    // than inference-flame's `qwenimage_gen`.
    //
    // Offloader is configured with `native_layout=false` (legacy default)
    // so weights arrive logically `[Cin, Cout]` already.
    let linear_bias =
        |x: &Tensor, weight: &Tensor, bias: &Tensor| -> Result<Tensor> {
            let _ = fused_linear3d_native;
            let dims = x.shape().dims().to_vec();
            let in_feat = *dims.last().unwrap();
            let batch: usize = dims[..dims.len()-1].iter().product();
            let out_feat = weight.shape().dims()[1];
            let x_2d = x.reshape(&[batch, in_feat])?;
            let mut out = x_2d.matmul(weight)?;
            out = out.add(bias)?;
            let mut out_shape = dims[..dims.len()-1].to_vec();
            out_shape.push(out_feat);
            out.reshape(&out_shape)
        };

    // LoRA adapter: maps the inference-flame weight suffix to EDv2's
    // `LoraTarget` enum and applies `forward_delta` if a per-block adapter
    // exists. Returns the base linear output unchanged when no adapter.
    let lin_lora = |x: &Tensor, w_suffix: &str, b_suffix: &str| -> Result<Tensor> {
        let weight = w(w_suffix)?;
        let bias = w(b_suffix)?;
        let base = linear_bias(x, weight, bias)?;
        let target = match w_suffix {
            "attn.to_q.weight" => Some(LoraTarget::ImgQ),
            "attn.to_k.weight" => Some(LoraTarget::ImgK),
            "attn.to_v.weight" => Some(LoraTarget::ImgV),
            "attn.to_out.0.weight" => Some(LoraTarget::ImgOut),
            "img_mlp.net.0.proj.weight" => Some(LoraTarget::ImgFfnUp),
            "img_mlp.net.2.weight" => Some(LoraTarget::ImgFfnDown),
            "attn.add_q_proj.weight" => Some(LoraTarget::TxtQ),
            "attn.add_k_proj.weight" => Some(LoraTarget::TxtK),
            "attn.add_v_proj.weight" => Some(LoraTarget::TxtV),
            "attn.to_add_out.weight" => Some(LoraTarget::TxtOut),
            "txt_mlp.net.0.proj.weight" => Some(LoraTarget::TxtFfnUp),
            "txt_mlp.net.2.weight" => Some(LoraTarget::TxtFfnDown),
            _ => None,
        };
        if let Some(target) = target {
            if let Some(lora) = bundle.adapters.get(&(block_idx, target)) {
                let input_3d = if x.shape().dims().len() == 2 { x.unsqueeze(0)? } else { x.clone() };
                let delta = lora.forward_delta(&input_3d)?;
                return base.add(&delta);
            }
        }
        Ok(base)
    };

    // RMS norm — autograd-aware via `flame_core::norm::rms_norm`.
    // 2026-05-09 fix: `cuda_ops_bf16::rms_norm_bf16` is inference-only
    // (no `requires_grad` propagation, no autograd op recording), so its
    // output reads `requires_grad=false` and the gradient chain dies at
    // every Q/K head-norm in this trainer's per-block forward. That's the
    // chroma-pattern bug (project_chroma_lora_broken_2026-05-09); identical
    // shape — fixed identically here. Klein's `head_rms_norm_local`
    // (klein.rs:491-497) is the canonical pattern.
    let rms_norm_iflame = |x: &Tensor, scale: &Tensor, eps: f32| -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::norm::rms_norm(&x_2d, &[hidden], Some(scale), eps)?;
        out.reshape(&dims)
    };

    // LayerNorm without affine.  Use the canonical
    // `flame_core::layer_norm::layer_norm` directly — same wrapper the
    // sibling `dual_stream_block_standalone` and the rest of qwen forward
    // already use (lines 982/985/1044/1055/etc).
    //
    // 2026-05-09 fix history: an earlier version of this fix hand-rolled
    // a BF16→F32 manual mean/var/rstd path mirroring
    // `wan22_fwd/block.rs::layer_norm_no_affine`.  That path NaN'd the
    // first-step gradients in 50-step GPU validation (grad_norm = NaN at
    // step 1, inf at step 3, 3.57e14 at step 4) — same shape as the
    // F32-sub-tape problem fixed in lycoris-rs ac5350d.  Switching to
    // the canonical wrapper avoids the F32 sub-tape entirely.
    let layer_norm_no_affine = |x: &Tensor, eps: f32| -> Result<Tensor> {
        let hidden = *x.shape().dims().last().unwrap();
        flame_core::layer_norm::layer_norm(x, &[hidden], None, None, eps)
    };

    let b = img.shape().dims()[0];
    let n_img = img.shape().dims()[1];
    let n_txt = txt.shape().dims()[1];

    // ── img_mod(temb) and txt_mod(temb) ──
    // nn.Sequential(SiLU, Linear(dim, 6*dim))
    //
    // `temb` arrives as [B, dim]. The Linear expects 3D input; unsqueeze to
    // [B, 1, dim] and squeeze back.
    let temb_silu = temb.silu()?;
    let img_mods = lin_lora(&temb_silu.unsqueeze(1)?, "img_mod.1.weight", "img_mod.1.bias")?
        .squeeze(Some(1))?;
    let txt_mods = lin_lora(&temb_silu.unsqueeze(1)?, "txt_mod.1.weight", "txt_mod.1.bias")?
        .squeeze(Some(1))?;
    // Split each into two halves (norm1, norm2), each 3*dim:
    //   [shift, scale, gate] for norm1 and then [shift, scale, gate] for norm2.
    let img_mod1 = img_mods.narrow(1, 0, 3 * dim)?;
    let img_mod2 = img_mods.narrow(1, 3 * dim, 3 * dim)?;
    let txt_mod1 = txt_mods.narrow(1, 0, 3 * dim)?;
    let txt_mod2 = txt_mods.narrow(1, 3 * dim, 3 * dim)?;

    let img_shift1 = img_mod1.narrow(1, 0, dim)?;
    let img_scale1 = img_mod1.narrow(1, dim, dim)?;
    let img_gate1  = img_mod1.narrow(1, 2 * dim, dim)?;
    let img_shift2 = img_mod2.narrow(1, 0, dim)?;
    let img_scale2 = img_mod2.narrow(1, dim, dim)?;
    let img_gate2  = img_mod2.narrow(1, 2 * dim, dim)?;

    let txt_shift1 = txt_mod1.narrow(1, 0, dim)?;
    let txt_scale1 = txt_mod1.narrow(1, dim, dim)?;
    let txt_gate1  = txt_mod1.narrow(1, 2 * dim, dim)?;
    let txt_shift2 = txt_mod2.narrow(1, 0, dim)?;
    let txt_scale2 = txt_mod2.narrow(1, dim, dim)?;
    let txt_gate2  = txt_mod2.narrow(1, 2 * dim, dim)?;

    // ── norm1 + modulate for both streams ──
    //   norm(x) * (1 + scale)[:, None] + shift[:, None]
    let img_normed = layer_norm_no_affine(img, NORM_EPS)?;
    let img_modulated = img_normed
        .mul(&img_scale1.add_scalar(1.0)?.unsqueeze(1)?)?
        .add(&img_shift1.unsqueeze(1)?)?;

    let txt_normed = layer_norm_no_affine(txt, NORM_EPS)?;
    let txt_modulated = txt_normed
        .mul(&txt_scale1.add_scalar(1.0)?.unsqueeze(1)?)?
        .add(&txt_shift1.unsqueeze(1)?)?;

    // ── Q/K/V projections — keep the legacy [B, S, H, D] layout (NOT
    // pre-permuted to [B, H, N, D]) so we can match legacy's per-stream
    // apply_rope. Joint attention then cats and permutes once.
    let img_q = lin_lora(&img_modulated, "attn.to_q.weight", "attn.to_q.bias")?
        .reshape(&[b, n_img, h, d])?;
    let img_k = lin_lora(&img_modulated, "attn.to_k.weight", "attn.to_k.bias")?
        .reshape(&[b, n_img, h, d])?;
    let img_v = lin_lora(&img_modulated, "attn.to_v.weight", "attn.to_v.bias")?
        .reshape(&[b, n_img, h, d])?;

    let txt_q = lin_lora(&txt_modulated, "attn.add_q_proj.weight", "attn.add_q_proj.bias")?
        .reshape(&[b, n_txt, h, d])?;
    let txt_k = lin_lora(&txt_modulated, "attn.add_k_proj.weight", "attn.add_k_proj.bias")?
        .reshape(&[b, n_txt, h, d])?;
    let txt_v = lin_lora(&txt_modulated, "attn.add_v_proj.weight", "attn.add_v_proj.bias")?
        .reshape(&[b, n_txt, h, d])?;

    // ── QK RMSNorm — legacy `rms_norm_per_head` operates on
    // [B, S, H*D] (pre-reshape) and reshapes to [B*S*H, D] internally;
    // we already have [B, S, H, D], so route through the same kernel
    // family by collapsing to 2D over the head dim.
    let img_q = rms_norm_iflame(&img_q, w("attn.norm_q.weight")?, NORM_EPS)?;
    let img_k = rms_norm_iflame(&img_k, w("attn.norm_k.weight")?, NORM_EPS)?;
    let txt_q = rms_norm_iflame(&txt_q, w("attn.norm_added_q.weight")?, NORM_EPS)?;
    let txt_k = rms_norm_iflame(&txt_k, w("attn.norm_added_k.weight")?, NORM_EPS)?;

    // ── Apply RoPE per-stream (legacy style) — bisect from joint
    // `rope_fused_bf16` to verify whether the joint-RoPE path is buggy.
    // pe_cos / pe_sin args are unused in this mode.
    let _ = (pe_cos, pe_sin);
    let img_q = apply_rope(&img_q, img_cos, img_sin)?;
    let img_k = apply_rope(&img_k, img_cos, img_sin)?;
    let txt_q = apply_rope(&txt_q, txt_cos, txt_sin)?;
    let txt_k = apply_rope(&txt_k, txt_cos, txt_sin)?;

    // ── Concat img + txt (legacy order) and permute to joint attn shape ──
    let q = Tensor::cat(&[&img_q, &txt_q], 1)?.permute(&[0, 2, 1, 3])?;
    let k = Tensor::cat(&[&img_k, &txt_k], 1)?.permute(&[0, 2, 1, 3])?;
    let v = Tensor::cat(&[&img_v, &txt_v], 1)?.permute(&[0, 2, 1, 3])?;

    // ── Joint SDPA ──
    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

    // ── Split img + txt back out (legacy order; matches the cat above) ──
    let total_n = n_txt + n_img;
    let attn_2d = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, total_n, dim])?;
    let img_attn = attn_2d.narrow(1, 0, n_img)?;
    let txt_attn = attn_2d.narrow(1, n_img, n_txt)?;

    let img_attn = lin_lora(&img_attn, "attn.to_out.0.weight", "attn.to_out.0.bias")?;
    let txt_attn = lin_lora(&txt_attn, "attn.to_add_out.weight", "attn.to_add_out.bias")?;

    // ── Gated residual (using gate1) ──
    let img = img.add(&img_gate1.unsqueeze(1)?.mul(&img_attn)?)?;
    let txt = txt.add(&txt_gate1.unsqueeze(1)?.mul(&txt_attn)?)?;

    // ── FFN path for img ──
    let img_normed2 = layer_norm_no_affine(&img, NORM_EPS)?;
    let img_mlp_in = img_normed2
        .mul(&img_scale2.add_scalar(1.0)?.unsqueeze(1)?)?
        .add(&img_shift2.unsqueeze(1)?)?;
    let img_mlp = lin_lora(&img_mlp_in, "img_mlp.net.0.proj.weight", "img_mlp.net.0.proj.bias")?;
    let img_mlp = img_mlp.gelu()?;
    let img_mlp = lin_lora(&img_mlp, "img_mlp.net.2.weight", "img_mlp.net.2.bias")?;
    let img = img.add(&img_gate2.unsqueeze(1)?.mul(&img_mlp)?)?;

    // ── FFN path for txt ──
    let txt_normed2 = layer_norm_no_affine(&txt, NORM_EPS)?;
    let txt_mlp_in = txt_normed2
        .mul(&txt_scale2.add_scalar(1.0)?.unsqueeze(1)?)?
        .add(&txt_shift2.unsqueeze(1)?)?;
    let txt_mlp = lin_lora(&txt_mlp_in, "txt_mlp.net.0.proj.weight", "txt_mlp.net.0.proj.bias")?;
    let txt_mlp = txt_mlp.gelu()?;
    let txt_mlp = lin_lora(&txt_mlp, "txt_mlp.net.2.weight", "txt_mlp.net.2.bias")?;
    let txt = txt.add(&txt_gate2.unsqueeze(1)?.mul(&txt_mlp)?)?;

    Ok((img, txt))
}

// ---------------------------------------------------------------------------
// RoPE — 3-axis positional embeddings for QwenImage
// ---------------------------------------------------------------------------

/// Compute per-axis RoPE frequencies: `exp(i * pos / theta^(2k/dim))`.
/// Returns `(cos, sin)` each `[seq, dim/2]` in F32.
fn rope_freqs_1d(positions: &[f32], dim: usize, theta: f64) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let device = flame_core::global_cuda_device();
    let mut cos_data = vec![0.0f32; positions.len() * half];
    let mut sin_data = vec![0.0f32; positions.len() * half];
    for (p_idx, &pos) in positions.iter().enumerate() {
        for k in 0..half {
            let freq = pos as f64 / theta.powf(2.0 * k as f64 / dim as f64);
            cos_data[p_idx * half + k] = freq.cos() as f32;
            sin_data[p_idx * half + k] = freq.sin() as f32;
        }
    }
    let cos = Tensor::from_vec(cos_data, Shape::from_dims(&[positions.len(), half]), device.clone())?;
    let sin = Tensor::from_vec(sin_data, Shape::from_dims(&[positions.len(), half]), device)?;
    Ok((cos, sin))
}

/// Build QwenImage 3-axis RoPE for image tokens.
///
/// `scale_rope=true`: center-symmetric positions for H/W.
/// Returns `(cos, sin)` each `[img_seq, HEAD_DIM/2]` in BF16.
pub fn compute_image_rope(
    height: usize, width: usize, theta: f64,
) -> Result<(Tensor, Tensor)> {
    let [frame_dim, h_dim, w_dim] = ROPE_AXES_DIMS;
    let frame = 1usize; // single image

    // Frame axis: position [0]
    let frame_pos: Vec<f32> = (0..frame).map(|i| i as f32).collect();
    let (fc, fs) = rope_freqs_1d(&frame_pos, frame_dim, theta)?;
    // → [1, 8] broadcast to [1*H*W, 8]

    // Height axis: center-symmetric (scale_rope=true)
    // positions: [-(H-H/2)+1, ..., -1, 0, 1, ..., H/2-1] → centered
    let h_pos: Vec<f32> = {
        let half = height / 2;
        let neg: Vec<f32> = (0..height - half).map(|i| -((height - half - i) as f32)).collect();
        let pos: Vec<f32> = (0..half).map(|i| i as f32).collect();
        [neg, pos].concat()
    };
    let (hc, hs) = rope_freqs_1d(&h_pos, h_dim, theta)?;

    // Width axis: same center-symmetric
    let w_pos: Vec<f32> = {
        let half = width / 2;
        let neg: Vec<f32> = (0..width - half).map(|i| -((width - half - i) as f32)).collect();
        let pos: Vec<f32> = (0..half).map(|i| i as f32).collect();
        [neg, pos].concat()
    };
    let (wc, ws) = rope_freqs_1d(&w_pos, w_dim, theta)?;

    // Build grid: [F, H, W] → flatten to [F*H*W]
    // frame_freqs: [F, 1, 1, frame_dim/2] broadcast to [F, H, W, frame_dim/2]
    // height_freqs: [1, H, 1, h_dim/2] broadcast to [F, H, W, h_dim/2]
    // width_freqs: [1, 1, W, w_dim/2] broadcast to [F, H, W, w_dim/2]
    // Concatenate along last dim → [F*H*W, HEAD_DIM/2]
    let seq = frame * height * width;
    let half_head = (frame_dim + h_dim + w_dim) / 2;
    let device = flame_core::global_cuda_device();

    let mut cos_host = vec![0.0f32; seq * half_head];
    let mut sin_host = vec![0.0f32; seq * half_head];

    let fc_v = fc.to_dtype(DType::F32)?.to_vec()?;
    let fs_v = fs.to_dtype(DType::F32)?.to_vec()?;
    let hc_v = hc.to_dtype(DType::F32)?.to_vec()?;
    let hs_v = hs.to_dtype(DType::F32)?.to_vec()?;
    let wc_v = wc.to_dtype(DType::F32)?.to_vec()?;
    let ws_v = ws.to_dtype(DType::F32)?.to_vec()?;

    let f_half = frame_dim / 2;
    let h_half = h_dim / 2;
    let w_half = w_dim / 2;

    for f in 0..frame {
        for h in 0..height {
            for w in 0..width {
                let idx = (f * height + h) * width + w;
                let base = idx * half_head;
                // Frame part
                for k in 0..f_half {
                    cos_host[base + k] = fc_v[f * f_half + k];
                    sin_host[base + k] = fs_v[f * f_half + k];
                }
                // Height part
                for k in 0..h_half {
                    cos_host[base + f_half + k] = hc_v[h * h_half + k];
                    sin_host[base + f_half + k] = hs_v[h * h_half + k];
                }
                // Width part
                for k in 0..w_half {
                    cos_host[base + f_half + h_half + k] = wc_v[w * w_half + k];
                    sin_host[base + f_half + h_half + k] = ws_v[w * w_half + k];
                }
            }
        }
    }

    let cos = Tensor::from_vec(cos_host, Shape::from_dims(&[seq, half_head]), device.clone())?
        .to_dtype(DType::BF16)?;
    let sin = Tensor::from_vec(sin_host, Shape::from_dims(&[seq, half_head]), device)?
        .to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

/// Build RoPE for text tokens: sequential positions starting after max(H/2, W/2).
///
/// When `scale_rope=True` (QwenImage default), the image RoPE uses center-symmetric
/// positions in `[-H/2, H/2)` and `[-W/2, W/2)`. The text offset must match:
///   `max_vid_index = max(height // 2, width // 2)`
/// Reference: `qwen_image_model.py:342`
///
/// Returns `(cos, sin)` each `[txt_seq, HEAD_DIM/2]` in BF16.
pub fn compute_text_rope(
    txt_seq: usize, height: usize, width: usize, theta: f64,
) -> Result<(Tensor, Tensor)> {
    // scale_rope=True: text positions start at max(H/2, W/2)
    let max_vid_idx = (height / 2).max(width / 2);
    let half_head = HEAD_DIM / 2;
    let device = flame_core::global_cuda_device();

    // All 3 axes use the same sequential positions for text
    let [frame_dim, h_dim, w_dim] = ROPE_AXES_DIMS;
    let positions: Vec<f32> = (0..txt_seq).map(|i| (max_vid_idx + i) as f32).collect();

    let (fc, fs) = rope_freqs_1d(&positions, frame_dim, theta)?;
    let (hc, hs) = rope_freqs_1d(&positions, h_dim, theta)?;
    let (wc, ws) = rope_freqs_1d(&positions, w_dim, theta)?;

    // Concat along last dim: [txt_seq, f_half] + [txt_seq, h_half] + [txt_seq, w_half]
    let cos = Tensor::cat(&[&fc, &hc, &wc], 1)?.to_dtype(DType::BF16)?;
    let sin = Tensor::cat(&[&fs, &hs, &ws], 1)?.to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

// ---------------------------------------------------------------------------
// Multi-region RoPE (Qwen-Image-Edit)
// ---------------------------------------------------------------------------
//
// Edit mode concatenates multiple image regions along the seq dim in this
// order: [target_region, control_region_0, control_region_1, ...]. Each region
// has its own (h, w) patch grid. Matching diffusers (`transformer_qwenimage.py:
// _compute_video_freqs(frame, height, width, idx)` at line 317), each region's
// frame axis uses `frame_offset = region_idx` — so the control frame positions
// start at 1 even though the actual frame count is 1 per image.
//
// The text RoPE offset (`max_vid_index`) is the MAX over ALL regions of
// `max(h/2, w/2)` with `scale_rope=True`.

/// `regions[i] = (height, width)` — one per image region (target, control_0, ...).
/// Returns `(img_cos, img_sin)` each `[sum(h_i * w_i), HEAD_DIM/2]` BF16.
pub fn compute_image_rope_multi(
    regions: &[(usize, usize)],
    theta: f64,
) -> Result<(Tensor, Tensor)> {
    if regions.is_empty() {
        return Err(flame_core::Error::InvalidInput(
            "compute_image_rope_multi: regions empty".into(),
        ));
    }
    let [frame_dim, h_dim, w_dim] = ROPE_AXES_DIMS;
    let f_half = frame_dim / 2;
    let h_half = h_dim / 2;
    let w_half = w_dim / 2;
    let half_head = f_half + h_half + w_half;
    let device = flame_core::global_cuda_device();

    let total_seq: usize = regions.iter().map(|(h, w)| h * w).sum();
    let mut cos_host = vec![0.0f32; total_seq * half_head];
    let mut sin_host = vec![0.0f32; total_seq * half_head];

    // Write per-region, using frame_offset = region_idx (so control's frame
    // positions start at 1 even though there's a single frame per region).
    let mut cursor = 0usize;
    for (region_idx, &(height, width)) in regions.iter().enumerate() {
        // Frame axis freqs at position = region_idx (single frame per region)
        let frame_pos: Vec<f32> = vec![region_idx as f32];
        let (fc, fs) = rope_freqs_1d(&frame_pos, frame_dim, theta)?;
        let fc_v = fc.to_dtype(DType::F32)?.to_vec()?;
        let fs_v = fs.to_dtype(DType::F32)?.to_vec()?;

        // Center-symmetric H (scale_rope=True)
        let h_pos: Vec<f32> = {
            let half = height / 2;
            let neg: Vec<f32> = (0..height - half).map(|i| -((height - half - i) as f32)).collect();
            let pos: Vec<f32> = (0..half).map(|i| i as f32).collect();
            [neg, pos].concat()
        };
        let (hc, hs) = rope_freqs_1d(&h_pos, h_dim, theta)?;
        let hc_v = hc.to_dtype(DType::F32)?.to_vec()?;
        let hs_v = hs.to_dtype(DType::F32)?.to_vec()?;

        // Center-symmetric W
        let w_pos: Vec<f32> = {
            let half = width / 2;
            let neg: Vec<f32> = (0..width - half).map(|i| -((width - half - i) as f32)).collect();
            let pos: Vec<f32> = (0..half).map(|i| i as f32).collect();
            [neg, pos].concat()
        };
        let (wc, ws) = rope_freqs_1d(&w_pos, w_dim, theta)?;
        let wc_v = wc.to_dtype(DType::F32)?.to_vec()?;
        let ws_v = ws.to_dtype(DType::F32)?.to_vec()?;

        for h in 0..height {
            for w in 0..width {
                let idx = cursor + h * width + w;
                let base = idx * half_head;
                for k in 0..f_half {
                    cos_host[base + k] = fc_v[k];
                    sin_host[base + k] = fs_v[k];
                }
                for k in 0..h_half {
                    cos_host[base + f_half + k] = hc_v[h * h_half + k];
                    sin_host[base + f_half + k] = hs_v[h * h_half + k];
                }
                for k in 0..w_half {
                    cos_host[base + f_half + h_half + k] = wc_v[w * w_half + k];
                    sin_host[base + f_half + h_half + k] = ws_v[w * w_half + k];
                }
            }
        }
        cursor += height * width;
    }

    let cos = Tensor::from_vec(cos_host, Shape::from_dims(&[total_seq, half_head]), device.clone())?
        .to_dtype(DType::BF16)?;
    let sin = Tensor::from_vec(sin_host, Shape::from_dims(&[total_seq, half_head]), device)?
        .to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

/// Text RoPE for edit mode.
///
/// `max_vid_index` is taken over ALL regions: `max over i of max(h_i/2, w_i/2)`
/// (with scale_rope=True). Matches `qwen_image_model.py:341-344`.
pub fn compute_text_rope_multi(
    txt_seq: usize,
    regions: &[(usize, usize)],
    theta: f64,
) -> Result<(Tensor, Tensor)> {
    let max_vid_idx: usize = regions.iter()
        .map(|&(h, w)| (h / 2).max(w / 2))
        .max()
        .unwrap_or(0);
    let [frame_dim, h_dim, w_dim] = ROPE_AXES_DIMS;
    let positions: Vec<f32> = (0..txt_seq).map(|i| (max_vid_idx + i) as f32).collect();
    let (fc, fs) = rope_freqs_1d(&positions, frame_dim, theta)?;
    let (hc, hs) = rope_freqs_1d(&positions, h_dim, theta)?;
    let (wc, ws) = rope_freqs_1d(&positions, w_dim, theta)?;
    let cos = Tensor::cat(&[&fc, &hc, &wc], 1)?.to_dtype(DType::BF16)?;
    let sin = Tensor::cat(&[&fs, &hs, &ws], 1)?.to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

/// Apply complex RoPE: x_out = x * cos + rotate(x) * sin
/// `x`: [B, S, H, D], `cos`/`sin`: [S, D/2]
/// Uses the complex multiplication path (pairs → rotate → add).
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    // Reshape to pairs: [B, S, H, D/2, 2]
    let x_pairs = x.reshape(&[b, s, h, d / 2, 2])?;
    let x_re = x_pairs.narrow(4, 0, 1)?.squeeze_dim(4)?; // [B, S, H, D/2]
    let x_im = x_pairs.narrow(4, 1, 1)?.squeeze_dim(4)?;

    // cos/sin: [S, D/2] → [1, S, 1, D/2]
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    // Complex multiply: (x_re + i*x_im) * (cos + i*sin)
    // out_re = x_re * cos - x_im * sin
    // out_im = x_re * sin + x_im * cos
    let out_re = x_re.mul(&cos)?.sub(&x_im.mul(&sin)?)?;
    let out_im = x_re.mul(&sin)?.add(&x_im.mul(&cos)?)?;

    // Interleave back: [B, S, H, D/2, 2] → [B, S, H, D]
    let out_re = out_re.unsqueeze(4)?;
    let out_im = out_im.unsqueeze(4)?;
    Tensor::cat(&[&out_re, &out_im], 4)?.reshape(&[b, s, h, d])
}

fn rms_norm_per_head(x: &Tensor, weight: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let batch: usize = dims[..dims.len()-1].iter().product();
    let flat = x.reshape(&[batch * num_heads, head_dim])?;
    let normed = flame_core::norm::rms_norm(&flat.unsqueeze(0)?, &[head_dim], Some(weight), NORM_EPS)?;
    normed.reshape(&dims)
}

/// Sinusoidal timestep embedding matching diffusers `Timesteps` + `get_timestep_embedding`.
///
/// Parameters match QwenImage's `QwenTimestepProjEmbeddings`:
///   `Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)`
///
/// Reference: `qwen_image_model.py:133-182`, `qwen_image_model.py:256`
///
/// The model receives sigma ∈ [0,1]. Python divides scheduler timesteps (1–1000) by 1000
/// before the model, then `scale=1000` in the embedding recovers the original range.
/// We apply the same scale here so the sinusoidal frequencies see values in ~[0, 1000].
fn sinusoidal_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let max_period: f64 = 10000.0;
    // downscale_freq_shift=0 → denominator is half
    let mut freqs = Vec::with_capacity(half);
    for i in 0..half {
        freqs.push((-max_period.ln() * i as f64 / half as f64).exp() as f32);
    }
    // scale=1000: multiply timestep before computing sin/cos
    let t_f32 = t.to_dtype(DType::F32)?.mul_scalar(1000.0)?;
    let t_2d = t_f32.unsqueeze(1)?;
    let freqs_t = Tensor::from_vec(freqs, Shape::from_dims(&[1, half]), t.device().clone())?;
    let args = t_2d.mul(&freqs_t)?;
    let cos_part = args.cos()?;
    let sin_part = args.sin()?;
    // flip_sin_to_cos=True → output order is (cos, sin)
    Tensor::cat(&[&cos_part, &sin_part], 1)?.to_dtype(DType::BF16)
}

/// Pack latents from [B, 16, H, W] → [B, H/2 * W/2, 64]
///
/// musubi-tuner `qwen_image_utils.py:852-869`
pub fn pack_latents(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let x = x.reshape(&[b, c, h / 2, 2, w / 2, 2])?;
    let x = x.permute(&[0, 2, 4, 1, 3, 5])?;
    x.reshape(&[b, (h / 2) * (w / 2), c * 4])
}

/// Unpack latents from [B, H/2 * W/2, 64] → [B, 16, H, W]
pub fn unpack_latents(x: &Tensor, h_lat: usize, w_lat: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let b = dims[0];
    let c = OUT_CHANNELS;
    let h_tok = h_lat / 2;
    let w_tok = w_lat / 2;
    let x = x.reshape(&[b, h_tok, w_tok, c, 2, 2])?;
    let x = x.permute(&[0, 3, 1, 4, 2, 5])?;
    x.reshape(&[b, c, h_lat, w_lat])
}
