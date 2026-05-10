//! Ernie model — matching inference-flame's verified forward.
//! Single-stream DiT, 36 layers × 4096 dim, 32 heads, SwiGLU, shared AdaLN.

use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};
use crate::adapter::{AdapterModule, LycorisLinear};
use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::lycoris::{LycorisAlgo, LycorisBundleConfig};
use crate::models::chroma::build_lycoris_linear;
use crate::models::TrainableModel;
use crate::Result;

pub const HIDDEN: usize = 4096;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const LAYERS: usize = 36;
pub const FFN: usize = 12288;
pub const IN_C: usize = 128;
pub const PATCH_SIZE: usize = 1;
pub const TEXT_DIM: usize = 3072;
pub const ROPE_THETA: f64 = 256.0;
pub const ROPE_AXES: [usize; 3] = [32, 48, 48];
pub const NORM_EPS: f32 = 1e-6;

pub struct ErnieModel {
    pub config: TrainConfig,
    pub device: Arc<CudaDevice>,
    pub weights: HashMap<String, Tensor>,
    /// Legacy plain-LoRA adapters. Always populated when `is_lora=true`; one
    /// per (layer, slot) of `LAYERS * 7`. When a non-`lora` LyCORIS algo is
    /// active, the corresponding entry in `lycoris_adapters` shadows this one
    /// (legacy slot is left untouched but never dispatched to).
    pub lora_adapters: Vec<LoRALinear>,
    /// Phase 2b: LyCORIS adapters parallel to `lora_adapters`. `Vec` length
    /// matches `lora_adapters.len()` exactly; entries are `None` for the
    /// legacy `--algo lora` path (byte-identical pre-2b) and `Some(_)` once
    /// `swap_lycoris_bundle` populates them.
    pub lycoris_adapters: Vec<Option<Arc<LycorisLinear>>>,
    /// Currently active algo. `LycorisAlgo::None` means `lora_adapters` is
    /// the live path (legacy plain-LoRA, byte-identical).
    pub algo: LycorisAlgo,
    pub parameters: Vec<Parameter>,
    pub is_lora: bool,
    /// When Some, per-layer transformer weights are streamed from pinned host RAM
    /// into reusable GPU slots per layer, per step via BlockOffloader.
    /// Block index space: `0..LAYERS` → `layers.{i}.*`.
    pub offloader: Option<std::sync::Arc<std::sync::Mutex<crate::training::block_offload::BlockOffloader>>>,
}

impl ErnieModel {
    pub fn load(paths: &[std::path::PathBuf], config: &TrainConfig, device: Arc<CudaDevice>) -> Result<Self> {
        let mut weights = HashMap::new();
        for p in paths {
            let part = flame_core::serialization::load_file(p, &device)?;
            for (k, v) in part {
                weights.insert(k, v.to_dtype(DType::BF16)?);
            }
        }
        log::info!("Ernie: {} tensors loaded", weights.len());
        let is_lora = config.is_lora();
        let mut lora_adapters = Vec::new();
        let mut parameters = Vec::new();
        if is_lora {
            let rank = config.lora_rank as usize;
            let alpha = config.lora_alpha as f32;
            for i in 0..LAYERS {
                let s = 42u64 + i as u64;
                // Q, K, V, out: 4096 → 4096
                lora_adapters.push(LoRALinear::new(HIDDEN, HIDDEN, rank, alpha, device.clone(), s)?);
                lora_adapters.push(LoRALinear::new(HIDDEN, HIDDEN, rank, alpha, device.clone(), s+1)?);
                lora_adapters.push(LoRALinear::new(HIDDEN, HIDDEN, rank, alpha, device.clone(), s+2)?);
                lora_adapters.push(LoRALinear::new(HIDDEN, HIDDEN, rank, alpha, device.clone(), s+3)?);
                // gate_proj: 4096 → 12288
                lora_adapters.push(LoRALinear::new(HIDDEN, FFN, rank, alpha, device.clone(), s+4)?);
                // up_proj: 4096 → 12288
                lora_adapters.push(LoRALinear::new(HIDDEN, FFN, rank, alpha, device.clone(), s+5)?);
                // linear_fc2 (down): 12288 → 4096
                lora_adapters.push(LoRALinear::new(FFN, HIDDEN, rank, alpha, device.clone(), s+6)?);
            }
            for l in &lora_adapters { parameters.extend(l.parameters()); }
        } else {
            for (_, t) in &weights {
                parameters.push(Parameter::new(t.to_dtype(DType::F32)?.requires_grad_(true)));
            }
        }
        let lycoris_adapters = vec![None; lora_adapters.len()];
        Ok(Self {
            config: config.clone(),
            device,
            weights,
            lora_adapters,
            lycoris_adapters,
            algo: LycorisAlgo::None,
            parameters,
            is_lora,
            offloader: None,
        })
    }

    /// Phase 2b: swap the legacy plain-LoRA bundle for a LyCORIS-aware one
    /// covering LoCon / LoHa / LoKr / Full / OFT. Mirrors
    /// `ChromaLoraBundle::new_with_config`'s construction pattern but inlines
    /// it into the model since ernie does not carry a separate bundle struct.
    /// The legacy `lora_adapters` entries are kept resident (never dispatched
    /// to once `algo != None`) so save/load can still reuse the per-slot
    /// shape table.
    ///
    /// Layout: `lycoris_adapters[layer*7 + slot]` for each
    /// `(layer ∈ 0..LAYERS, slot ∈ 0..7)`. Slot order matches `LORA_SLOT_KEYS`.
    pub fn swap_lycoris_bundle(&mut self, config: &LycorisBundleConfig) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "swap_lycoris_bundle: model is not in LoRA mode".into(),
            ));
        }
        if config.algo == LycorisAlgo::None {
            // Legacy path retained — no swap. Caller short-circuits but be
            // defensive.
            return Ok(());
        }
        // Per-slot in/out dims: must match `ErnieModel::load`'s slot order
        // (Q, K, V, out, gate_proj, up_proj, linear_fc2).
        const SLOT_DIMS: [(usize, usize); 7] = [
            (HIDDEN, HIDDEN), // 0: to_q
            (HIDDEN, HIDDEN), // 1: to_k
            (HIDDEN, HIDDEN), // 2: to_v
            (HIDDEN, HIDDEN), // 3: to_out.0
            (HIDDEN, FFN),    // 4: gate_proj
            (HIDDEN, FFN),    // 5: up_proj
            (FFN, HIDDEN),    // 6: linear_fc2
        ];
        let mut lycoris_adapters: Vec<Option<Arc<LycorisLinear>>> =
            Vec::with_capacity(self.lora_adapters.len());
        let mut params: Vec<Parameter> = Vec::new();
        for layer in 0..LAYERS {
            for &(in_dim, out_dim) in SLOT_DIMS.iter() {
                let wrapper = build_lycoris_linear(config, in_dim, out_dim, self.device.clone())
                    .map_err(|e| crate::EriDiffusionError::Model(format!(
                        "swap_lycoris_bundle: build_lycoris_linear({in_dim}, {out_dim}): {e}"
                    )))?;
                let arc = Arc::new(wrapper);
                params.extend(arc.to_parameters());
                lycoris_adapters.push(Some(arc));
                let _ = layer; // index only used for asserts in tests.
            }
        }
        debug_assert_eq!(lycoris_adapters.len(), self.lora_adapters.len());
        self.lycoris_adapters = lycoris_adapters;
        self.algo = config.algo;
        self.parameters = params;
        Ok(())
    }

    /// Look up the active adapter at flat index `adapter_idx` (= `layer*7 + slot`).
    /// Prefers the LyCORIS slot when populated; falls back to the legacy
    /// `LoRALinear` entry. Returns `None` when neither is populated (e.g.
    /// out-of-range index).
    pub fn adapter_for(&self, adapter_idx: usize) -> Option<&dyn AdapterModule> {
        if let Some(Some(lyc)) = self.lycoris_adapters.get(adapter_idx) {
            return Some(lyc.as_ref() as &dyn AdapterModule);
        }
        if let Some(legacy) = self.lora_adapters.get(adapter_idx) {
            return Some(legacy as &dyn AdapterModule);
        }
        None
    }

    /// Enable per-layer block offloading via `BlockOffloader`. Drops all
    /// `layers.{i}.*` weights from VRAM; blocks are streamed from pinned host RAM
    /// into reusable GPU slots per layer, per step. Works for both base and LoRA
    /// inference.
    pub fn enable_offload(&mut self, shards: Vec<std::path::PathBuf>) -> Result<()> {
        let to_drop: Vec<String> = self.weights.keys()
            .filter(|k| k.starts_with("layers."))
            .cloned()
            .collect();
        let n = to_drop.len();
        for k in to_drop { self.weights.remove(&k); }
        log::info!("Ernie offload: dropped {} per-layer weight tensors", n);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);

        struct ErnieFacilitator;
        impl crate::training::block_offload::BlockFacilitator for ErnieFacilitator {
            fn block_count(&self) -> usize { LAYERS }
            fn classify_key(&self, key: &str) -> Option<usize> {
                let rest = key.strip_prefix("layers.")?;
                rest.split('.').next()?.parse().ok()
            }
        }

        let shard_strs: Vec<String> = shards.iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let path_refs: Vec<&str> = shard_strs.iter().map(|s| s.as_str()).collect();

        let use_streaming = std::env::var("ERNIE_BLOCK_STREAMING")
            .ok()
            .map(|v| !matches!(v.as_str(), "0" | "" | "false" | "False"))
            .unwrap_or(true);

        let offloader = if use_streaming {
            log::info!("Ernie BlockOffloader: streaming mode");
            crate::training::block_offload::BlockOffloader::load_streaming(
                &path_refs, &ErnieFacilitator, self.device.clone(),
            )
        } else {
            log::info!("Ernie BlockOffloader: pinned-RAM mode");
            crate::training::block_offload::BlockOffloader::load(
                &path_refs, &ErnieFacilitator, self.device.clone(),
            )
        }
        // native_layout=true: leave 2D .weight tensors in on-disk [Cout, Cin] layout.
        // Ernie model code calls `.transpose()` itself before matmul.
        .map(|o| o.with_native_layout(true))
        .map_err(|e| crate::EriDiffusionError::Model(format!("BlockOffloader: {e}")))?;

        self.offloader = Some(std::sync::Arc::new(std::sync::Mutex::new(offloader)));
        log::info!("Ernie BlockOffloader ready ({} layers)", LAYERS);
        Ok(())
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| crate::EriDiffusionError::Model(format!("missing: {}", key)))
    }

    fn linear(&self, x: &Tensor, w_key: &str, bias_key: Option<&str>) -> Result<Tensor> {
        let w = self.w(w_key)?;
        let mut out = x.matmul(&w.transpose()?)?;
        if let Some(bk) = bias_key { out = out.add(self.w(bk)?)?; }
        Ok(out)
    }

    /// Linear with LoRA delta injected. adapter_idx: which LoRALinear in self.lora_adapters.
    fn linear_lora(&self, x: &Tensor, w_key: &str, bias_key: Option<&str>, adapter_idx: usize) -> Result<Tensor> {
        let base = self.linear(x, w_key, bias_key)?;
        if self.is_lora {
            if let Some(adapter) = self.lora_adapters.get(adapter_idx) {
                let delta = adapter.forward_delta(x)?;
                return base.add(&delta).map_err(Into::into);
            }
        }
        Ok(base)
    }

    /// Mirrors diffusers `ErnieImagePatchEmbedDynamic.forward`:
    ///   conv1×1 NCHW→NCHW (linearized as 1×1 matmul), then [B,C,H,W] → [B,H*W,C].
    /// Critical: NCHW must be permuted to NHWC before flattening — a direct reshape
    /// from [B,C,H,W] to [B*H*W,C] scrambles channel data (channel-c row k ends up
    /// holding spatial-position-(k*C+c) of channel 0).
    fn patch_embed(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b_sz, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let n_img = h * w / (PATCH_SIZE * PATCH_SIZE);
        // [B, IN_C, H, W] → permute → [B, H, W, IN_C] → flatten spatial → [B*N_img, IN_C].
        let x_nhwc = x.permute(&[0, 2, 3, 1])?.contiguous()?;
        let x_f = x_nhwc.reshape(&[b_sz * n_img, IN_C * PATCH_SIZE * PATCH_SIZE])?;
        let w_proj = self.w("x_embedder.proj.weight")?; // [HIDDEN, IN_C, 1, 1]
        let w_f = w_proj.reshape(&[HIDDEN, IN_C])?;
        let out = x_f.matmul(&w_f.transpose()?)?.add(self.w("x_embedder.proj.bias")?)?;
        out.reshape(&[b_sz, n_img, HIDDEN]).map_err(Into::into)
    }

    fn rms_norm(&self, x: &Tensor, scale_key: &str) -> Result<Tensor> {
        let scale = self.w(scale_key)?;
        flame_core::norm::rms_norm(x, &[HIDDEN], Some(scale), NORM_EPS).map_err(Into::into)
    }

    fn qk_rms_norm(&self, x: &Tensor, scale_key: &str) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let scale = self.w(scale_key)?;
        let x_h = x.reshape(&[batch * HEADS, HEAD_DIM])?;
        let n = flame_core::norm::rms_norm(&x_h, &[HEAD_DIM], Some(scale), NORM_EPS)?;
        n.reshape(&dims).map_err(Into::into)
    }

    /// 3-axis RoPE matching diffusers ErnieImageEmbedND3 + apply_rotary_emb.
    /// Image positions: [text_lens, row, col] with row-major (Hp, Wp) per `indexing="ij"`.
    /// Text positions:  [t, 0, 0].
    ///
    /// Layout of cos/sin (full HEAD_DIM, NOT half) follows diffusers' replicate-pair-then-flatten:
    ///   axis-0 (axes_dim=32, 16 freqs) occupies positions [0..32]   as f0,f0,f1,f1,...,f15,f15
    ///   axis-1 (axes_dim=48, 24 freqs) occupies positions [32..80]  same pattern
    ///   axis-2 (axes_dim=48, 24 freqs) occupies positions [80..128] same pattern
    /// Note: this is NOT classic half-split (cos[d] != cos[d+half]); the rotation in `rope()`
    /// applies cos/sin element-wise to the full head, with rotate-half inside.
    fn build_rope(&self, n_img: usize, n_txt: usize, hp: usize, wp: usize) -> Result<(Tensor, Tensor)> {
        let _ = hp;
        let total = n_img + n_txt;
        let mut cos = vec![0f32; total * HEAD_DIM];
        let mut sin = vec![0f32; total * HEAD_DIM];
        let axis_offsets = [
            0usize,
            ROPE_AXES[0],
            ROPE_AXES[0] + ROPE_AXES[1],
        ];
        for s in 0..total {
            let pos = if s < n_img {
                [n_txt as f32, (s / wp) as f32, (s % wp) as f32]
            } else {
                [(s - n_img) as f32, 0.0, 0.0]
            };
            for axis_idx in 0..3 {
                let axis_dim = ROPE_AXES[axis_idx];
                let half_axis = axis_dim / 2;
                let p = pos[axis_idx];
                let off = axis_offsets[axis_idx];
                for j in 0..half_axis {
                    let freq = (ROPE_THETA.powf(-(2.0 * j as f64) / axis_dim as f64)) as f32;
                    let a = p * freq;
                    let (c, sv) = (a.cos(), a.sin());
                    let row = s * HEAD_DIM;
                    cos[row + off + 2 * j] = c;
                    cos[row + off + 2 * j + 1] = c;
                    sin[row + off + 2 * j] = sv;
                    sin[row + off + 2 * j + 1] = sv;
                }
            }
        }
        Ok((
            Tensor::from_vec(cos, Shape::from_dims(&[1, 1, total, HEAD_DIM]), self.device.clone())?
                .to_dtype(DType::BF16)?,
            Tensor::from_vec(sin, Shape::from_dims(&[1, 1, total, HEAD_DIM]), self.device.clone())?
                .to_dtype(DType::BF16)?,
        ))
    }

    fn timestep_embedding(&self, t: &Tensor) -> Result<Tensor> {
        let t_v = t.to_vec()?;
        let dim = HIDDEN;
        let b = t_v.len();
        let h = dim / 2;
        let mut d = vec![0f32; b * dim];
        for (bi, &tv) in t_v.iter().enumerate() {
            for j in 0..h {
                let f = (-(10000.0f64.ln())*(j as f64)/(h as f64)).exp() as f32;
                let a = tv * f;  // NO 1000x scaling
                d[bi*dim+j] = a.sin(); d[bi*dim+h+j] = a.cos();
            }
        }
        let emb = Tensor::from_vec(d, Shape::from_dims(&[b, dim]), self.device.clone())?.to_dtype(DType::BF16)?;
        let h1 = self.linear(&emb, "time_embedding.linear_1.weight", Some("time_embedding.linear_1.bias"))?;
        let h1 = h1.silu()?;
        self.linear(&h1, "time_embedding.linear_2.weight", Some("time_embedding.linear_2.bias"))
    }

    pub fn forward(&mut self, img: &Tensor, txt: &Tensor, timestep: &Tensor) -> Result<Tensor> {
        let dims = img.shape().dims();
        let b = dims[0];
        let n_img = dims[1] * dims[2] * dims[3] / (IN_C * PATCH_SIZE * PATCH_SIZE);
        let n_txt = txt.shape().dims()[1];

        let img_tokens = self.patch_embed(img)?;
        let txt_tokens = self.linear(txt, "text_proj.weight", None)?;
        let x = Tensor::cat(&[&img_tokens, &txt_tokens], 1)?;
        let n_total = n_img + n_txt;

        // Timestep + shared AdaLN (SiLU!)
        let t_emb = self.timestep_embedding(timestep)?;
        let mod_out = t_emb.silu()?.matmul(&self.w("adaLN_modulation.1.weight")?.transpose()?)?.add(self.w("adaLN_modulation.1.bias")?)?;
        let chunks = mod_out.chunk(6, 1)?;

        // Unsqueeze to [B, 1, HIDDEN] for broadcasting with [B, N, HIDDEN]
        let s_msa = chunks[0].unsqueeze(1)?;
        let sc_msa = chunks[1].unsqueeze(1)?;
        let g_msa = chunks[2].unsqueeze(1)?;
        let s_mlp = chunks[3].unsqueeze(1)?;
        let sc_mlp = chunks[4].unsqueeze(1)?;
        let g_mlp = chunks[5].unsqueeze(1)?;

        // RoPE — Hp=H, Wp=W since patch_size=1 (no spatial patchify in the DiT itself).
        let (cos, sin) = self.build_rope(n_img, n_txt, dims[2], dims[3])?;
        let cos_b = cos.to_dtype(DType::BF16)?;
        let sin_b = sin.to_dtype(DType::BF16)?;

        let mut x = x;
        // Inference fast path: skip HashMap clone + checkpoint closure overhead.
        // Training path: unchanged (HashMap clone + grad-checkpoint for activation offload).
        let is_inference = !flame_core::autograd::AutogradContext::is_recording();
        let use_checkpoint = std::env::var("ERNIE_GRAD_CHECKPOINT").map(|v| v != "0").unwrap_or(true);
        for i in 0..LAYERS {
            // BlockOffloader: stream layer i from pinned host RAM into GPU slot.
            if let Some(ref off) = self.offloader {
                let arc = off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .ensure_block(i)
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader ensure_block({i}): {e}")))?;
                // Merge the block's weights into self.weights for the forward body.
                for (k, v) in arc.iter() {
                    self.weights.insert(k.clone(), v.clone());
                }
            }

            if is_inference {
                // Inference fast path — borrow weights directly, no clone, no closure.
                let lora_base = i * 7;
                let lora_slice: Option<&[LoRALinear]> = if self.is_lora {
                    Some(&self.lora_adapters[lora_base..lora_base+7])
                } else {
                    None
                };
                let lycoris_slice: Option<&[Option<Arc<LycorisLinear>>]> = if self.is_lora {
                    Some(&self.lycoris_adapters[lora_base..lora_base+7])
                } else {
                    None
                };
                x = block_forward_iflame(
                    &x,
                    &sc_msa, &s_msa, &g_msa,
                    &sc_mlp, &s_mlp, &g_mlp,
                    &cos_b, &sin_b,
                    &self.weights,
                    lora_slice,
                    lycoris_slice,
                    i, b, n_total,
                )?;
            } else {
                // Training path — extract this layer's weights into a self-contained map so the
                // checkpoint closure (which must be 'static) can own them.
                let layer_prefix = format!("layers.{}.", i);
                let mut layer_weights: HashMap<String, Tensor> = HashMap::new();
                for (k, v) in self.weights.iter() {
                    if k.starts_with(&layer_prefix) {
                        layer_weights.insert(k.clone(), v.clone());
                    }
                }
                let lora_base = i * 7;
                let lora_adapters: Option<Vec<LoRALinear>> = if self.is_lora {
                    Some(self.lora_adapters[lora_base..lora_base+7].to_vec())
                } else {
                    None
                };
                // Phase 2b: clone the LyCORIS slot Arcs (cheap refcount bump).
                // Each closure capture must own a 'static-able view; the
                // checkpoint closure path requires this.
                let lycoris_adapters: Option<Vec<Option<Arc<LycorisLinear>>>> = if self.is_lora {
                    Some(self.lycoris_adapters[lora_base..lora_base+7].to_vec())
                } else {
                    None
                };

                let x_in = x.clone();
                let cos_c = cos_b.clone();
                let sin_c = sin_b.clone();
                let s_msa_c = s_msa.clone();
                let sc_msa_c = sc_msa.clone();
                let g_msa_c = g_msa.clone();
                let s_mlp_c = s_mlp.clone();
                let sc_mlp_c = sc_mlp.clone();
                let g_mlp_c = g_mlp.clone();

                let result = if use_checkpoint {
                    flame_core::autograd::AutogradContext::checkpoint(
                        &[x_in.clone()],
                        move || ernie_layer_forward_standalone(
                            x_in.clone(),
                            sc_msa_c.clone(), s_msa_c.clone(), g_msa_c.clone(),
                            sc_mlp_c.clone(), s_mlp_c.clone(), g_mlp_c.clone(),
                            cos_c.clone(), sin_c.clone(),
                            layer_weights.clone(),
                            lora_adapters.clone(),
                            lycoris_adapters.clone(),
                            i, b, n_total,
                        ),
                    )?
                } else {
                    ernie_layer_forward_standalone(
                        x_in,
                        sc_msa_c, s_msa_c, g_msa_c,
                        sc_mlp_c, s_mlp_c, g_mlp_c,
                        cos_c, sin_c,
                        layer_weights,
                        lora_adapters,
                        lycoris_adapters,
                        i, b, n_total,
                    )?
                };
                x = result;
            }

            // Evict this layer's weights from self.weights and release the GPU slot.
            if let Some(ref off) = self.offloader {
                let prefix = format!("layers.{}.", i);
                self.weights.retain(|k, _| !k.starts_with(&prefix));
                off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .evict_block();
            }
        }

        // ErnieImageAdaLNContinuous: LayerNorm (no affine) + linear → chunk(scale, shift).
        let x_n = flame_core::layer_norm::layer_norm(&x, &[HIDDEN], None, None, NORM_EPS)?;
        let final_mod = self.linear(&t_emb, "final_norm.linear.weight", Some("final_norm.linear.bias"))?;
        let f_chunks = final_mod.chunk(2, 1)?;
        let final_scale = f_chunks[0].unsqueeze(1)?;
        let final_shift = f_chunks[1].unsqueeze(1)?;
        let x_final = x_n.mul(&final_scale.add_scalar(1.0)?)?.add(&final_shift)?;

        // final_linear: HIDDEN → IN_C * PATCH_SIZE^2 = 128 (patch=1).
        let projected = self.linear(&x_final, "final_linear.weight", Some("final_linear.bias"))?;

        // Drop text tokens, reshape [B, n_img, IN_C] → [B, IN_C, H, W] (patch_size=1).
        let img_only = projected.narrow(1, 0, n_img)?.contiguous()?;
        let h = dims[2];
        let w = dims[3];
        img_only.permute(&[0, 2, 1])?.contiguous()?.reshape(&[b, IN_C, h, w]).map_err(Into::into)
    }
}

fn rh(x: &Tensor, b: usize, n: usize) -> Result<Tensor> {
    x.reshape(&[b, n, HEADS, HEAD_DIM])?.permute(&[0,2,1,3]).map_err(Into::into)
}
/// Diffusers ErnieImage rotate-half RoPE on full HEAD_DIM with interleaved-doubled freqs.
/// Reference: transformer_ernie_image.py apply_rotary_emb (rotary_interleaved=False).
///   x_first = x[..., :half], x_second = x[..., half:]
///   x_rot = cat([-x_second, x_first], dim=-1)
///   out = x * cos + x_rot * sin
fn rope(q: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let q_bf16 = q.to_dtype(DType::BF16)?.contiguous()?;
    let half = HEAD_DIM / 2;
    let x_first = q_bf16.narrow(3, 0, half)?.contiguous()?;
    let x_second = q_bf16.narrow(3, half, half)?.contiguous()?;
    let neg_second = x_second.mul_scalar(-1.0f32)?;
    let x_rot = Tensor::cat(&[&neg_second, &x_first], 3)?;
    let prod_cos = q_bf16.mul(cos)?;
    let prod_sin = x_rot.mul(sin)?;
    prod_cos.add(&prod_sin).map_err(Into::into)
}

#[allow(dead_code)]
fn dbg_stats(name: &str, t: &Tensor) {
    if let Ok(v) = t.to_dtype(DType::F32).and_then(|t| t.to_vec()) {
        let mut nan = 0usize;
        let mut inf = 0usize;
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &x in &v {
            if x.is_nan() { nan += 1; }
            else if x.is_infinite() { inf += 1; }
            else {
                if x < mn { mn = x; }
                if x > mx { mx = x; }
                sum += x as f64;
            }
        }
        let n = v.len() as f64;
        let mean = sum / n.max(1.0);
        eprintln!("[stat] {name}: shape={:?} n={} nan={} inf={} min={:.4} max={:.4} mean={:.4}",
            t.shape().dims(), v.len(), nan, inf, mn, mx, mean);
    } else {
        eprintln!("[stat] {name}: <to_vec failed>");
    }
}

/// Standalone single-layer forward — used inside `AutogradContext::checkpoint`
/// to drop per-layer intermediates from the autograd tape (recomputed on
/// backward). Mirrors the in-place layer body from `ErnieModel::forward` but
/// takes explicit weight + LoRA inputs so the closure can satisfy `'static`.
/// Returns `flame_core::Result` so it slots directly into AutogradContext.
#[allow(clippy::too_many_arguments)]
fn ernie_layer_forward_standalone(
    x: Tensor,
    sc_msa: Tensor, s_msa: Tensor, g_msa: Tensor,
    sc_mlp: Tensor, s_mlp: Tensor, g_mlp: Tensor,
    cos_b: Tensor, sin_b: Tensor,
    layer_weights: HashMap<String, Tensor>,
    lora_adapters: Option<Vec<LoRALinear>>,
    lycoris_adapters: Option<Vec<Option<Arc<LycorisLinear>>>>,
    layer_idx: usize,
    b: usize,
    n_total: usize,
) -> flame_core::Result<Tensor> {
    let pre = format!("layers.{}.self_attention", layer_idx);

    let w = |key: &str| -> flame_core::Result<&Tensor> {
        layer_weights.get(key).ok_or_else(||
            flame_core::FlameError::InvalidInput(format!("ernie layer {}: missing weight {}", layer_idx, key)))
    };
    let linear_no_lora = |x: &Tensor, w_key: &str| -> flame_core::Result<Tensor> {
        x.matmul(&w(w_key)?.transpose()?)
    };
    // Phase 2b dispatch: prefer LyCORIS adapter at this slot, else fall back
    // to the legacy LoRALinear. Mirrors ErnieModel::adapter_for but inlined
    // since the closure captures Vecs by value (must be 'static for
    // AutogradContext::checkpoint).
    let linear_lora = |x: &Tensor, w_key: &str, adapter_idx: usize| -> flame_core::Result<Tensor> {
        let base = linear_no_lora(x, w_key)?;
        if let Some(ref lyc) = lycoris_adapters {
            if let Some(Some(adapter)) = lyc.get(adapter_idx) {
                let delta = adapter.forward_delta(x)
                    .map_err(|e| flame_core::FlameError::InvalidInput(format!("lycoris delta: {e}")))?;
                return base.add(&delta);
            }
        }
        if let Some(ref adapters) = lora_adapters {
            let delta = adapters[adapter_idx].forward_delta(x)
                .map_err(|e| flame_core::FlameError::InvalidInput(format!("lora delta: {e}")))?;
            base.add(&delta)
        } else {
            Ok(base)
        }
    };
    let rms_norm_full = |x: &Tensor, scale_key: &str| -> flame_core::Result<Tensor> {
        flame_core::norm::rms_norm(x, &[HIDDEN], Some(w(scale_key)?), NORM_EPS)
    };
    let qk_rms_norm_local = |x: &Tensor, scale_key: &str| -> flame_core::Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let x_h = x.reshape(&[batch * HEADS, HEAD_DIM])?;
        let n = flame_core::norm::rms_norm(&x_h, &[HEAD_DIM], Some(w(scale_key)?), NORM_EPS)?;
        n.reshape(&dims)
    };
    let rh_local = |x: &Tensor| -> flame_core::Result<Tensor> {
        x.reshape(&[b, n_total, HEADS, HEAD_DIM])?.permute(&[0,2,1,3])
    };
    let rope_local = |q: &Tensor| -> flame_core::Result<Tensor> {
        let q_bf16 = q.to_dtype(DType::BF16)?.contiguous()?;
        let half = HEAD_DIM / 2;
        let x_first = q_bf16.narrow(3, 0, half)?.contiguous()?;
        let x_second = q_bf16.narrow(3, half, half)?.contiguous()?;
        let neg_second = x_second.mul_scalar(-1.0f32)?;
        let x_rot = Tensor::cat(&[&neg_second, &x_first], 3)?;
        let prod_cos = q_bf16.mul(&cos_b)?;
        let prod_sin = x_rot.mul(&sin_b)?;
        prod_cos.add(&prod_sin)
    };

    let r = x.clone();
    let n = rms_norm_full(&x, &format!("layers.{}.adaLN_sa_ln.weight", layer_idx))?;
    let m = n.mul(&sc_msa.add_scalar(1.0)?)?.add(&s_msa)?;
    let q = linear_lora(&m, &format!("{}.to_q.weight", pre), 0)?;
    let k = linear_lora(&m, &format!("{}.to_k.weight", pre), 1)?;
    let v = linear_lora(&m, &format!("{}.to_v.weight", pre), 2)?;
    let q_n = qk_rms_norm_local(&q, &format!("{}.norm_q.weight", pre))?;
    let k_n = qk_rms_norm_local(&k, &format!("{}.norm_k.weight", pre))?;
    let (qh, kh, vh) = (rh_local(&q_n)?, rh_local(&k_n)?, rh_local(&v)?);
    let (qh, kh) = (rope_local(&qh)?, rope_local(&kh)?);
    let attn = flame_core::attention::sdpa(&qh, &kh, &vh, None)?
        .permute(&[0,2,1,3])?.reshape(&[b, n_total, HIDDEN])?;
    let out = linear_lora(&attn, &format!("{}.to_out.0.weight", pre), 3)?;
    let x = r.add(&g_msa.mul(&out)?)?;

    let r2 = x.clone();
    let n2 = rms_norm_full(&x, &format!("layers.{}.adaLN_mlp_ln.weight", layer_idx))?;
    let m2 = n2.mul(&sc_mlp.add_scalar(1.0)?)?.add(&s_mlp)?;
    let mlp = format!("layers.{}.mlp", layer_idx);
    let gate = linear_lora(&m2, &format!("{}.gate_proj.weight", mlp), 4)?.gelu()?;
    let up = linear_lora(&m2, &format!("{}.up_proj.weight", mlp), 5)?;
    let gated = up.mul(&gate)?;
    let down = linear_lora(&gated, &format!("{}.linear_fc2.weight", mlp), 6)?;
    r2.add(&g_mlp.mul(&down)?)
}

/// Inference-fast block forward — mirrors `ernie_layer_forward_standalone` but
/// borrows `weights` directly (no HashMap clone, no checkpoint closure overhead).
/// Only called from the inference branch (`!AutogradContext::is_recording()`).
///
/// Operation order is identical to `ernie_layer_forward_standalone`:
///   AdaLN-pre SA → gated residual → AdaLN-pre SwiGLU → gated residual.
#[allow(clippy::too_many_arguments)]
fn block_forward_iflame(
    x: &Tensor,
    sc_msa: &Tensor, s_msa: &Tensor, g_msa: &Tensor,
    sc_mlp: &Tensor, s_mlp: &Tensor, g_mlp: &Tensor,
    cos_b: &Tensor, sin_b: &Tensor,
    weights: &HashMap<String, Tensor>,
    lora_adapters: Option<&[LoRALinear]>,
    lycoris_adapters: Option<&[Option<Arc<LycorisLinear>>]>,
    layer_idx: usize,
    b: usize,
    n_total: usize,
) -> crate::Result<Tensor> {
    let pre = format!("layers.{}.self_attention", layer_idx);

    let w = |key: &str| -> crate::Result<&Tensor> {
        weights.get(key).ok_or_else(||
            crate::EriDiffusionError::Model(format!("ernie block {}: missing weight {}", layer_idx, key)))
    };
    let linear_no_lora = |x: &Tensor, w_key: &str| -> crate::Result<Tensor> {
        Ok(x.matmul(&w(w_key)?.transpose()?)?)
    };
    // Phase 2b dispatch — see ernie_layer_forward_standalone for rationale.
    let linear_lora = |x: &Tensor, w_key: &str, adapter_idx: usize| -> crate::Result<Tensor> {
        let base = linear_no_lora(x, w_key)?;
        if let Some(lyc) = lycoris_adapters {
            if let Some(Some(adapter)) = lyc.get(adapter_idx) {
                let delta = adapter.forward_delta(x)?;
                return Ok(base.add(&delta)?);
            }
        }
        if let Some(adapters) = lora_adapters {
            if let Some(adapter) = adapters.get(adapter_idx) {
                let delta = adapter.forward_delta(x)?;
                return Ok(base.add(&delta)?);
            }
        }
        Ok(base)
    };
    let rms_norm_full = |x: &Tensor, scale_key: &str| -> crate::Result<Tensor> {
        Ok(flame_core::norm::rms_norm(x, &[HIDDEN], Some(w(scale_key)?), NORM_EPS)?)
    };
    let qk_rms_norm_local = |x: &Tensor, scale_key: &str| -> crate::Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let x_h = x.reshape(&[batch * HEADS, HEAD_DIM])?;
        let n = flame_core::norm::rms_norm(&x_h, &[HEAD_DIM], Some(w(scale_key)?), NORM_EPS)?;
        Ok(n.reshape(&dims)?)
    };
    let rh_local = |x: &Tensor| -> crate::Result<Tensor> {
        Ok(x.reshape(&[b, n_total, HEADS, HEAD_DIM])?.permute(&[0,2,1,3])?)
    };
    // Rotate-half RoPE: x * cos + [-x[half:], x[:half]] * sin.
    // cos/sin shape: [1, 1, total, HEAD_DIM] — broadcasts over [B, H, total, HEAD_DIM].
    let rope_local = |q: &Tensor| -> crate::Result<Tensor> {
        let q_bf16 = q.to_dtype(DType::BF16)?.contiguous()?;
        let half = HEAD_DIM / 2;
        let x_first = q_bf16.narrow(3, 0, half)?.contiguous()?;
        let x_second = q_bf16.narrow(3, half, half)?.contiguous()?;
        let neg_second = x_second.mul_scalar(-1.0f32)?;
        let x_rot = Tensor::cat(&[&neg_second, &x_first], 3)?;
        let prod_cos = q_bf16.mul(cos_b)?;
        let prod_sin = x_rot.mul(sin_b)?;
        Ok(prod_cos.add(&prod_sin)?)
    };

    // Self-attention path
    let r = x.clone();
    let n = rms_norm_full(x, &format!("layers.{}.adaLN_sa_ln.weight", layer_idx))?;
    let m = n.mul(&sc_msa.add_scalar(1.0)?)?.add(s_msa)?;
    let q = linear_lora(&m, &format!("{}.to_q.weight", pre), 0)?;
    let k = linear_lora(&m, &format!("{}.to_k.weight", pre), 1)?;
    let v = linear_lora(&m, &format!("{}.to_v.weight", pre), 2)?;
    let q_n = qk_rms_norm_local(&q, &format!("{}.norm_q.weight", pre))?;
    let k_n = qk_rms_norm_local(&k, &format!("{}.norm_k.weight", pre))?;
    let (qh, kh, vh) = (rh_local(&q_n)?, rh_local(&k_n)?, rh_local(&v)?);
    let (qh, kh) = (rope_local(&qh)?, rope_local(&kh)?);
    let attn = flame_core::attention::sdpa(&qh, &kh, &vh, None)?
        .permute(&[0,2,1,3])?.reshape(&[b, n_total, HIDDEN])?;
    let out = linear_lora(&attn, &format!("{}.to_out.0.weight", pre), 3)?;
    let x = r.add(&g_msa.mul(&out)?)?;

    // FFN path
    let r2 = x.clone();
    let n2 = rms_norm_full(&x, &format!("layers.{}.adaLN_mlp_ln.weight", layer_idx))?;
    let m2 = n2.mul(&sc_mlp.add_scalar(1.0)?)?.add(s_mlp)?;
    let mlp = format!("layers.{}.mlp", layer_idx);
    let gate = linear_lora(&m2, &format!("{}.gate_proj.weight", mlp), 4)?.gelu()?;
    let up = linear_lora(&m2, &format!("{}.up_proj.weight", mlp), 5)?;
    let gated = up.mul(&gate)?;
    let down = linear_lora(&gated, &format!("{}.linear_fc2.weight", mlp), 6)?;
    Ok(r2.add(&g_mlp.mul(&down)?)?)
}

/// LoRA adapter slot ↔ base-module key mapping. Order matches `ErnieModel::load`.
/// Layer index `i` of LAYERS contributes 7 adapters at offsets [0..7].
const LORA_SLOT_KEYS: [&str; 7] = [
    "self_attention.to_q",
    "self_attention.to_k",
    "self_attention.to_v",
    "self_attention.to_out.0",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.linear_fc2",
];

impl TrainableModel for ErnieModel {
    fn forward(&mut self, noisy: &Tensor, timestep: &Tensor, context: &[Tensor], _p: Option<&Tensor>) -> Result<Tensor> {
        let txt = context.first().ok_or_else(|| crate::EriDiffusionError::Model("Ernie needs text embeddings".into()))?.clone();
        ErnieModel::forward(self, noisy, &txt, timestep)
    }
    fn parameters(&self) -> Vec<Parameter> { self.parameters.clone() }
    fn post_optimizer_step(&mut self) {}

    fn save_weights(&self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "save_weights for non-LoRA Ernie not implemented yet".into(),
            ));
        }
        let mut out = std::collections::HashMap::new();
        if self.algo == LycorisAlgo::None {
            for (i, adapter) in self.lora_adapters.iter().enumerate() {
                let layer_idx = i / 7;
                let slot = i % 7;
                let prefix = format!("layers.{}.{}", layer_idx, LORA_SLOT_KEYS[slot]);
                adapter.save_tensors(&prefix, &mut out)?;
            }
        } else {
            for (i, slot_opt) in self.lycoris_adapters.iter().enumerate() {
                if let Some(adapter) = slot_opt {
                    let layer_idx = i / 7;
                    let slot = i % 7;
                    let prefix = format!("layers.{}.{}", layer_idx, LORA_SLOT_KEYS[slot]);
                    for (leaf, t) in adapter.named_tensors() {
                        out.insert(format!("{prefix}.{leaf}"), t);
                    }
                }
            }
        }
        flame_core::serialization::save_file(&out, std::path::Path::new(path))
            .map_err(|e| crate::EriDiffusionError::Safetensors(format!("save_file: {e}")))?;
        Ok(())
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "load_weights for non-LoRA Ernie not implemented yet".into(),
            ));
        }
        let source = flame_core::serialization::load_file(
            std::path::Path::new(path),
            &self.device,
        ).map_err(|e| crate::EriDiffusionError::Safetensors(format!("load_file: {e}")))?;
        if self.algo == LycorisAlgo::None {
            for (i, adapter) in self.lora_adapters.iter().enumerate() {
                let layer_idx = i / 7;
                let slot = i % 7;
                let prefix = format!("layers.{}.{}", layer_idx, LORA_SLOT_KEYS[slot]);
                adapter.load_tensors(&prefix, &source)?;
            }
        } else {
            // Phase 2b: LyCORIS resume not yet wired (parity with chroma's
            // staged rollout — chroma exposes load_named_tensors only on
            // LycorisLinear, not via TrainableModel::load_weights). Defer
            // until train_ernie's --resume-full path needs it.
            return Err(crate::EriDiffusionError::Model(format!(
                "load_weights: LyCORIS algo='{}' resume not yet wired for ernie",
                self.algo.as_str(),
            )));
        }
        Ok(())
    }
}

impl ErnieModel {
    /// Canonical (name, Parameter) pairs for full-checkpoint save/resume.
    /// Mirrors `<ErnieModel as TrainableModel>::save_weights` ordering exactly:
    /// `layers.{layer_idx}.{LORA_SLOT_KEYS[slot]}.lora_{A,B}.weight`, with
    /// adapters indexed `i = layer_idx * 7 + slot`. Order is deterministic
    /// (the adapter Vec's natural order).
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::with_capacity(self.lora_adapters.len() * 2);
        if self.algo == LycorisAlgo::None {
            for (i, adapter) in self.lora_adapters.iter().enumerate() {
                let layer_idx = i / 7;
                let slot = i % 7;
                let prefix = format!("layers.{}.{}", layer_idx, LORA_SLOT_KEYS[slot]);
                out.push((format!("{prefix}.lora_A.weight"), adapter.lora_a().clone()));
                out.push((format!("{prefix}.lora_B.weight"), adapter.lora_b().clone()));
            }
        } else {
            // Phase 2b: zip named_tensors() leaf names with to_parameters()
            // (matching order — same convention as chroma::named_parameters).
            for (i, slot_opt) in self.lycoris_adapters.iter().enumerate() {
                if let Some(adapter) = slot_opt {
                    let layer_idx = i / 7;
                    let slot = i % 7;
                    let prefix = format!("layers.{}.{}", layer_idx, LORA_SLOT_KEYS[slot]);
                    let params = adapter.to_parameters();
                    let names = adapter.named_tensors();
                    for (param, (leaf, _)) in params.into_iter().zip(names.into_iter()) {
                        out.push((format!("{prefix}.{leaf}"), param));
                    }
                }
            }
        }
        out
    }
}
