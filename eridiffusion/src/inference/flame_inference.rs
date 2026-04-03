//! Inference implementation for Flux and SD3.5 using tokenizers and safetensors

use crate::inference::mmdit_streaming::{build_streaming_config, StreamingMMDiT};
use crate::loaders::lazy_safetensors::LazySafetensorsLoader;
use crate::loaders::load_mmdit_weights;
use crate::loaders::weight_loader::WeightLoader;
use crate::models::{
    flux_complete::{FluxConfig, FluxModel},
    mmdit_blocks::{ArenaScratch, MMDiT, MMDiTConfig},
    text_encoder_complete::{
        CLIPConfig, CLIPTextEncoder as FlameClipTextEncoder, T5Config, T5Encoder,
    },
    text_encoders_cpu::TextEncodersCpuSnapshot,
    vae_complete::{AutoEncoderKL, VAEConfig},
};
use flame_core::device::Device;
use flame_core::memory_pool::MEMORY_POOL;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Text encoder types
pub enum TextEncoderType {
    ClipL,
    ClipG,
    T5XXL,
}

/// Combined text encoder for multi-model encoding
pub struct TextEncoders {
    pub clip_l: Option<CLIPTextEncoder>,
    pub clip_g: Option<CLIPTextEncoder>,
    pub t5_xxl: Option<T5TextEncoder>,
    clip_l_path: Option<PathBuf>,
    clip_g_path: Option<PathBuf>,
    t5_path: Option<PathBuf>,
    cpu_snapshots: Option<crate::models::text_encoders_cpu::TextEncodersCpuSnapshot>,
    pub device: Device,
}

impl TextEncoders {
    /// Load text encoders from safetensors files
    pub fn from_safetensors(
        clip_l_path: Option<&Path>,
        clip_g_path: Option<&Path>,
        t5_xxl_path: Option<&Path>,
        device: Device,
    ) -> flame_core::Result<Self> {
        let preload = should_keep_text_models_loaded();
        let offload = !preload && should_offload_text_models();
        // Disable CPU snapshots to save RAM. For one-shot inference, reloading from disk is fine.
        let cpu_snapshots = None;

        let (clip_l, clip_l_path_buf) = if let Some(path) = clip_l_path {
            if preload {
                (
                    Some(CLIPTextEncoder::from_safetensors(path, device.clone())?),
                    Some(path.to_path_buf()),
                )
            } else {
                (None, Some(path.to_path_buf()))
            }
        } else {
            (None, None)
        };

        let (clip_g, clip_g_path_buf) = if let Some(path) = clip_g_path {
            if preload {
                (
                    Some(CLIPTextEncoder::from_safetensors(path, device.clone())?),
                    Some(path.to_path_buf()),
                )
            } else {
                (None, Some(path.to_path_buf()))
            }
        } else {
            (None, None)
        };

        let (t5_xxl, t5_path_buf) = if let Some(path) = t5_xxl_path {
            if preload {
                (
                    Some(T5TextEncoder::from_safetensors(path, device.clone())?),
                    Some(path.to_path_buf()),
                )
            } else {
                (None, Some(path.to_path_buf()))
            }
        } else {
            (None, None)
        };

        Ok(Self {
            clip_l,
            clip_g,
            t5_xxl,
            clip_l_path: clip_l_path_buf,
            clip_g_path: clip_g_path_buf,
            t5_path: t5_path_buf,
            cpu_snapshots,
            device,
        })
    }

    /// Encode prompt for SD3.5 (uses all three encoders)
    pub fn encode_sd35(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
    ) -> flame_core::Result<(Tensor, Tensor, Tensor, Tensor)> {
        let clip_l_out = if let Some(clip_l) = self.ensure_clip_l()? {
            println!("Encoding with CLIP-L...");
            let ((pos, pos_pooled), (neg, neg_pooled)) = clip_l.encode_pair(prompt, negative_prompt)?;
            Some((ensure_bf16(pos)?, ensure_bf16(neg)?, ensure_bf16(pos_pooled)?, ensure_bf16(neg_pooled)?))
        } else {
            None
        };
        
        if should_offload_text_models() || !should_keep_text_models_loaded() {
            self.clip_l = None;
            MEMORY_POOL.clear_all_caches();
        }

        let clip_g_out = if let Some(clip_g) = self.ensure_clip_g()? {
            println!("Encoding with CLIP-G...");
            let ((pos, pos_pooled), (neg, neg_pooled)) = clip_g.encode_pair(prompt, negative_prompt)?;
            Some((ensure_bf16(pos)?, ensure_bf16(neg)?, ensure_bf16(pos_pooled)?, ensure_bf16(neg_pooled)?))
        } else {
            None
        };

        if should_offload_text_models() || !should_keep_text_models_loaded() {
            self.clip_g = None;
            MEMORY_POOL.clear_all_caches();
        }

        let t5_out = if let Some(t5) = self.ensure_t5()? {
            println!("Encoding with T5-XXL...");
            let (pos, neg) = t5.encode_pair(prompt, negative_prompt)?;
            Some((ensure_bf16(pos)?, ensure_bf16(neg)?))
        } else {
            None
        };

        if should_offload_text_models() || !should_keep_text_models_loaded() {
            self.t5_xxl = None;
            MEMORY_POOL.clear_all_caches();
        }

        // Process CLIP embeddings
        let (clip_l_pos, clip_l_neg, clip_l_pooled, clip_l_neg_pooled) = clip_l_out.ok_or_else(|| Error::InvalidOperation("CLIP-L required".into()))?;
        let (clip_g_pos, clip_g_neg, clip_g_pooled, clip_g_neg_pooled) = clip_g_out.ok_or_else(|| Error::InvalidOperation("CLIP-G required".into()))?;
        
        // Concatenate CLIP-L and CLIP-G along feature dimension (dim 2)
        // CLIP-L: [B, 77, 768]
        // CLIP-G: [B, 77, 1280]
        // Result: [B, 77, 2048]
        let clip_pos = Tensor::cat(&[&clip_l_pos, &clip_g_pos], 2)?;
        let clip_neg = Tensor::cat(&[&clip_l_neg, &clip_g_neg], 2)?;
        
        // Pad to 4096 features
        // We need to pad from 2048 to 4096.
        // Tensor::pad is not available directly in flame_core usually, we might need to use `zeros` and `cat` or `slice_assign`.
        // Let's use `cat` with zeros.
        let pad_shape = Shape::from_dims(&[clip_pos.shape().dims()[0], clip_pos.shape().dims()[1], 4096 - 2048]);
        let zeros = Tensor::zeros_dtype(pad_shape, clip_pos.dtype(), clip_pos.device().clone())?;
        
        let clip_pos_padded = Tensor::cat(&[&clip_pos, &zeros], 2)?;
        let clip_neg_padded = Tensor::cat(&[&clip_neg, &zeros], 2)?;
        
        // Process T5 embeddings
        let (t5_pos, t5_neg) = t5_out.ok_or_else(|| Error::InvalidOperation("T5 required".into()))?;
        // T5: [B, 256/512, 4096]
        
        // Concatenate CLIP and T5 along sequence dimension (dim 1)
        let combined = Tensor::cat(&[&clip_pos_padded, &t5_pos], 1)?;
        let combined_neg = Tensor::cat(&[&clip_neg_padded, &t5_neg], 1)?;
        
        // Concatenate pooled embeddings
        let combined_pooled = Tensor::cat(&[&clip_l_pooled, &clip_g_pooled], 1)?;
        let combined_pooled_neg = Tensor::cat(&[&clip_l_neg_pooled, &clip_g_neg_pooled], 1)?;

        if !should_keep_text_models_loaded() {
            self.release_text_models()?;
        }

        Ok((combined, combined_neg, combined_pooled, combined_pooled_neg))
    }

    /// Encode prompt for Flux (uses CLIP-L and T5-XXL)
    pub fn encode_flux(&mut self, prompt: &str) -> flame_core::Result<(Tensor, Tensor)> {
        // Flux uses CLIP for conditioning and T5 for main encoding
        let clip = self.ensure_clip_l()?.ok_or_else(|| {
            flame_core::Error::InvalidOperation("CLIP-L required for Flux".into())
        })?;
        let (clip_embed, clip_pooled) = clip.encode(prompt)?;
        let clip_pooled = ensure_bf16(clip_pooled)?;

        let t5 = self.ensure_t5()?.ok_or_else(|| {
            flame_core::Error::InvalidOperation("T5-XXL required for Flux".into())
        })?;
        let t5_embed = ensure_bf16(t5.encode(prompt)?)?;

        if !should_keep_text_models_loaded() {
            self.release_text_models()?;
        }

        Ok((clip_pooled, t5_embed))
    }

    fn ensure_clip_l(&mut self) -> flame_core::Result<Option<&CLIPTextEncoder>> {
        if self.clip_l.is_none() {
            if let Some(ref snapshots) = self.cpu_snapshots {
                if let Some(ref snapshot) = snapshots.clip_l {
                    let path = self.clip_l_path.as_ref().ok_or_else(|| {
                        Error::InvalidOperation("CLIP-L path missing for snapshot".into())
                    })?;
                    let tokenizer_path = path.with_extension("tokenizer.json");
                    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                        Error::InvalidOperation(format!(
                            "Failed to load CLIP-L tokenizer {}: {}",
                            tokenizer_path.display(),
                            e
                        ))
                    })?;
                    let text_model = snapshot.instantiate(&self.device)?;
                    let max_length = snapshot.config().max_position_embeddings;
                    self.clip_l = Some(CLIPTextEncoder {
                        tokenizer,
                        text_model,
                        max_length,
                        device: self.device.clone(),
                    });
                }
            } else if let Some(path) = &self.clip_l_path {
                self.clip_l = Some(CLIPTextEncoder::from_safetensors(path, self.device.clone())?);
            }
        }
        Ok(self.clip_l.as_ref())
    }

    fn ensure_clip_g(&mut self) -> flame_core::Result<Option<&CLIPTextEncoder>> {
        if self.clip_g.is_none() {
            if let Some(ref snapshots) = self.cpu_snapshots {
                if let Some(ref snapshot) = snapshots.clip_g {
                    let path = self.clip_g_path.as_ref().ok_or_else(|| {
                        Error::InvalidOperation("CLIP-G path missing for snapshot".into())
                    })?;
                    let tokenizer_path = path.with_extension("tokenizer.json");
                    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                        Error::InvalidOperation(format!(
                            "Failed to load CLIP-G tokenizer {}: {}",
                            tokenizer_path.display(),
                            e
                        ))
                    })?;
                    let text_model = snapshot.instantiate(&self.device)?;
                    let max_length = snapshot.config().max_position_embeddings;
                    self.clip_g = Some(CLIPTextEncoder {
                        tokenizer,
                        text_model,
                        max_length,
                        device: self.device.clone(),
                    });
                }
            } else if let Some(path) = &self.clip_g_path {
                self.clip_g = Some(CLIPTextEncoder::from_safetensors(path, self.device.clone())?);
            }
        }
        Ok(self.clip_g.as_ref())
    }

    fn ensure_t5(&mut self) -> flame_core::Result<Option<&T5TextEncoder>> {
        if self.t5_xxl.is_none() {
            if let Some(ref snapshots) = self.cpu_snapshots {
                if let Some(ref snapshot) = snapshots.t5 {
                    let path = self.t5_path.as_ref().ok_or_else(|| {
                        Error::InvalidOperation("T5 path missing for snapshot".into())
                    })?;
                    self.t5_xxl =
                        Some(T5TextEncoder::from_snapshot(path, snapshot, self.device.clone())?);
                }
            } else if let Some(path) = &self.t5_path {
                println!("Loading T5 from safetensors: {:?}", path);
                self.t5_xxl = Some(T5TextEncoder::from_safetensors(path, self.device.clone())?);
                println!("T5 loaded successfully");
            }
        }
        Ok(self.t5_xxl.as_ref())
    }

    pub fn release_text_models(&mut self) -> flame_core::Result<()> {
        self.clip_l = None;
        self.clip_g = None;
        self.t5_xxl = None;
        self.device.synchronize()?;
        MEMORY_POOL.clear_all_caches();
        Ok(())
    }
}

fn should_keep_text_models_loaded() -> bool {
    matches!(
        env::var("SD35_KEEP_TEXT_MODELS").ok().as_deref(),
        Some("1") | Some("true") | Some("ON") | Some("on")
    )
}

fn should_offload_text_models() -> bool {
    matches!(
        env::var("ENC_OFFLOAD_CPU").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

fn ensure_bf16(tensor: Tensor) -> flame_core::Result<Tensor> {
    if tensor.dtype() == DType::BF16 && tensor.storage_dtype() == DType::BF16 {
        Ok(tensor)
    } else {
        tensor.to_dtype(DType::BF16)
    }
}

fn y_pooled_default_dim() -> usize {
    2048
}

/// CLIP Text Encoder
pub struct CLIPTextEncoder {
    pub tokenizer: Tokenizer,
    pub text_model: FlameClipTextEncoder,
    pub max_length: usize,
    pub device: Device,
}

impl CLIPTextEncoder {
    pub fn from_safetensors(path: &Path, device: Device) -> flame_core::Result<Self> {
        // Load tokenizer
        let tokenizer_path = path.with_extension("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load tokenizer: {}", e))
        })?;

        // Load weights efficiently using WeightLoader (mmap)
        let loader = crate::loaders::WeightLoader::from_safetensors_with_dtype(
            path,
            device.clone(),
            DType::BF16,
        )?;
        let weights = &loader.weights;

        // Determine hidden size directly from embedding weight shapes (authoritative in C++).
        let hidden_size = weights
            .iter()
            .find_map(|(name, tensor)| {
                if name.ends_with("token_embedding.weight") {
                    tensor.shape().dims().get(1).copied()
                } else if name.ends_with("position_embedding.weight") {
                    tensor.shape().dims().get(1).copied()
                } else {
                    None
                }
            })
            .unwrap_or_else(|| if path.to_string_lossy().contains("clip_l") { 768 } else { 1280 });

        // Build CLIP config based on inferred hidden size and weight metadata.
        let mut config = match hidden_size {
            768 => CLIPConfig::clip_l(),
            1280 => CLIPConfig::clip_g(),
            other => {
                let mut cfg = CLIPConfig::clip_l();
                cfg.hidden_size = other;
                cfg.intermediate_size = other * 4;
                cfg.num_attention_heads = other / 64;
                cfg.projection_dim = Some(other);
                cfg
            }
        };
        if let Some(token_weight) = weights.get("text_model.embeddings.token_embedding.weight") {
            config.vocab_size = token_weight.shape().dims()[0];
        }
        if let Some(pos_weight) = weights.get("text_model.embeddings.position_embedding.weight") {
            config.max_position_embeddings = pos_weight.shape().dims()[0];
        }

        let max_length = config.max_position_embeddings;
        let text_model = FlameClipTextEncoder::new(config, device.clone(), weights.clone())?;

        Ok(Self { tokenizer, text_model, max_length, device })
    }

    pub fn encode(&self, text: &str) -> flame_core::Result<(Tensor, Tensor)> {
        // Tokenize
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Tokenization failed: {}", e))
        })?;

        let mut input_ids = encoding.get_ids().to_vec();

        // Pad or truncate to max_length
        input_ids.resize(self.max_length, 0);

        // Convert to tensor
        let input_ids_i64: Vec<i64> = input_ids.into_iter().map(|x| x as i64).collect();
        let input_ids_f32: Vec<f32> = input_ids_i64.iter().map(|&x| x as f32).collect();
        let input_ids = Tensor::from_vec(
            input_ids_f32,
            Shape::from_dims(&[1, self.max_length]),
            self.device.cuda_device_arc(),
        )?
        .to_dtype(DType::I32)?;

        // Encode using the text model
        let output = self.text_model.forward(&input_ids, None)?;
        Ok((output.last_hidden_state, output.pooled_output))
    }

    pub fn encode_pair(
        &self,
        prompt: &str,
        negative: &str,
    ) -> flame_core::Result<((Tensor, Tensor), (Tensor, Tensor))> {
        let pos = self.encode(prompt)?;
        let neg = self.encode(negative)?;
        Ok((pos, neg))
    }
}

/// T5 Text Encoder
pub struct T5TextEncoder {
    pub tokenizer: Tokenizer,
    pub text_model: T5Encoder,
    pub max_length: usize,
    pub device: Device,
}

impl T5TextEncoder {
    pub fn from_safetensors(path: &Path, device: Device) -> flame_core::Result<Self> {
        // Load tokenizer
        let tokenizer_path = path.with_extension("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load tokenizer: {}", e))
        })?;

        // Load weights efficiently using WeightLoader (mmap)
        let weight_loader = crate::loaders::WeightLoader::from_safetensors_with_dtype(
            path,
            device.clone(),
            DType::BF16,
        )?;

        // Create T5 config (T5-XXL for SD3/Flux)
        let config = T5Config::t5_xxl();

        // Create T5Encoder with weights
        let text_model = T5Encoder::new(config, device.clone(), &weight_loader)?;

        Ok(Self {
            tokenizer,
            text_model,
            max_length: 256, // T5 typically uses 256 or 512 tokens. SD3.5 uses 256 by default in Diffusers.
            device,
        })
    }

    pub fn from_snapshot(
        path: &Path,
        snapshot: &crate::models::text_encoders_cpu::T5CpuSnapshot,
        device: Device,
    ) -> flame_core::Result<Self> {
        let tokenizer_path = path.with_extension("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to load tokenizer {}: {}",
                tokenizer_path.display(),
                e
            ))
        })?;

        let text_model = snapshot.instantiate(&device)?;

        Ok(Self { tokenizer, text_model, max_length: 256, device })
    }

    pub fn encode(&self, text: &str) -> flame_core::Result<Tensor> {
        // Tokenize
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Tokenization failed: {}", e))
        })?;

        let mut input_ids = encoding.get_ids().to_vec();

        // Pad or truncate
        input_ids.resize(self.max_length, 0);

        // Convert to tensor
        let input_ids_i64: Vec<i64> = input_ids.into_iter().map(|x| x as i64).collect();
        let input_ids_f32: Vec<f32> = input_ids_i64.iter().map(|&x| x as f32).collect();
        let input_ids = Tensor::from_vec(
            input_ids_f32,
            Shape::from_dims(&[1, self.max_length]),
            self.device.cuda_device_arc(),
        )?
        .to_dtype(DType::I32)?;

        // Encode
        let output = self.text_model.forward(&input_ids)?;
        Ok(output.last_hidden_state)
    }

    pub fn encode_pair(
        &self,
        prompt: &str,
        negative: &str,
    ) -> flame_core::Result<(Tensor, Tensor)> {
        let pos = self.encode(prompt)?;
        let neg = self.encode(negative)?;
        Ok((pos, neg))
    }
}

/// SD3.5 Inference Pipeline
pub struct SD35Pipeline {
    pub vae: AutoEncoderKL,
    pub mmdit: StreamingMMDiT,
    pub text_encoders: Option<TextEncoders>,
    pub scheduler: FlowMatchScheduler,
    pub device: Device,
}

impl SD35Pipeline {
    fn build_components(
        vae_path: &Path,
        mmdit_path: &Path,
        device: &Device,
    ) -> flame_core::Result<(AutoEncoderKL, StreamingMMDiT, FlowMatchScheduler)> {
        let vae_weights = load_safetensors(vae_path)?;
        let vae_config = VAEConfig::sd3();
        let vae = AutoEncoderKL::new(vae_config, device, vae_weights)?;

        let lazy = LazySafetensorsLoader::new(mmdit_path)?;
        let mmdit_meta =
            WeightLoader::infer_mmdit_metadata_from_keys(lazy.keys().map(|k| k.as_str()));
        let mmdit_config = build_streaming_config(&mmdit_meta, &lazy, device)?;
        eprintln!(
            "[sd35] streaming config hidden={} context_dim={} pooled_dim={:?}",
            mmdit_config.hidden_size, mmdit_config.context_dim, mmdit_config.pooled_dim
        );
        let mut mmdit =
            StreamingMMDiT::from_checkpoint(mmdit_config.clone(), mmdit_path, device.clone())
                .map_err(|err| Error::InvalidOperation(err.to_string()))?;
        mmdit.config.context_dim = mmdit_config.context_dim;
        mmdit.config.pooled_dim = mmdit_config.pooled_dim;

        let scheduler = FlowMatchScheduler::new(1000, device.clone());
        Ok((vae, mmdit, scheduler))
    }

    /// Load from safetensors files
    pub fn from_safetensors(
        vae_path: &Path,
        mmdit_path: &Path,
        clip_l_path: &Path,
        clip_g_path: &Path,
        t5_xxl_path: &Path,
        device: Device,
    ) -> flame_core::Result<Self> {
        let (vae, mmdit, scheduler) = Self::build_components(vae_path, mmdit_path, &device)?;
        let text_encoders = TextEncoders::from_safetensors(
            Some(clip_l_path),
            Some(clip_g_path),
            Some(t5_xxl_path),
            device.clone(),
        )?;
        Ok(Self { vae, mmdit, text_encoders: Some(text_encoders), scheduler, device })
    }

    /// Load pipeline components without instantiating text encoders.
    pub fn from_safetensors_without_text(
        vae_path: &Path,
        mmdit_path: &Path,
        device: Device,
    ) -> flame_core::Result<Self> {
        let (vae, mmdit, scheduler) = Self::build_components(vae_path, mmdit_path, &device)?;
        Ok(Self { vae, mmdit, text_encoders: None, scheduler, device })
    }

    /// Generate image from prompt
    pub fn generate(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
        width: usize,
        height: usize,
        num_steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
    ) -> flame_core::Result<Tensor> {
        // Encode prompts
        let encoders = self
            .text_encoders
            .as_mut()
            .ok_or_else(|| Error::InvalidOperation("text encoders not initialised".into()))?;
        let (text_embeds, neg_embeds, pooled_embeds, pooled_neg_embeds) = encoders.encode_sd35(prompt, negative_prompt)?;
        self.generate_with_embeddings(
            &text_embeds,
            &neg_embeds,
            &pooled_embeds,
            &pooled_neg_embeds,
            width,
            height,
            num_steps,
            guidance_scale,
            seed,
        )
    }

    /// Generate image from precomputed text embeddings
    pub fn generate_with_embeddings(
        &mut self,
        text_embeds: &Tensor,
        neg_embeds: &Tensor,
        pooled_embeds: &Tensor,
        pooled_neg_embeds: &Tensor,
        width: usize,
        height: usize,
        num_steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
    ) -> flame_core::Result<Tensor> {
        if let Some(s) = seed {
            // TODO: Implement random seed setting
            // flame_core::random::manual_seed(s);
            let _ = s;
        }

        // Create latents
        let latent_height = height / 8;
        let latent_width = width / 8;
        let mut latents = Tensor::randn(
            Shape::from_dims(&[1, 16, latent_height, latent_width]),
            0.0f32,
            1.0f32,
            self.device.cuda_device_arc(),
        )?;

        // Set timesteps
        let timesteps = self.scheduler.get_timesteps(num_steps);

        // Denoising loop
        // Denoising loop
        for i in 0..num_steps {
            let t = timesteps[i];
            let t_next = timesteps[i + 1];
            let dt = t_next - t;

            // Expand latents for CFG
            let latent_input = Tensor::cat(&[&latents, &latents], 0)?;

            // Create timestep embedding
            let timestep =
                Tensor::full(Shape::from_dims(&[1]), t as f32, self.device.cuda_device().clone())?
                    .unsqueeze(0)?
                    .repeat(&[2, 1])?;

            // Concat positive and negative embeddings
            let text_input = Tensor::cat(&[&neg_embeds, &text_embeds], 0)?;

            // Concat pooled embeddings
            let pooled_input = Tensor::cat(&[&pooled_neg_embeds, &pooled_embeds], 0)?;

            // Predict noise
            // Assuming the fourth parameter is y (pooled embeddings), using zeros as placeholder
            let pooled_dim = self.mmdit.config.pooled_dim.unwrap_or_else(|| y_pooled_default_dim());
            // let y = Tensor::zeros_dtype(
            //     Shape::from_dims(&[2, pooled_dim]),
            //     text_input.dtype(),
            //     text_input.device().clone(),
            // )?;
            let y = pooled_input;
            let timestep_ref =
                if timestep.dtype() == DType::BF16 && timestep.storage_dtype() == DType::BF16 {
                    let scratch = ArenaScratch::from_tensor(&timestep);
                    Some(scratch.copy_from(&timestep)?)
                } else {
                    None
                };
            let text_ref =
                if text_input.dtype() == DType::BF16 && text_input.storage_dtype() == DType::BF16 {
                    let scratch = ArenaScratch::from_tensor(&text_input);
                    Some(scratch.copy_from(&text_input)?)
                } else {
                    None
                };
            let y_ref = if y.dtype() == DType::BF16 && y.storage_dtype() == DType::BF16 {
                let scratch = ArenaScratch::from_tensor(&y);
                Some(scratch.copy_from(&y)?)
            } else {
                None
            };
            let timestep_arg = timestep_ref.as_ref().unwrap_or(&timestep);
            let text_arg = text_ref.as_ref().unwrap_or(&text_input);
            let y_arg = y_ref.as_ref().unwrap_or(&y);
            let noise_pred =
                self.mmdit.forward(&latent_input, timestep_arg, text_arg, Some(y_arg))?;

            // Perform guidance
            let chunks = noise_pred.chunk(2, 0)?;
            let noise_pred_uncond = &chunks[0];
            let noise_pred_cond = &chunks[1];

            let noise_pred = noise_pred_uncond
                .add(&noise_pred_cond.sub(noise_pred_uncond)?.mul_scalar(guidance_scale as f32)?)?;

            // Scheduler step
            latents = self.scheduler.step(&noise_pred, &latents, dt)?; // TODO: Use gradient_map instead of individual tensor

            // Progress callback
            if i % 10 == 0 {
                println!("Step {}/{}", i + 1, num_steps);
            }
        }

        // Decode latents
        self.vae.decode(&latents)
    }
}

/// Flux Inference Pipeline
pub struct FluxPipeline {
    pub vae: AutoEncoderKL,
    pub flux: FluxModel,
    pub text_encoders: TextEncoders,
    pub scheduler: FluxScheduler,
    pub device: Device,
}

impl FluxPipeline {
    /// Load from safetensors files
    pub fn from_safetensors(
        vae_path: &Path,
        flux_path: &Path,
        clip_l_path: &Path,
        t5_xxl_path: &Path,
        device: Device,
    ) -> flame_core::Result<Self> {
        // Load VAE (Flux uses same VAE as SD3.5)
        let vae_weights = load_safetensors(vae_path)?;
        let vae_config = VAEConfig::sd3();
        let vae = AutoEncoderKL::new(vae_config, &device, vae_weights)?;

        // Load Flux model
        let flux_weights = load_safetensors(flux_path)?;
        let flux_config = FluxConfig::flux_dev();
        let flux = FluxModel::new(flux_config, device.clone(), flux_weights)?;

        // Load text encoders
        let text_encoders = TextEncoders::from_safetensors(
            Some(clip_l_path),
            None,
            Some(t5_xxl_path),
            device.clone(),
        )?;

        let scheduler = FluxScheduler::new(device.clone());

        Ok(Self { vae, flux, text_encoders, scheduler, device })
    }

    /// Generate image from prompt
    pub fn generate(
        &mut self,
        prompt: &str,
        width: usize,
        height: usize,
        num_steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
    ) -> flame_core::Result<Tensor> {
        // Set random seed
        if let Some(s) = seed {
            // TODO: Implement random seed setting
            // flame_core::random::manual_seed(s);
        }

        // Encode prompt
        let (clip_embed, t5_embed) = self.text_encoders.encode_flux(prompt)?;

        // Create latents with patchification
        let latent_height = height / 8;
        let latent_width = width / 8;
        let mut latents = Tensor::randn(
            Shape::from_dims(&[1, 16, latent_height, latent_width]),
            0.0f32,
            1.0f32,
            self.device.cuda_device_arc(),
        )?;

        // Patchify for Flux (2x2 patches)
        latents = self.patchify_latents(&latents)?;

        // Set timesteps with shifted sigmoid schedule
        let timesteps = self.scheduler.get_timesteps(num_steps);

        // Denoising loop
        for (i, &t) in timesteps.iter().enumerate() {
            // Create timestep embedding
            let timestep =
                Tensor::full(Shape::from_dims(&[1]), t as f32, self.device.cuda_device().clone())?;

            // Create guidance tensor
            let guidance = Tensor::full(
                Shape::from_dims(&[1]),
                guidance_scale,
                self.device.cuda_device().clone(),
            )?;

            // Flux forward pass
            let noise_pred =
                self.flux.forward(&latents, &timestep, &t5_embed, &clip_embed, Some(&guidance))?;

            // Scheduler step
            latents = self.scheduler.step(&noise_pred, &latents, t)?; // TODO: Use gradient_map instead of individual tensor

            if i % 10 == 0 {
                println!("Step {}/{}", i + 1, num_steps);
            }
        }

        // Unpatchify
        latents = self.unpatchify_latents(&latents, latent_height, latent_width)?;

        // Decode
        self.vae.decode(&latents)
    }

    fn patchify_latents(&self, latents: &Tensor) -> flame_core::Result<Tensor> {
        // Convert [B, 16, H, W] to [B, (H/2)*(W/2), 64]
        let shape = latents.shape();
        let dims = shape.dims();
        if dims.len() != 4 {
            return Err(flame_core::Error::InvalidOperation("Invalid latent shape".into()));
        }
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        // Reshape to patches
        latents
            .reshape(&[batch, 16, height / 2, 2, width / 2, 2])?
            .permute(&[0, 2, 4, 3, 5, 1])?
            .reshape(&[batch, (height / 2) * (width / 2), 64])
    }

    fn unpatchify_latents(
        &self,
        latents: &Tensor,
        height: usize,
        width: usize,
    ) -> flame_core::Result<Tensor> {
        let shape = latents.shape();
        let batch = shape.dims()[0];

        latents
            .reshape(&[batch, height / 2, width / 2, 2, 2, 16])?
            .permute(&[0, 5, 1, 3, 2, 4])?
            .reshape(&[batch, 16, height, width])
    }
}

/// Flow Matching Scheduler for SD3.5
pub struct FlowMatchScheduler {
    pub num_steps: usize,
    pub device: Device,
}

impl FlowMatchScheduler {
    pub fn new(num_steps: usize, device: Device) -> Self {
        Self { num_steps, device }
    }

    pub fn get_timesteps(&self, num_inference_steps: usize) -> Vec<f32> {
        // Use shift parameter for better quality (SD 3.5 default is 3.0)
        let shift = 3.0f32;
        
        let mut timesteps = Vec::with_capacity(num_inference_steps + 1);
        for i in 0..=num_inference_steps {
            let t_linear = 1.0 - (i as f32 / num_inference_steps as f32);
            // Apply shifted sigmoid for better distribution of timesteps
            let t_shifted = 1.0 / (1.0 + (-shift * (2.0 * t_linear - 1.0)).exp());
            timesteps.push(t_shifted);
        }
        timesteps
    }

    pub fn step(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        dt: f32,
    ) -> flame_core::Result<Tensor> {
        // Simple Euler integration: x_{t+dt} = x_t + velocity * dt
        let dt_tensor = Tensor::full(
            model_output.shape().clone(),
            dt,
            model_output.device().clone(),
        )?;
        let step_output = model_output.mul(&dt_tensor)?;
        sample.add(&step_output)
    }
}

/// Flux Scheduler with shifted sigmoid
pub struct FluxScheduler {
    pub device: Device,
}

impl FluxScheduler {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn get_timesteps(&self, num_steps: usize) -> Vec<f32> {
        // Flux uses shifted sigmoid schedule
        let shift = 3.0;

        (0..num_steps)
            .map(|i| {
                let t = i as f32 / (num_steps - 1) as f32;
                let sigmoid_t = 1.0 / (1.0 + (-shift * (2.0 * t - 1.0)).exp());
                sigmoid_t
            })
            .collect()
    }

    pub fn step(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep: f32,
    ) -> flame_core::Result<Tensor> {
        // Similar to flow matching but with Flux-specific scaling
        let dt = 1.0 / self.get_timesteps(50).len() as f32;
        let dt_tensor =
            Tensor::full(model_output.shape().clone(), dt, model_output.device().clone())?;
        let scaled = model_output.mul(&dt_tensor)?;
        sample.sub(&scaled)
    }
}

/// Helper function to load safetensors
fn load_safetensors(path: &Path) -> flame_core::Result<HashMap<String, Tensor>> {
    // Memory-map to avoid loading entire file into RAM
    let file = std::fs::File::open(path)
        .map_err(|e| flame_core::Error::InvalidOperation(format!("Failed to open file: {}", e)))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| flame_core::Error::InvalidOperation(format!("Failed to mmap file: {}", e)))?;

    let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to parse safetensors: {}", e))
    })?;

    let device = Device::cuda(0)?;
    let mut result = HashMap::new();

    for (name, view) in tensors.tensors() {
        let shape = Shape::from_dims(view.shape());
        // Convert based on source dtype to minimize conversions
        let tensor = match view.dtype() {
            safetensors::Dtype::F32 => {
                let data = view.data();
                let float_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Tensor::from_vec(float_data, shape, device.cuda_device_arc())?
            }
            safetensors::Dtype::F16 => {
                let data = view.data();
                let float_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                // Keep FP16 on device to reduce memory if possible
                Tensor::from_vec_dtype(float_data, shape, device.cuda_device().clone(), DType::F16)?
            }
            safetensors::Dtype::BF16 => {
                let data = view.data();
                let float_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                Tensor::from_vec_dtype(
                    float_data,
                    shape,
                    device.cuda_device().clone(),
                    DType::BF16,
                )?
            }
            _ => {
                // Fallback: read as f32
                let data = view.data();
                let float_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Tensor::from_vec(float_data, shape, device.cuda_device_arc())?
            }
        };
        result.insert(name.to_string(), tensor);
    }

    Ok(result)
}

// CLIPTextTransformer is imported from the clip module
// T5TextTransformer is replaced by T5Encoder from text_encoder_complete module

// FluxModel is imported from flux_complete module

// FluxModel already has a complete forward implementation in flux_complete module

// Extension trait for AutoEncoderKL
impl AutoEncoderKL {
    pub fn from_weights(
        weights: HashMap<String, Tensor>,
        config: VAEConfig,
        device: Device,
    ) -> flame_core::Result<Self> {
        // Load weights into VAE
        AutoEncoderKL::new(config, &device, weights)
    }
}

// Extension trait for MMDiT
impl MMDiT {
    pub fn from_weights(
        weights: HashMap<String, Tensor>,
        config: MMDiTConfig,
        device: Arc<CudaDevice>,
    ) -> flame_core::Result<Self> {
        let device_wrapped = Device::from(device.clone());
        let loader = WeightLoader::from_tensor_map(weights, device_wrapped.clone());
        let meta = loader.infer_mmdit_metadata();
        let mut config = config;
        config.qk_norm = meta.qk_norm;
        config.x_self_attn_layers = meta.x_self_attn_layers;
        config.context_dim = 4096;
        config.pooled_dim = Some(2048);
        let mut mmdit = MMDiT::new(config, &device_wrapped)?;
        load_mmdit_weights(&mut mmdit, &loader)?;
        Ok(mmdit)
    }
}

/// Example usage
#[cfg(all(test, feature = "dev-bins"))]
mod tests {
    use super::*;

    #[test]
    fn test_sd35_inference() -> flame_core::Result<()> {
        let device = Device::cuda(0)?;

        // Load pipeline
        let pipeline = SD35Pipeline::from_safetensors(
            Path::new("/path/to/sd35_vae.safetensors"),
            Path::new("/path/to/sd35_mmdit.safetensors"),
            Path::new("/path/to/clip_l.safetensors"),
            Path::new("/path/to/clip_g.safetensors"),
            Path::new("/path/to/t5xxl.safetensors"),
            device,
        )?;

        // Generate image
        let image = pipeline.generate(
            "A beautiful sunset over mountains",
            "",
            1024,
            1024,
            50,
            7.5,
            Some(42),
        )?;

        assert_eq!(image.shape().dims(), &[1, 3, 1024, 1024]);
        Ok(())
    }

    #[test]
    fn test_flux_inference(device: &CudaDevice) -> flame_core::Result<()> {
        let device = Device::cuda(0)?;

        // Load pipeline
        let pipeline = FluxPipeline::from_safetensors(
            Path::new("/path/to/flux_vae.safetensors"),
            Path::new("/path/to/flux.safetensors"),
            Path::new("/path/to/clip_l.safetensors"),
            Path::new("/path/to/t5xxl.safetensors"),
            device,
        )?;

        // Generate image
        let image = pipeline.generate(
            "A futuristic city with flying cars",
            1024,
            1024,
            20, // Flux schnell uses fewer steps
            3.5,
            Some(42),
        )?;

        assert_eq!(image.shape().dims(), &[1, 3, 1024, 1024]);
        Ok(())
    }
}
