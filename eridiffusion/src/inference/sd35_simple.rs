use crate::inference::flame_inference::{FlowMatchScheduler, TextEncoders};
use crate::inference::mmdit_streaming::build_streaming_config;
use crate::loaders::lazy_safetensors::LazySafetensorsLoader;
use crate::loaders::mmdit_weights::load_mmdit_weights;
use crate::loaders::weight_loader::WeightLoader;
use crate::models::mmdit_blocks::MMDiT;
use crate::models::vae_complete::{AutoEncoderKL, VAEConfig};
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Paths to required SD 3.5 weights.
pub struct Sd35SimplePaths {
    pub vae: PathBuf,
    pub mmdit: PathBuf,
    pub clip_l: PathBuf,
    pub clip_g: PathBuf,
    pub t5: PathBuf,
}

/// Resolve SD3.5 weight locations using SD35_* environment variables.
pub fn resolve_sd35_simple_paths(variant: &str) -> Result<Sd35SimplePaths> {
    let model_root = std::env::var("SD35_MODEL_ROOT")
        .unwrap_or_else(|_| "/home/alex/SwarmUI/Models/diffusion_models".into());
    let clip_root =
        std::env::var("SD35_CLIP_ROOT").unwrap_or_else(|_| "/home/alex/SwarmUI/Models/clip".into());
    let t5_root =
        std::env::var("SD35_T5_ROOT").unwrap_or_else(|_| "/home/alex/SwarmUI/Models/clip".into());

    let mmdit_name = match variant {
        "medium" => "sd35-medium-mmdit.safetensors",
        "large" => "sd35-large-mmdit.safetensors",
        other => {
            return Err(Error::InvalidInput(format!(
                "Unknown SD3.5 variant '{other}'. Expected 'medium' or 'large'."
            )))
        }
    };

    let mmdit = Path::new(&model_root).join(mmdit_name);
    let vae = Path::new(&model_root).join("sd35-vae.safetensors");
    let clip_l = Path::new(&clip_root).join("clip_l.safetensors");
    let clip_g = Path::new(&clip_root).join("clip_g.safetensors");
    let t5 = Path::new(&t5_root).join("t5xxl_fp16.safetensors");

    for path in [&vae, &mmdit, &clip_l, &clip_g, &t5] {
        if !path.exists() {
            return Err(Error::InvalidInput(format!(
                "Required SD3.5 weight file not found: {}",
                path.display()
            )));
        }
    }

    Ok(Sd35SimplePaths { vae, mmdit, clip_l, clip_g, t5 })
}

/// Minimal SD3.5 inference pipeline (no streaming/arena paths).
pub struct Sd35SimpleModel {
    mmdit: MMDiT,
    vae_path: PathBuf,
    scheduler: FlowMatchScheduler,
    device: Device,
}

impl Sd35SimpleModel {
    /// Load eager MMDiT and store VAE path for later.
    pub fn load(paths: &Sd35SimplePaths, device: Device) -> Result<Self> {
        let mmdit = load_eager_mmdit(&paths.mmdit, &device)?;
        println!("MMDiT loaded");
        let scheduler = FlowMatchScheduler::new(1000, device.clone());
        Ok(Self { mmdit, vae_path: paths.vae.clone(), scheduler, device })
    }

    /// Run batched-CFG sampling and return BCHW tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &mut self,
        cond_embed: Tensor,
        uncond_embed: Tensor,
        cond_pooled: Tensor,
        uncond_pooled: Tensor,
        steps: usize,
        cfg_scale: f32,
        width: usize,
        height: usize,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        if width % 8 != 0 || height % 8 != 0 {
            return Err(Error::InvalidInput(
                "SD3.5 requires width/height divisible by 8".into(),
            ));
        }
        if let Some(s) = seed {
            println!("sd35_simple: seed={} (deterministic RNG pending)", s);
        }
        /*
        println!("Loading text encoders...");
        let text_encoders = TextEncoders::new(
            &self.clip_l_path,
            &self.clip_g_path,
            &self.t5_path,
            &self.device,
        )?;

        println!("Encoding prompts...");
        let (prompt_embeds, pooled_prompt_embeds) = text_encoders.encode(prompt)?;
        
        // Split into uncond and cond
        let (uncond_embeds, cond_embeds) = prompt_embeds.chunk(2, 0)?;
        let (uncond_pooled, cond_pooled) = pooled_prompt_embeds.chunk(2, 0)?;

        // Ensure BF16
        let mut text_batch = Tensor::cat(&[&uncond_embeds, &cond_embeds], 0)?;
        text_batch = ensure_bf16(text_batch)?;

        let mut pooled_batch = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;
        pooled_batch = ensure_bf16(pooled_batch)?;
        */

        /*
        let mut cond_embed = ensure_bf16(cond_embed)?;
        let mut uncond_embed = ensure_bf16(uncond_embed)?;
        let mut cond_pooled = ensure_bf16(cond_pooled)?;
        let mut uncond_pooled = ensure_bf16(uncond_pooled)?;
        
        let context_dim = self.mmdit.config().context_dim;
        cond_embed = clamp_to_context(cond_embed, context_dim)?;
        uncond_embed = clamp_to_context(uncond_embed, context_dim)?;
        let mut text_batch = Tensor::cat(&[&uncond_embed, &cond_embed], 0)?;
        text_batch = ensure_bf16(text_batch)?;

        let mut pooled_batch = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;
        pooled_batch = ensure_bf16(pooled_batch)?;
        */

        // Prepare latents
        // Use a fixed seed for reproducibility or load from file if needed.
        // For now, let's use a simple deterministic pattern or zeros for parity check if we want to match Python exactly.
        // But Python diffusers uses randn.
        // Let's generate randn in Rust and save it, then load it in Python.
        
        let mut latents = Tensor::randn(
            Shape::from_dims(&[1, 16, 64, 64]),
            0.0,
            1.0,
            self.device.cuda_device_arc(),
        )?.to_dtype(DType::BF16)?;
        
        // Save initial latents
        {
             let data = latents.to_vec_f32()?;
             let shape = latents.shape().dims().to_vec();
             use safetensors::tensor::{TensorView, Dtype};
             let tensor = TensorView::new(Dtype::F32, shape, &data_as_u8(&data)).unwrap();
             safetensors::serialize_to_file([("init_latents", tensor)], &None, Path::new("debug_latents_init.safetensors")).ok();
        }

        let mut cfg_shape = latents.shape().dims().to_vec();
        cfg_shape[0] = 2;

        let timesteps = self.scheduler.get_timesteps(steps);
        
        // Save text embeddings for Python script to use
        // (Already saved in main, but we need to ensure we use the same ones here if we were loading them)
        // We will assume the Python script loads `debug_text_encoders.safetensors`

        for i in 0..steps {
            let t = timesteps[i];
            let t_next = timesteps[i + 1];
            let dt = t_next - t;
            println!("sd35_simple: step {} / {} (t={:.4}, dt={:.4})", i + 1, steps, t, dt);
            
            let latent_input = latents.expand(&cfg_shape)?;
            let latent_input = ensure_bf16(latent_input)?;
            
            let mut timestep = Tensor::full(
                Shape::from_dims(&[2]),
                t * 1000.0,
                self.device.cuda_device_arc(),
            )?;
            timestep = ensure_bf16(timestep)?;
            
            let mut pooled_batch = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;
            pooled_batch = ensure_bf16(pooled_batch)?;
            
            let mut text_batch = Tensor::cat(&[&uncond_embed, &cond_embed], 0)?;
            text_batch = ensure_bf16(text_batch)?;

            let start = std::time::Instant::now();
            let noise = self.mmdit.forward(&latent_input, &timestep, &text_batch, Some(&pooled_batch))?;
            self.device.synchronize()?;
            println!("  mmdit forward: {:?}", start.elapsed());

            let preds = noise.chunk(2, 0)?;
            if preds.len() != 2 {
                return Err(Error::InvalidOperation(
                    "CFG batch must produce two predictions".into(),
                ));
            }
            let noise_uncond = preds[0].clone_result()?;
            let noise_cond = preds[1].clone_result()?;
            let guided = noise_cond
                .sub(&noise_uncond)?
                .mul_scalar(cfg_scale)?
                .add(&noise_uncond)?;
            let guided = ensure_bf16(guided)?;
            
            // Save noise prediction for step 0
            if i == 0 {
                 let data = guided.to_vec_f32()?;
                 let shape = guided.shape().dims().to_vec();
                 use safetensors::tensor::{TensorView, Dtype};
                 let tensor = TensorView::new(Dtype::F32, shape, &data_as_u8(&data)).unwrap();
                 safetensors::serialize_to_file([("noise_pred_step0", tensor)], &None, Path::new("debug_noise_pred_step0.safetensors")).ok();
            }

            latents = self.scheduler.step(&guided, &latents, dt)?;
            latents = ensure_bf16(latents)?;
            
            // Save latents after step 0
            if i == 0 {
                 let data = latents.to_vec_f32()?;
                 let shape = latents.shape().dims().to_vec();
                 use safetensors::tensor::{TensorView, Dtype};
                 let tensor = TensorView::new(Dtype::F32, shape, &data_as_u8(&data)).unwrap();
                 safetensors::serialize_to_file([("latents_step1", tensor)], &None, Path::new("debug_latents_step1.safetensors")).ok();
                 
                 // Break after 1 step for verification to save time
                 // break; 
            }
            
            // Clear memory pool to prevent OOM
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            unsafe {
                flame_core::tensor_storage::clear_bf16_pool();
            }
            self.device.synchronize()?;
        }

        // Load VAE just in time
        println!("Loading VAE for decoding...");
        println!("Latents shape: {:?}", latents.shape());
        println!("VAE Path: {}", self.vae_path.display());
        let vae_weights = load_tensor_map(&self.vae_path, &self.device)?;
        let vae = crate::models::vae::AutoEncoderKL::new(crate::models::vae::VAEConfig::sd3(), &self.device, vae_weights)?;
        
        // SD3 VAE scaling: (latents / 1.5305) + 0.0609
        let latents_scaled = latents.div_scalar(1.5305)?.add_scalar(0.0609)?;
        let decoded = vae.decode(&latents_scaled)?;
        Ok(decoded)
    }
}

fn load_eager_mmdit(path: &Path, device: &Device) -> Result<MMDiT> {
    let lazy = LazySafetensorsLoader::new(path)?;
    let metadata = WeightLoader::infer_mmdit_metadata_from_keys(lazy.keys().map(|k| k.as_str()));
    let config = build_streaming_config(&metadata, &lazy, device)?;
    println!("Inferred config: hidden_size={}, num_heads={}, depth={}, qk_norm={:?}", 
             config.hidden_size, config.num_heads, config.depth, config.qk_norm);
    let mut mmdit = MMDiT::new(config.clone(), device)?;
    let loader =
        WeightLoader::from_safetensors_with_dtype(path, device.clone(), DType::BF16)?;
    load_mmdit_weights(&mut mmdit, &loader)?;
    for block in mmdit.blocks_mut() {
        block.disable_grads();
    }
    Ok(mmdit)
}

fn load_tensor_map(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let file = std::fs::File::open(path).map_err(|e| {
        Error::InvalidOperation(format!("Failed to open {}: {e}", path.display()))
    })?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::InvalidOperation(format!("Failed to mmap {}: {e}", path.display())))?;
    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| Error::InvalidOperation(format!("Failed to parse safetensors: {e}")))?;

    let mut out = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape = Shape::from_dims(view.shape());
        let tensor = match view.dtype() {
            safetensors::Dtype::F32 => {
                let bytes = view.data();
                let mut values = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Tensor::from_vec(values, shape, device.cuda_device_arc())?
            }
            safetensors::Dtype::F16 => {
                let bytes = view.data();
                let mut values = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    values.push(half::f16::from_bits(bits).to_f32());
                }
                Tensor::from_vec(values, shape, device.cuda_device_arc())?
            }
            safetensors::Dtype::BF16 => {
                let bytes = view.data();
                let mut values = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    values.push(half::bf16::from_bits(bits).to_f32());
                }
                Tensor::from_vec(values, shape, device.cuda_device_arc())?
            }
            other => {
                return Err(Error::InvalidOperation(format!(
                    "Unsupported dtype {:?} in {}",
                    other,
                    path.display()
                )));
            }
        };
        out.insert(name.to_string(), tensor);
    }
    Ok(out)
}

fn ensure_bf16(mut tensor: Tensor) -> Result<Tensor> {
    if tensor.dtype() != DType::BF16 || tensor.storage_dtype() != DType::BF16 {
        tensor = tensor.to_dtype(DType::BF16)?;
    }
    Ok(tensor)
}

fn clamp_to_context(mut tensor: Tensor, context_dim: usize) -> Result<Tensor> {
    let last = tensor.shape().dims().last().copied().unwrap_or(context_dim);
    if last != context_dim {
        let clamp = context_dim.min(last);
        tensor = tensor.narrow(tensor.shape().rank() - 1, 0, clamp)?;
    }
    Ok(tensor)
}

/// Save BCHW tensor (B=1, C=3) to PNG.
pub fn save_tensor_image(tensor: &Tensor, path: &Path) -> Result<()> {
    let dims = tensor.shape().dims();
    println!("save_tensor_image: shape={:?}", dims);
    
    if dims.len() != 4 || dims[0] != 1 || dims[1] < 3 {
        return Err(Error::InvalidShape(format!(
            "Expected tensor shape [1, C>=3, H, W], got {:?}",
            dims
        )));
    }
    
    // Get data as F32 (handles BF16 conversion if needed)
    // Note: to_vec() returns the underlying storage (NCHW), ignoring strides if permuted.
    // So we work with the original NCHW tensor.
    let data = tensor.to_vec()?;
    
    let channels = dims[1];
    let h = dims[2];
    let w = dims[3];
    let hw = h * w;
    
    // Check stats
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean_val = data.iter().sum::<f32>() / data.len() as f32;
    println!("Image Tensor Stats: min={:.4} max={:.4} mean={:.4}", min_val, max_val, mean_val);

    let mut buffer = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            // NCHW indexing: c * (H * W) + y * W + x
            let offset_r = 0 * hw + y * w + x;
            let offset_g = 1 * hw + y * w + x;
            let offset_b = 2 * hw + y * w + x;
            
            let r = data[offset_r];
            let g = data[offset_g];
            let b = data[offset_b];
            
            let dst = ((y * w) + x) * 3;
            
            buffer[dst] = ((r * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            buffer[dst + 1] = ((g * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            buffer[dst + 2] = ((b * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    
    let w_u32 = u32::try_from(w)
        .map_err(|_| Error::InvalidInput(format!("width {w} exceeds PNG limit")))?;
    let h_u32 = u32::try_from(h)
        .map_err(|_| Error::InvalidInput(format!("height {h} exceeds PNG limit")))?;
        
    image::save_buffer_with_format(
        path,
        &buffer,
        w_u32,
        h_u32,
        image::ColorType::Rgb8,
        image::ImageFormat::Png,
    )
    .map_err(|e| Error::InvalidOperation(format!("Failed to save image: {e}")))?;
    Ok(())
}

pub fn encode_prompts(
    paths: &Sd35SimplePaths,
    prompt: &str,
    negative: &str,
    device: Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let cache_path = Path::new("debug_text_encoders.safetensors");
    if cache_path.exists() {
        println!("Loading pre-computed text embeddings from {}...", cache_path.display());
        let file = std::fs::File::open(cache_path).map_err(|e| Error::InvalidOperation(e.to_string()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::InvalidOperation(e.to_string()))?;
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| Error::InvalidOperation(e.to_string()))?;
        
        let load = |name: &str| -> Result<Tensor> {
            let view = tensors.tensor(name).map_err(|e| Error::InvalidOperation(e.to_string()))?;
            let shape = Shape::from_dims(view.shape());
            // Assuming BF16 or F32. The file was saved as F32 or BF16?
            // parity_check_text_encoder.py loaded it.
            // Let's assume it matches what we need or convert.
            // Actually, let's just use load_tensor_map logic or similar.
            // But we need to return Tensors on device.
            
            let data = view.data();
            let tensor = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let values: Vec<f32> = data.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Tensor::from_vec(values, shape, device.cuda_device_arc())?
                }
                safetensors::Dtype::BF16 => {
                     let values: Vec<f32> = data.chunks_exact(2)
                        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect();
                    Tensor::from_vec(values, shape, device.cuda_device_arc())?
                }
                _ => return Err(Error::InvalidOperation("Unsupported dtype".into())),
            };
            // Ensure BF16
            tensor.to_dtype(DType::BF16)
        };
        
        let cond = load("cond")?;
        let uncond = load("uncond")?;
        let cond_pooled = load("cond_pooled")?;
        let uncond_pooled = load("uncond_pooled")?;
        
        println!("Loaded embeddings successfully.");
        return Ok((cond, uncond, cond_pooled, uncond_pooled));
    }

    println!("Loading text encoders...");
    let mut text = TextEncoders::from_safetensors(
        Some(&paths.clip_l),
        Some(&paths.clip_g),
        Some(&paths.t5),
        device.clone(),
    )?;
    println!("Encoding prompts...");
    let (cond, uncond, cond_pooled, uncond_pooled) = text.encode_sd35(prompt, negative)?;
    println!("Encoding done, dropping text encoders...");
    Ok((cond, uncond, cond_pooled, uncond_pooled))
}

fn data_as_u8<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    }
}
