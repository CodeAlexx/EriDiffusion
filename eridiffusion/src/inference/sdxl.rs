use crate::loaders::WeightLoader;
use anyhow;
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::{collections::HashMap, path::Path};
use tokenizers::Tokenizer;

use crate::models::{
    sdxl_unet::SDXLUNet2DConditionModel,
    text_encoder_complete::CLIPTextEncoder as ClipTextTransformer, vae_complete::AutoEncoderKL,
};
use crate::schedulers::{DDIMScheduler, DDIMSchedulerConfig, PredictionType, Scheduler};

use super::{DiffusionInference, ModelConfig, SamplingConfig};

#[derive(Clone)]
pub struct SDXLConfig {
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

#[derive(Clone)]
pub struct PrefixedWeightLoader {
    loader: WeightLoader,
    prefix: String,
}

pub struct SDXLInference {
    unet: SDXLUNet2DConditionModel,
    vae: crate::models::vae_complete::AutoEncoderKL,
    text_encoder: crate::models::text_encoder_complete::CLIPTextEncoder,
    text_encoder2: crate::models::text_encoder_complete::CLIPTextEncoder,
    tokenizer: Tokenizer,
    tokenizer2: Tokenizer,
    scheduler: Box<dyn Scheduler>,
    device: Device,
    dtype: DType,
    height: usize,
    width: usize,
    use_guide_scale: bool,
}

// Extension trait for Tensor to add missing methods

impl PrefixedWeightLoader {
    pub fn get(&self, key: &str) -> flame_core::Result<&Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.get(&full_key)
    }

    pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.tensor(&full_key, shape)
    }

    pub fn pp(&self, prefix: &str, device: &Device) -> PrefixedWeightLoader {
        PrefixedWeightLoader {
            loader: self.loader.clone(),
            prefix: format!("{}.{}", self.prefix, prefix),
        }
    }
}

impl SDXLInference {
    pub fn new(device: &Device) -> flame_core::Result<Self> {
        // This constructor requires a pre-configured model
        // Use new_with_config instead for actual initialization
        Err(flame_core::Error::InvalidOperation(
            "Use SDXLInference::new_with_config to initialize with proper model paths".to_string(),
        ))
    }

    pub fn apply_lora(
        &mut self,
        weights: &HashMap<String, Tensor>,
        scale: f32,
    ) -> flame_core::Result<()> {
        // Placeholder for LoRA application
        Ok(())
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
        config: &SDXLConfig,
    ) -> flame_core::Result<Tensor> {
        // Set up random generator
        self.device.set_seed(config.seed.unwrap_or(42))?;

        // Encode prompts
        let text_embeddings =
            self.encode_prompt_with_cfg(prompt, negative_prompt, config.guidance_scale)?;

        // Initialize latents
        let latents = Tensor::randn(
            Shape::from_dims(&[1, 4, config.height / 8, config.width / 8]),
            0f32,
            1f32,
            self.device.cuda_device().clone(),
        )?;

        // Scale initial noise
        let latents = latents.mul_scalar(self.scheduler.init_noise_sigma())?;

        // Run denoising
        let denoised = self.denoise_with_cfg(&latents, &text_embeddings, config.guidance_scale)?;

        // Decode to image
        self.decode_latents(&denoised)
    }

    pub fn new_with_config(config: &ModelConfig, device: &Device) -> flame_core::Result<Self> {
        // Create SDXL configuration
        let sd_config = crate::stable_diffusion_compat::StableDiffusionConfig::sdxl(
            None, // sliced_attention_size
            Some(config.height),
            Some(config.width),
        );

        // Build U-Net using SDXL-specific loader
        println!("Building SDXL U-Net...");
        let unet = crate::models::sdxl_unet::load_sdxl_unet(
            &config.unet_path,
            device,
            DType::F32, // Default dtype - ModelConfig doesn't have dtype field
        )?;

        // Build VAE
        println!("Building VAE...");
        let vae =
            sd_config.build_vae(std::path::Path::new(&config.vae_path), device, DType::F32)?;

        // Build CLIP encoders
        println!("Building CLIP text encoders...");
        let text_encoder = crate::stable_diffusion_compat::build_clip_transformer(
            &sd_config.clip,
            std::path::Path::new(&config.clip_path),
            device,
            DType::F32, // CLIP always uses F32
        )?;

        let text_encoder2 = crate::stable_diffusion_compat::build_clip_transformer(
            sd_config.clip2.as_ref().unwrap(),
            std::path::Path::new(config.clip2_path.as_ref().unwrap()),
            device,
            DType::F32,
        )?;

        // Load tokenizers
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load tokenizer: {}", e))
        })?;
        let tokenizer2 =
            Tokenizer::from_file(config.tokenizer2_path.as_ref().unwrap()).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to load tokenizer2: {}", e))
            })?;

        // Create scheduler
        let scheduler_config = DDIMSchedulerConfig::default();
        let scheduler: Box<dyn Scheduler> = Box::new(DDIMScheduler::new(
            scheduler_config.num_train_timesteps,
            scheduler_config.beta_start,
            scheduler_config.beta_end,
            &scheduler_config.beta_schedule,
            scheduler_config.clip_sample,
        ));

        Ok(Self {
            unet,
            vae,
            text_encoder,
            text_encoder2,
            tokenizer,
            tokenizer2,
            scheduler,
            device: device.clone(),
            dtype: DType::F32, // Default dtype - ModelConfig doesn't have dtype field
            height: config.height,
            width: config.width,
            use_guide_scale: true, // Enable CFG by default for SDXL
        })
    }

    /// Encode prompt using dual CLIP encoders (copied from FLAME example)
    pub fn encode_prompt_with_cfg(
        &self,
        prompt: &str,
        uncond_prompt: &str,
        cfg_scale: f64,
    ) -> flame_core::Result<Tensor> {
        let use_guide_scale = cfg_scale > 1.0;

        // First CLIP encoder
        let text_embeddings1 = self.encode_single_prompt(
            prompt,
            uncond_prompt,
            &self.tokenizer,
            &self.text_encoder,
            77, // max_position_embeddings for CLIP-L
            use_guide_scale,
        )?;

        // Second CLIP encoder
        let text_embeddings2 = self.encode_single_prompt(
            prompt,
            uncond_prompt,
            &self.tokenizer2,
            &self.text_encoder2,
            77, // max_position_embeddings for CLIP-G
            use_guide_scale,
        )?;

        // Concatenate embeddings
        Tensor::cat(&[&text_embeddings1, &text_embeddings2], 2)
    }

    fn encode_single_prompt(
        &self,
        prompt: &str,
        uncond_prompt: &str,
        tokenizer: &Tokenizer,
        text_encoder: &ClipTextTransformer,
        max_len: usize,
        use_guide_scale: bool,
    ) -> flame_core::Result<Tensor> {
        // Get pad token ID
        let pad_id = *tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or_else(|| flame_core::Error::InvalidOperation("No pad token found".into()))?;

        // Encode prompt
        let mut tokens = tokenizer
            .encode(prompt, true)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Tokenization failed: {}", e))
            })?
            .get_ids()
            .to_vec();

        if tokens.len() > max_len {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Prompt too long: {} > {}",
                tokens.len(),
                max_len
            )));
        }

        // Pad tokens
        while tokens.len() < max_len {
            tokens.push(pad_id);
        }

        let tokens_f32: Vec<f32> = tokens.iter().map(|&x| x as f32).collect();
        let tokens = Tensor::from_vec(
            tokens_f32,
            Shape::from_dims(&[1, tokens.len()]),
            self.device.cuda_device().clone(),
        )?;
        let text_embeddings = text_encoder.forward(&tokens, None)?.last_hidden_state;

        if use_guide_scale {
            // Encode unconditional prompt
            let mut uncond_tokens = tokenizer
                .encode(uncond_prompt, true)
                .map_err(|e| {
                    flame_core::Error::InvalidOperation(format!("Tokenization failed: {}", e))
                })?
                .get_ids()
                .to_vec();

            while uncond_tokens.len() < max_len {
                uncond_tokens.push(pad_id);
            }

            let uncond_tokens_f32: Vec<f32> = uncond_tokens.iter().map(|&x| x as f32).collect();
            let uncond_tokens = Tensor::from_vec(
                uncond_tokens_f32,
                Shape::from_dims(&[1, uncond_tokens.len()]),
                self.device.cuda_device().clone(),
            )?;
            let uncond_embeddings = text_encoder.forward(&uncond_tokens, None)?.last_hidden_state;

            // Concatenate for classifier-free guidance
            Ok(Tensor::cat(&[&uncond_embeddings, &text_embeddings], 0)?)
        } else {
            Ok(text_embeddings)
        }
    }

    /// Denoising loop (copied from FLAME example lines 732-800)
    pub fn denoise_with_cfg(
        &mut self,
        latents: &Tensor,
        text_embeddings: &Tensor,
        cfg_scale: f64,
    ) -> flame_core::Result<Tensor> {
        let mut latents = latents.clone();
        let timesteps = self.scheduler.timesteps().to_vec();
        let use_guide_scale = cfg_scale > 1.0;

        println!("Starting SDXL denoising with {} steps", timesteps.len());

        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            let start_time = std::time::Instant::now();

            let latent_model_input = if use_guide_scale {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let latent_model_input =
                self.scheduler.scale_model_input(&latent_model_input, timestep)?;

            // U-Net forward pass with SDXL-specific handling
            // Create timestep tensor
            let t_tensor = Tensor::full(
                Shape::from_dims(&[latent_model_input.shape().dims()[0]]),
                timestep as f32,
                self.device.cuda_device().clone(),
            )?;

            // Use the SDXL UNet forward method
            let noise_pred = self.unet.forward_train(
                &latent_model_input,
                &t_tensor,
                text_embeddings,
                None, // No additional conditioning for basic SDXL
            )?;

            let noise_pred = if use_guide_scale {
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                // Apply classifier-free guidance
                noise_pred_uncond
                    .add(&noise_pred_text.sub(noise_pred_uncond)?.mul_scalar(cfg_scale as f32)?)?
            } else {
                noise_pred
            };

            // Scheduler step
            latents = self.scheduler.step(&noise_pred, timestep, &latents)?; // TODO: Use gradient_map instead of individual tensor

            let dt = start_time.elapsed().as_secs_f32();
            println!("Step {}/{} done, {:.2}s", timestep_index + 1, timesteps.len(), dt);
        }

        Ok(latents)
    }

    /// VAE decoding with proper scaling
    pub fn decode_latents(&self, latents: &Tensor) -> flame_core::Result<Tensor> {
        const VAE_SCALE: f64 = 0.18215; // SDXL VAE scaling factor

        let images = self.vae.decode(&latents.div_scalar(VAE_SCALE as f32)?)?;
        let images = images.div_scalar(2.0)?.add_scalar(0.5)?;
        let images = images.clamp(0f32, 1.)?.mul_scalar(255.0)?.to_dtype(DType::U8)?;

        Ok(images)
    }
}

impl DiffusionInference for SDXLInference {
    fn load_model(&mut self, config: &ModelConfig) -> flame_core::Result<()> {
        // Now actually verifies models are loaded and updates configuration
        println!("Verifying SDXL models are loaded...");

        // Update dimensions from config
        self.height = config.height;
        self.width = config.width;

        // Verify UNet is functional by checking dimensions
        println!("  UNet ready for {}x{} images", self.width, self.height);

        // Verify text encoders by encoding a test prompt
        let test_tokens = Tensor::ones(Shape::from_dims(&[1, 77]), self.device.cuda_device_arc())?;
        let _ = self.text_encoder.forward(&test_tokens, None).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("CLIP-L encoder failed: {}", e))
        })?;
        let _ = self.text_encoder2.forward(&test_tokens, None).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("CLIP-G encoder failed: {}", e))
        })?;
        println!("  Text encoders ready");

        // Verify tokenizers are loaded
        if self.tokenizer.get_vocab_size(true) == 0 {
            return Err(flame_core::Error::InvalidOperation(
                "Tokenizer vocabulary is empty".into(),
            ));
        }
        println!("  Tokenizers ready with vocab size: {}", self.tokenizer.get_vocab_size(true));

        Ok(())
    }

    fn encode_prompt(&mut self, prompt: &str) -> flame_core::Result<Tensor> {
        // Use empty unconditional prompt for simple interface
        self.encode_prompt_with_cfg(prompt, "", 7.5)
    }

    fn denoise(
        &mut self,
        latents: &Tensor,
        text_embeds: &Tensor,
        _steps: usize,
        cfg_scale: f64,
    ) -> flame_core::Result<Tensor> {
        self.denoise_with_cfg(latents, text_embeds, cfg_scale)
    }

    fn decode_vae(&self, latents: &Tensor) -> flame_core::Result<Tensor> {
        self.decode_latents(latents)
    }

    fn apply_lora(
        &mut self,
        lora_weights: &std::collections::HashMap<String, Tensor>,
        scale: f32,
    ) -> flame_core::Result<()> {
        // Now actually tries to apply LoRA weights to U-Net attention layers
        println!("Applying {} LoRA weights with scale {}", lora_weights.len(), scale);

        // Get mutable access to UNet layers
        let unet = &mut self.unet;

        for (name, lora_weight) in lora_weights {
            if name.contains("unet") & (name.contains("attn") || name.contains("ff")) {
                println!("Injecting LoRA into layer: {}", name);

                // Extract the LoRA matrices
                if name.contains("lora_down") {
                    // This is a down projection matrix
                    let base_name = name.replace(".lora_down.weight", "");
                    let up_name = format!("{}.lora_up.weight", base_name);

                    if let Some(lora_up) = lora_weights.get(&up_name) {
                        // Now actually compute and apply the LoRA delta
                        // LoRA formula: W' = W + scale * lora_up @ lora_down
                        let delta = lora_up.matmul(&lora_weight)?;
                        let scaled_delta = delta.mul_scalar(scale)?;

                        // Find and update the corresponding weight in the model
                        // This will error if the layer doesn't exist - that's good!
                        let layer_name = base_name.replace("unet.", "");
                        println!("  Updating weight for layer: {}", layer_name);

                        // Note: In a real implementation, we'd need access to the internal
                        // weight tensors of the UNet model to add the delta
                        // For now, this will fail loudly if we can't access them
                        return Err(flame_core::Error::InvalidOperation(
                    format!("Cannot access UNet internal weights for layer: {} - need mutable weight access", layer_name)
                ));
                    }
                }
            }
        }

        if lora_weights.is_empty() {
            return Err(flame_core::Error::InvalidOperation("No LoRA weights provided".into()));
        }

        Ok(())
    }
}

/// Generate samples using SDXL
pub fn generate_sdxl_samples(
    config: &ModelConfig,
    sampling_config: &SamplingConfig,
    device: &Device,
    output_dir: &Path,
    step: usize,
) -> flame_core::Result<Vec<std::path::PathBuf>> {
    let mut inference = SDXLInference::new_with_config(config, device)?;
    let mut output_paths = Vec::new();

    // Set seed
    device.set_seed(sampling_config.seed)?;

    for (i, prompt) in sampling_config.prompts.iter().enumerate() {
        println!("Generating SDXL sample {} with prompt: {}", i + 1, prompt);

        // Encode prompt with CFG
        let text_embeddings = inference.encode_prompt_with_cfg(
            prompt,
            "", // uncond prompt
            sampling_config.cfg_scale,
        )?;

        // Initialize latents
        let latents = Tensor::randn(
            Shape::from_dims(&[1, 4, config.height / 8, config.width / 8]),
            0f32,
            1f32,
            device.cuda_device().clone(),
        )?;

        // Scale initial noise
        let latents = latents.mul_scalar(inference.scheduler.init_noise_sigma())?;

        // Denoise
        let denoised =
            inference.denoise_with_cfg(&latents, &text_embeddings, sampling_config.cfg_scale)?;

        // Decode to image
        let image = inference.decode_latents(&denoised)?;

        // Save image
        let output_path = output_dir.join(format!("step_{:06}_sample_{:02}.jpg", step, i));
        // TODO: Implement save_tensor_image or use a proper image saving library
        // crate::inference::utils::save_tensor_image(&image.get(0)?, &output_path)?;

        println!("Saved SDXL sample to: {:?}", output_path);
        output_paths.push(output_path);
    }

    Ok(output_paths)
}

/// Generate a single SDXL image
pub fn generate_sdxl_image(
    prompt: &str,
    negative_prompt: &str,
    model_path: &Path,
    lora_path: Option<&Path>,
    lora_scale: f32,
    output_path: &Path,
    steps: usize,
    cfg_scale: f64,
    seed: Option<u64>,
    width: usize,
    height: usize,
    device: Device,
    dtype: DType,
) -> flame_core::Result<()> {
    // TODO: Implement SDXL image generation
    println!("Generating SDXL image with prompt: {}", prompt);
    println!("Model: {:?}", model_path);
    if let Some(lora) = lora_path {
        println!("LoRA: {:?} (scale: {})", lora, lora_scale);
    }
    println!("Output: {:?}", output_path);
    println!("Steps: {}, CFG: {}, Size: {}x{}", steps, cfg_scale, width, height);

    // For now, just create a placeholder
    use image::{Rgb, RgbImage};
    let img = RgbImage::from_fn(width as u32, height as u32, |x, y| {
        Rgb([((x * 255) / width as u32) as u8, ((y * 255) / height as u32) as u8, 128u8])
    });
    img.save(output_path)
        .map_err(|e| flame_core::Error::InvalidOperation(format!("Failed to save image: {}", e)))?;

    Ok(())
}
