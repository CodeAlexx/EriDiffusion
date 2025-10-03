// FLAME-based SDXL inference implementation
use anyhow::Result;
use eridiffusion::{
    device::cuda_device,
    loaders::unified_loader::UnifiedLoader,
    models::{
        sdxl_unet_complete::UNet2DConditionModel,
        text_encoder_complete::{CLIPConfig, CLIPTextEncoder},
        vae_complete::AutoencoderKL,
    },
    samplers::flame_schedulers::{DDIMScheduler, SchedulerConfig},
};
use flame_core::{DType, Device, Shape, Tensor};
use std::collections::HashMap;

#[derive(Debug)]
struct SDXLInferenceConfig {
    model_path: String,
    vae_path: String,
    clip_l_path: String,
    clip_g_path: String,
    prompt: String,
    negative_prompt: String,
    width: usize,
    height: usize,
    num_inference_steps: usize,
    guidance_scale: f32,
    seed: u64,
}

impl Default for SDXLInferenceConfig {
    fn default() -> Self {
        Self {
            model_path:
                "/home/alex/SwarmUI/Models/diffusion_models/sd_xl_base_1.0_0.9vae.safetensors"
                    .to_string(),
            vae_path: "/home/alex/SwarmUI/Models/vae/sdxl_vae.safetensors".to_string(),
            clip_l_path: "/home/alex/SwarmUI/Models/clip/clip_l.safetensors".to_string(),
            clip_g_path: "/home/alex/SwarmUI/Models/clip/clip_g.safetensors".to_string(),
            prompt: "A beautiful landscape painting".to_string(),
            negative_prompt: "".to_string(),
            width: 1024,
            height: 1024,
            num_inference_steps: 30,
            guidance_scale: 7.5,
            seed: 42,
        }
    }
}

struct SDXLInference {
    device: Device,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder_l: CLIPTextEncoder,
    text_encoder_g: CLIPTextEncoder,
    scheduler: DDIMScheduler,
}

impl SDXLInference {
    fn new(config: &SDXLInferenceConfig) -> Result<Self> {
        println!("Initializing FLAME-based SDXL inference...");

        // Get cached CUDA device
        let device = cuda_device(0)?;

        // Load models using UnifiedLoader
        println!("Loading SDXL UNet...");
        let unet_weights = UnifiedLoader::load_safetensors(&config.model_path, &device)?;
        let unet = UNet2DConditionModel::from_weights(unet_weights, &device)?;

        println!("Loading VAE...");
        let vae_weights = UnifiedLoader::load_safetensors(&config.vae_path, &device)?;
        let vae = AutoencoderKL::from_weights(vae_weights, &device)?;

        println!("Loading CLIP-L text encoder...");
        let clip_l_weights = UnifiedLoader::load_safetensors(&config.clip_l_path, &device)?;
        let text_encoder_l = CLIPTextEncoder::new(CLIPConfig::clip_l(), &device, clip_l_weights)?;

        println!("Loading CLIP-G text encoder...");
        let clip_g_weights = UnifiedLoader::load_safetensors(&config.clip_g_path, &device)?;
        let text_encoder_g = CLIPTextEncoder::new(CLIPConfig::clip_g(), &device, clip_g_weights)?;

        // Initialize scheduler
        let scheduler_config = SchedulerConfig {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".to_string(),
            prediction_type: "epsilon".to_string(),
        };
        let scheduler = DDIMScheduler::new(scheduler_config);

        Ok(Self { device, unet, vae, text_encoder_l, text_encoder_g, scheduler })
    }

    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Result<(Tensor, Tensor)> {
        println!("Encoding prompts with CLIP...");

        // Encode with CLIP-L (77 tokens max)
        let prompt_embeds_l = self.text_encoder_l.encode(prompt, 77)?;
        let negative_embeds_l = self.text_encoder_l.encode(negative_prompt, 77)?;

        // Encode with CLIP-G (77 tokens max)
        let prompt_embeds_g = self.text_encoder_g.encode(prompt, 77)?;
        let negative_embeds_g = self.text_encoder_g.encode(negative_prompt, 77)?;

        // Concatenate embeddings (SDXL uses both)
        let prompt_embeds = Tensor::cat(&[&prompt_embeds_l, &prompt_embeds_g], 2)?;
        let negative_embeds = Tensor::cat(&[&negative_embeds_l, &negative_embeds_g], 2)?;

        // For classifier-free guidance, concatenate negative and positive
        let text_embeddings = Tensor::cat(&[&negative_embeds, &prompt_embeds], 0)?;

        // Create time ids for SDXL conditioning
        let time_ids = self.create_time_ids()?;

        Ok((text_embeddings, time_ids))
    }

    fn create_time_ids(&self) -> Result<Tensor> {
        // SDXL time conditioning: [original_height, original_width, crop_top, crop_left, target_height, target_width]
        let time_ids = vec![1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0];
        let time_ids_tensor = Tensor::from_vec(time_ids, Shape::from(&[1, 6]), &self.device)?;

        // Duplicate for negative and positive prompts
        Tensor::cat(&[&time_ids_tensor, &time_ids_tensor], 0)
    }

    fn generate(&mut self, config: &SDXLInferenceConfig) -> Result<Tensor> {
        // Encode prompts
        let (text_embeddings, time_ids) =
            self.encode_prompt(&config.prompt, &config.negative_prompt)?;

        // Initialize latents
        let latent_shape = Shape::from(&[1, 4, config.height / 8, config.width / 8]);
        let mut latents = Tensor::randn(latent_shape, DType::F32, &self.device)?;

        // Scale initial noise by scheduler init noise sigma
        let init_noise_sigma = self.scheduler.init_noise_sigma();
        latents = latents.mul_scalar(init_noise_sigma)?;

        // Set timesteps
        self.scheduler.set_timesteps(config.num_inference_steps);
        let timesteps = self.scheduler.timesteps();

        println!("Running {} denoising steps...", timesteps.len());

        // Denoising loop
        for (i, &t) in timesteps.iter().enumerate() {
            // Expand latents for classifier-free guidance
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;

            // Scale latents by scheduler scale
            let latent_model_input = self.scheduler.scale_model_input(&latent_model_input, t)?;

            // Create timestep tensor
            let timestep = Tensor::full(Shape::from(&[2]), t as f32, &self.device)?;

            // Predict noise residual
            let added_cond_kwargs = HashMap::from([("time_ids".to_string(), time_ids.clone())]);

            let noise_pred = self.unet.forward(
                &latent_model_input,
                &timestep,
                &text_embeddings,
                Some(&added_cond_kwargs),
            )?;

            // Perform guidance
            let noise_pred_chunks = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_cond) =
                (&noise_pred_chunks[0], &noise_pred_chunks[1]);
            let noise_pred = noise_pred_uncond
                .add(&noise_pred_cond.sub(noise_pred_uncond)?.mul_scalar(config.guidance_scale)?)?;

            // Compute previous noisy sample
            latents = self.scheduler.step(&noise_pred, t, &latents)?;

            if i % 5 == 0 {
                println!("Step {}/{}", i + 1, timesteps.len());
            }
        }

        // Decode latents with VAE
        println!("Decoding latents to image...");
        let latents_scaled = latents.div_scalar(0.13025)?;
        let image = self.vae.decode(&latents_scaled)?;

        // Post-process image
        let image = image.div_scalar(2.0)?.add_scalar(0.5)?;
        let image = image.clamp(0.0, 1.0)?;
        let image = image.mul_scalar(255.0)?;

        Ok(image)
    }
}

fn main() -> Result<()> {
    let config = SDXLInferenceConfig::default();

    let mut inference = SDXLInference::new(&config)?;
    let image = inference.generate(&config)?;

    println!("Generated image shape: {:?}", image.shape());
    println!("FLAME-based SDXL inference completed successfully!");

    // TODO: Save image to file

    Ok(())
}
