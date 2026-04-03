// FLAME-based SD 3.5 inference implementation
use anyhow::Result;
use eridiffusion::{
    device::cuda_device,
    loaders::unified_loader::UnifiedLoader,
    models::{
        sd35_mmdit_complete::{SD35Config, SD35MMDiT},
        text_encoder_complete::{CLIPConfig, CLIPTextEncoder, T5Config, T5Encoder},
        vae_complete::AutoencoderKL,
    },
    samplers::flame_schedulers::{FlowMatchScheduler, SchedulerConfig},
};
use flame_core::{DType, Device, Shape, Tensor};
use std::collections::HashMap;

#[derive(Debug)]
struct SD35InferenceConfig {
    model_path: String,
    vae_path: String,
    clip_l_path: String,
    clip_g_path: String,
    t5_path: String,
    prompt: String,
    negative_prompt: String,
    width: usize,
    height: usize,
    num_inference_steps: usize,
    guidance_scale: f32,
    seed: u64,
    is_turbo: bool,
}

impl Default for SD35InferenceConfig {
    fn default() -> Self {
        Self {
            model_path: "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors"
                .to_string(),
            vae_path: "/home/alex/SwarmUI/Models/vae/sdxl_vae.safetensors".to_string(), // SD3.5 uses 16ch VAE built-in
            clip_l_path: "/home/alex/SwarmUI/Models/clip/clip_l.safetensors".to_string(),
            clip_g_path: "/home/alex/SwarmUI/Models/clip/clip_g.safetensors".to_string(),
            t5_path: "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors".to_string(),
            prompt: "A futuristic cityscape at sunset".to_string(),
            negative_prompt: "".to_string(),
            width: 1024,
            height: 1024,
            num_inference_steps: 50,
            guidance_scale: 7.0,
            seed: 42,
            is_turbo: false,
        }
    }
}

struct SD35Inference {
    device: Device,
    mmdit: SD35MMDiT,
    vae: AutoencoderKL,
    text_encoder_l: CLIPTextEncoder,
    text_encoder_g: CLIPTextEncoder,
    text_encoder_t5: T5Encoder,
    scheduler: FlowMatchScheduler,
}

impl SD35Inference {
    fn new(config: &SD35InferenceConfig) -> Result<Self> {
        println!("Initializing FLAME-based SD 3.5 inference...");

        // Get cached CUDA device
        let device = cuda_device(0)?;

        // Load SD 3.5 MMDiT model
        println!("Loading SD 3.5 MMDiT...");
        let model_weights = UnifiedLoader::load_safetensors(&config.model_path, &device)?;

        // Configure for Large model
        let mmdit_config = SD35Config {
            patch_size: 2,
            in_channels: 16, // SD3.5 uses 16-channel VAE
            hidden_size: 1536,
            depth: 38,
            num_heads: 24,
            mlp_ratio: 4.0,
            pos_embed_max_size: 192,
        };

        let mmdit = SD35MMDiT::from_weights(model_weights, mmdit_config, &device)?;

        // Load VAE (SD3.5 uses 16-channel VAE, might be included in main model)
        println!("Loading VAE...");
        let vae_weights = UnifiedLoader::load_safetensors(&config.vae_path, &device)?;
        let vae = AutoencoderKL::from_weights(vae_weights, &device)?;

        // Load text encoders
        println!("Loading CLIP-L text encoder...");
        let clip_l_weights = UnifiedLoader::load_safetensors(&config.clip_l_path, &device)?;
        let text_encoder_l = CLIPTextEncoder::new(CLIPConfig::clip_l(), &device, clip_l_weights)?;

        println!("Loading CLIP-G text encoder...");
        let clip_g_weights = UnifiedLoader::load_safetensors(&config.clip_g_path, &device)?;
        let text_encoder_g = CLIPTextEncoder::new(CLIPConfig::clip_g(), &device, clip_g_weights)?;

        println!("Loading T5-XXL text encoder...");
        let t5_weights = UnifiedLoader::load_safetensors(&config.t5_path, &device)?;
        let text_encoder_t5 = T5Encoder::new(T5Config::t5_xxl(), &device, t5_weights)?;

        // Initialize flow matching scheduler
        let scheduler_config = SchedulerConfig {
            num_train_timesteps: 1000,
            shift: if config.is_turbo { 3.0 } else { 1.0 },
            use_dynamic_shifting: !config.is_turbo,
            ..Default::default()
        };
        let scheduler = FlowMatchScheduler::new(scheduler_config);

        Ok(Self { device, mmdit, vae, text_encoder_l, text_encoder_g, text_encoder_t5, scheduler })
    }

    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Result<Tensor> {
        println!("Encoding prompts with triple text encoders...");

        // Encode with CLIP-L
        let prompt_embeds_l = self.text_encoder_l.encode(prompt, 77)?;
        let negative_embeds_l = self.text_encoder_l.encode(negative_prompt, 77)?;

        // Encode with CLIP-G
        let prompt_embeds_g = self.text_encoder_g.encode(prompt, 77)?;
        let negative_embeds_g = self.text_encoder_g.encode(negative_prompt, 77)?;

        // Encode with T5-XXL (uses 256 tokens for SD3.5)
        let prompt_ids_t5 = self.tokenize_t5(prompt, 256)?;
        let negative_ids_t5 = self.tokenize_t5(negative_prompt, 256)?;

        let prompt_embeds_t5 = self.text_encoder_t5.forward(&prompt_ids_t5)?;
        let negative_embeds_t5 = self.text_encoder_t5.forward(&negative_ids_t5)?;

        // Concatenate all embeddings
        // SD3.5 concatenates in order: CLIP-L, CLIP-G, T5
        let prompt_embeds = self.concat_text_embeddings(
            &prompt_embeds_l,
            &prompt_embeds_g,
            &prompt_embeds_t5.last_hidden_state,
        )?;

        let negative_embeds = self.concat_text_embeddings(
            &negative_embeds_l,
            &negative_embeds_g,
            &negative_embeds_t5.last_hidden_state,
        )?;

        // For classifier-free guidance
        let text_embeddings = Tensor::cat(&[&negative_embeds, &prompt_embeds], 0)?;

        Ok(text_embeddings)
    }

    fn tokenize_t5(&self, text: &str, max_length: usize) -> Result<Tensor> {
        // Simplified T5 tokenization - in production would use proper tokenizer
        let tokens = vec![1; max_length]; // Placeholder
        let tensor = Tensor::from_vec(
            tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
            Shape::from(&[1, max_length]),
            &self.device,
        )?;
        Ok(tensor)
    }

    fn concat_text_embeddings(
        &self,
        clip_l: &Tensor,
        clip_g: &Tensor,
        t5: &Tensor,
    ) -> Result<Tensor> {
        // SD3.5 concatenates embeddings along sequence dimension
        // Ensure all have same batch size
        let batch_size = clip_l.shape().dims()[0];

        // Project to same hidden dim if needed
        // For now assume they're compatible

        Tensor::cat(&[clip_l, clip_g, t5], 1)
    }

    fn generate(&mut self, config: &SD35InferenceConfig) -> Result<Tensor> {
        // Encode prompts
        let text_embeddings = self.encode_prompt(&config.prompt, &config.negative_prompt)?;

        // Initialize latents (16 channels for SD3.5)
        let latent_shape = Shape::from(&[1, 16, config.height / 8, config.width / 8]);
        let mut latents = Tensor::randn(latent_shape, DType::F32, &self.device)?;

        // Set timesteps
        self.scheduler.set_timesteps(config.num_inference_steps);
        let timesteps = self.scheduler.timesteps();

        println!("Running {} flow matching steps...", timesteps.len());

        // Flow matching loop
        for (i, &t) in timesteps.iter().enumerate() {
            // Expand latents for classifier-free guidance
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;

            // Create timestep tensor
            let timestep = Tensor::full(Shape::from(&[2]), t as f32, &self.device)?;

            // Predict velocity (SD3.5 uses v-prediction)
            let velocity_pred = self.mmdit.forward(
                &latent_model_input,
                &timestep,
                &text_embeddings,
                None, // pooled_proj for SD3.5
            )?;

            // Perform guidance
            let velocity_chunks = velocity_pred.chunk(2, 0)?;
            let (velocity_uncond, velocity_cond) = (&velocity_chunks[0], &velocity_chunks[1]);
            let velocity_pred = velocity_uncond
                .add(&velocity_cond.sub(velocity_uncond)?.mul_scalar(config.guidance_scale)?)?;

            // Flow matching step
            latents = self.scheduler.step(&velocity_pred, t, &latents)?;

            if i % 10 == 0 {
                println!("Step {}/{}", i + 1, timesteps.len());
            }
        }

        // Decode latents with 16-channel VAE
        println!("Decoding latents to image...");
        let latents_scaled = latents.div_scalar(1.5305)?.add_scalar(0.0609)?; // SD3.5 scaling
        let image = self.vae.decode(&latents_scaled)?;

        // Post-process image
        let image = image.div_scalar(2.0)?.add_scalar(0.5)?;
        let image = image.clamp(0.0, 1.0)?;
        let image = image.mul_scalar(255.0)?;

        Ok(image)
    }
}

fn main() -> Result<()> {
    let config = SD35InferenceConfig {
        prompt: "A majestic dragon flying over a crystal castle, highly detailed, fantasy art"
            .to_string(),
        num_inference_steps: 28, // Turbo uses fewer steps
        is_turbo: false,
        ..Default::default()
    };

    let mut inference = SD35Inference::new(&config)?;
    let image = inference.generate(&config)?;

    println!("Generated image shape: {:?}", image.shape());
    println!("FLAME-based SD 3.5 inference completed successfully!");

    // TODO: Save image to file

    Ok(())
}
