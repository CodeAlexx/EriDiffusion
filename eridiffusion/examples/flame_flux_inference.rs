// FLAME-based Flux inference implementation
use anyhow::Result;
use eridiffusion::{
    device::cuda_device,
    loaders::unified_loader::UnifiedLoader,
    models::{
        flux_model_complete::{FluxModel, FluxModelConfig},
        flux_vae::FluxVAE,
        text_encoder_complete::{CLIPConfig, CLIPTextEncoder, T5Config, T5Encoder},
    },
    samplers::flame_schedulers::{FluxScheduler, FluxSchedulerConfig},
};
use flame_core::{DType, Device, Shape, Tensor};

#[derive(Debug)]
struct FluxInferenceConfig {
    model_path: String,
    vae_path: String,
    clip_path: String,
    t5_path: String,
    prompt: String,
    width: usize,
    height: usize,
    num_inference_steps: usize,
    guidance_scale: f32,
    seed: u64,
    is_schnell: bool,
}

impl Default for FluxInferenceConfig {
    fn default() -> Self {
        Self {
            model_path: "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors"
                .to_string(),
            vae_path: "/home/alex/SwarmUI/Models/vae/ae.safetensors".to_string(),
            clip_path: "/home/alex/SwarmUI/Models/clip/clip_l.safetensors".to_string(),
            t5_path: "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors".to_string(),
            prompt: "A cyberpunk cat hacker in a neon-lit server room".to_string(),
            width: 1024,
            height: 1024,
            num_inference_steps: 50,
            guidance_scale: 3.5,
            seed: 42,
            is_schnell: false,
        }
    }
}

struct FluxInference {
    device: Device,
    model: FluxModel,
    vae: FluxVAE,
    text_encoder_clip: CLIPTextEncoder,
    text_encoder_t5: T5Encoder,
    scheduler: FluxScheduler,
}

impl FluxInference {
    fn new(config: &FluxInferenceConfig) -> Result<Self> {
        println!("Initializing FLAME-based Flux inference...");

        // Get cached CUDA device
        let device = cuda_device(0)?;

        // Load Flux model
        println!("Loading Flux model...");
        let model_weights = UnifiedLoader::load_safetensors(&config.model_path, &device)?;

        // Configure model
        let model_config = if config.is_schnell {
            FluxModelConfig::flux_schnell()
        } else {
            FluxModelConfig::flux_dev()
        };

        let model = FluxModel::new(model_config, &device, model_weights)?;

        // Load Flux VAE (16-channel with 2x2 patches)
        println!("Loading Flux VAE...");
        let vae_weights = UnifiedLoader::load_safetensors(&config.vae_path, &device)?;
        let vae = FluxVAE::from_weights(vae_weights, &device)?;

        // Load text encoders
        println!("Loading CLIP text encoder...");
        let clip_weights = UnifiedLoader::load_safetensors(&config.clip_path, &device)?;
        let text_encoder_clip = CLIPTextEncoder::new(CLIPConfig::clip_l(), &device, clip_weights)?;

        println!("Loading T5-XXL text encoder...");
        let t5_weights = UnifiedLoader::load_safetensors(&config.t5_path, &device)?;
        let text_encoder_t5 = T5Encoder::new(T5Config::t5_xxl(), &device, t5_weights)?;

        // Initialize Flux scheduler
        let scheduler_config = FluxSchedulerConfig {
            num_train_timesteps: 1000,
            shift: 1.0,
            use_shifted_sigmoid: true,
            ..Default::default()
        };
        let scheduler = FluxScheduler::new(scheduler_config);

        Ok(Self { device, model, vae, text_encoder_clip, text_encoder_t5, scheduler })
    }

    fn encode_prompt(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        println!("Encoding prompt with CLIP and T5...");

        // Encode with CLIP-L for pooled output
        let clip_output = self.text_encoder_clip.encode(prompt, 77)?;
        let pooled_output = self.get_pooled_clip_output(&clip_output)?;

        // Encode with T5 for sequence embeddings (Flux uses 512 tokens)
        let t5_ids = self.tokenize_t5(prompt, 512)?;
        let t5_output = self.text_encoder_t5.forward(&t5_ids)?;

        Ok((t5_output.last_hidden_state, pooled_output))
    }

    fn get_pooled_clip_output(&self, clip_output: &Tensor) -> Result<Tensor> {
        // Extract pooled output from CLIP (usually last token or EOS position)
        // For simplicity, take mean of sequence
        let pooled = clip_output.mean_keepdim(1, true)?;
        pooled.squeeze(1)
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

    fn patchify_for_flux(&self, x: &Tensor) -> Result<Tensor> {
        // Flux uses 2x2 patches on 16-channel latents
        // Input: [B, 16, H, W] -> Output: [B, H*W/4, 64]
        let shape = x.shape();
        let dims = shape.dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Reshape to extract 2x2 patches
        let x = x.reshape(&[b, c, h / 2, 2, w / 2, 2])?;
        let x = x.permute(&[0, 2, 4, 1, 3, 5])?; // [B, H/2, W/2, C, 2, 2]
        let x = x.reshape(&[b, (h / 2) * (w / 2), c * 4])?; // [B, num_patches, 64]

        Ok(x)
    }

    fn unpatchify_from_flux(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        // Reverse of patchify
        // Input: [B, num_patches, 64] -> Output: [B, 16, H, W]
        let shape = x.shape();
        let dims = shape.dims();
        let b = dims[0];

        let x = x.reshape(&[b, h / 2, w / 2, 16, 2, 2])?;
        let x = x.permute(&[0, 3, 1, 4, 2, 5])?; // [B, C, H/2, 2, W/2, 2]
        let x = x.reshape(&[b, 16, h, w])?;

        Ok(x)
    }

    fn generate(&mut self, config: &FluxInferenceConfig) -> Result<Tensor> {
        // Encode prompt
        let (text_embeddings, pooled_embeddings) = self.encode_prompt(&config.prompt)?;

        // Initialize latents (16 channels for Flux)
        let latent_h = config.height / 8;
        let latent_w = config.width / 8;
        let latent_shape = Shape::from(&[1, 16, latent_h, latent_w]);
        let latents = Tensor::randn(latent_shape, DType::F32, &self.device)?;

        // Patchify latents for Flux
        let mut latents_patched = self.patchify_for_flux(&latents)?;

        // Set timesteps with shifted sigmoid schedule
        self.scheduler.set_timesteps(config.num_inference_steps);
        let timesteps = self.scheduler.timesteps();

        // Create guidance tensor
        let guidance = if config.is_schnell {
            Tensor::full(Shape::from(&[1]), 1.0, &self.device)?
        } else {
            Tensor::full(Shape::from(&[1]), config.guidance_scale, &self.device)?
        };

        println!("Running {} Flux denoising steps...", timesteps.len());

        // Denoising loop
        for (i, &t) in timesteps.iter().enumerate() {
            // Create timestep tensor
            let timestep = Tensor::full(Shape::from(&[1]), t as f32, &self.device)?;

            // Run model forward pass
            let noise_pred = self.model.forward(
                &latents_patched,
                &text_embeddings,
                &timestep,
                &pooled_embeddings,
                Some(&guidance),
            )?;

            // Flux step (no classifier-free guidance needed for Schnell)
            if !config.is_schnell && config.guidance_scale > 1.0 {
                // For Dev model, could implement guidance if needed
                // For now, use the prediction directly
            }

            // Update latents
            latents_patched = self.scheduler.step(&noise_pred, t, &latents_patched)?;

            if i % 10 == 0 {
                println!("Step {}/{}", i + 1, timesteps.len());
            }
        }

        // Unpatchify latents
        let latents_final = self.unpatchify_from_flux(&latents_patched, latent_h, latent_w)?;

        // Decode with VAE
        println!("Decoding latents to image...");
        let latents_scaled = latents_final.div_scalar(0.3611)?.add_scalar(0.1159)?; // Flux scaling
        let image = self.vae.decode(&latents_scaled)?;

        // Post-process image
        let image = image.clamp(-1.0, 1.0)?;
        let image = image.div_scalar(2.0)?.add_scalar(0.5)?;
        let image = image.mul_scalar(255.0)?;

        Ok(image)
    }
}

fn main() -> Result<()> {
    let config = FluxInferenceConfig {
        prompt: "A majestic owl perched on a glowing crystal tree under the aurora borealis, \
                 digital art, highly detailed, 8k resolution"
            .to_string(),
        num_inference_steps: 4, // Schnell uses very few steps
        is_schnell: false,      // Use Dev for better quality
        guidance_scale: 3.5,
        ..Default::default()
    };

    let mut inference = FluxInference::new(&config)?;
    let image = inference.generate(&config)?;

    println!("Generated image shape: {:?}", image.shape());
    println!("FLAME-based Flux inference completed successfully!");

    // TODO: Save image to file

    Ok(())
}
