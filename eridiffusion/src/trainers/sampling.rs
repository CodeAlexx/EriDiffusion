use anyhow::Result;
use candle_core::{Device, Tensor, DType, Module, D};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use std::path::PathBuf;
use std::fs;

pub struct SD35Sampler {
    vae: AutoEncoderKL,
    device: Device,
    output_dir: PathBuf,
}

impl SD35Sampler {
    pub fn new(vae: AutoEncoderKL, device: Device, output_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&output_dir)?;
        Ok(Self {
            vae,
            device,
            output_dir,
        })
    }
    
    /// Generate samples during training
    pub fn generate_samples(
        &self,
        mmdit: &impl Module,
        text_embeds: &Tensor,      // [batch, seq_len, 4096]
        pooled_embeds: &Tensor,     // [batch, 2048] 
        prompts: &[String],
        step: usize,
        num_inference_steps: usize,
        guidance_scale: f32,
    ) -> Result<Vec<PathBuf>> {
        println!("\n=== Generating validation samples at step {} ===", step);
        
        let mut saved_paths = Vec::new();
        let batch_size = prompts.len();
        
        // SD3.5 uses 16-channel latents
        let latent_channels = 16;
        let height = 1024;
        let width = 1024;
        let latent_height = height / 8;
        let latent_width = width / 8;
        
        // Generate samples for each prompt
        for (idx, prompt) in prompts.iter().enumerate() {
            println!("Generating image {}/{}: {}", idx + 1, prompts.len(), prompt);
            
            // Get embeddings for this prompt
            let prompt_embeds = text_embeds.narrow(0, idx, 1)?;
            let prompt_pooled = pooled_embeds.narrow(0, idx, 1)?;
            
            // Initialize random latent
            let mut latent = Tensor::randn(
                0.0f32,
                1.0f32,
                (1, latent_channels, latent_height, latent_width),
                &self.device,
            )?;
            
            // SD3.5 uses flow matching with shifted timesteps
            let shift = 3.0;
            
            // Sampling loop (simplified Euler method)
            for i in 0..num_inference_steps {
                let t = 1.0 - (i as f32 / (num_inference_steps - 1) as f32);
                
                // Apply time shift for SD3.5
                let shifted_t = (t * 1000.0 + shift) / (1000.0 + shift);
                let timestep = Tensor::new(&[shifted_t * 1000.0], &self.device)?
                    .to_dtype(DType::F32)?;
                
                // Prepare inputs
                let latent_input = if guidance_scale > 1.0 {
                    // Classifier-free guidance - duplicate for conditional and unconditional
                    Tensor::cat(&[&latent, &latent], 0)?
                } else {
                    latent.clone()
                };
                
                let embeds_input = if guidance_scale > 1.0 {
                    // For unconditional, we use empty embeddings
                    let uncond_embeds = Tensor::zeros_like(&prompt_embeds)?;
                    Tensor::cat(&[&prompt_embeds, &uncond_embeds], 0)?
                } else {
                    prompt_embeds.clone()
                };
                
                let pooled_input = if guidance_scale > 1.0 {
                    let uncond_pooled = Tensor::zeros_like(&prompt_pooled)?;
                    Tensor::cat(&[&prompt_pooled, &uncond_pooled], 0)?
                } else {
                    prompt_pooled.clone()
                };
                
                // Prepare timestep for conditional and unconditional
                let timestep_input = if guidance_scale > 1.0 {
                    Tensor::cat(&[&timestep, &timestep], 0)?
                } else {
                    timestep.clone()
                };
                
                // Forward pass through MMDiT
                // Note: This is a simplified interface - actual MMDiT forward may differ
                let velocity_pred = self.mmdit_forward(
                    mmdit,
                    &latent_input,
                    &timestep_input,
                    &embeds_input,
                    &pooled_input,
                )?;
                
                // Apply classifier-free guidance
                let velocity = if guidance_scale > 1.0 {
                    let cond_velocity = velocity_pred.narrow(0, 0, 1)?;
                    let uncond_velocity = velocity_pred.narrow(0, 1, 1)?;
                    
                    // guidance = uncond + scale * (cond - uncond)
                    let diff = cond_velocity.sub(&uncond_velocity)?;
                    uncond_velocity.add(&diff.affine(guidance_scale as f64, 0.0)?)?
                } else {
                    velocity_pred
                };
                
                // Update latent using Euler method
                let dt = 1.0 / num_inference_steps as f32;
                latent = (latent - velocity.affine(dt as f64, 0.0)?)?;
            }
            
            // Decode latent to image
            let image = self.vae.decode(&latent)?;
            
            // Convert to RGB and save
            let path = self.save_image(&image, step, idx, prompt)?;
            saved_paths.push(path);
        }
        
        println!("Generated {} validation images", saved_paths.len());
        Ok(saved_paths)
    }
    
    /// Simplified MMDiT forward for sampling
    fn mmdit_forward(
        &self,
        mmdit: &impl Module,
        latent: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        pooled: &Tensor,
    ) -> Result<Tensor> {
        // This is a placeholder - actual implementation depends on MMDiT interface
        // For now, return a dummy velocity
        Ok(Tensor::randn_like(latent, 0.0, 1.0)?)
    }
    
    /// Save tensor as image
    fn save_image(
        &self,
        image_tensor: &Tensor,
        step: usize,
        idx: usize,
        prompt: &str,
    ) -> Result<PathBuf> {
        // Assume tensor is [1, 3, H, W] in range [-1, 1]
        let image_tensor = image_tensor.squeeze(0)?; // Remove batch dimension
        let (channels, height, width) = (3, 1024, 1024); // SD3.5 default
        
        // Convert to [0, 255] range
        let image_tensor = ((image_tensor + 1.0)? * 127.5)?;
        let image_tensor = image_tensor.clamp(0.0, 255.0)?;
        
        // Convert to u8 data
        let data = image_tensor.flatten_all()?.to_vec1::<f32>()?;
        
        // Create PPM image format (simple, no dependencies)
        let mut ppm_data = format!("P3\n{} {}\n255\n", width, height);
        
        for y in 0..height {
            for x in 0..width {
                let r = data[y * width + x] as u8;
                let g = data[height * width + y * width + x] as u8;
                let b = data[2 * height * width + y * width + x] as u8;
                ppm_data.push_str(&format!("{} {} {} ", r, g, b));
            }
            ppm_data.push('\n');
        }
        
        // Create filename
        let safe_prompt = prompt
            .chars()
            .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { '_' })
            .collect::<String>()
            .trim()
            .chars()
            .take(50)
            .collect::<String>()
            .replace(' ', "_");
        
        let filename = format!("step_{:06}_sample_{:02}_{}.ppm", step, idx, safe_prompt);
        let path = self.output_dir.join(filename);
        
        std::fs::write(&path, ppm_data)?;
        println!("Saved sample to: {}", path.display());
        
        Ok(path)
    }
}

/// Convert PPM to PNG using external tool if available
pub fn convert_ppm_to_png(ppm_path: &PathBuf) -> Result<PathBuf> {
    let png_path = ppm_path.with_extension("png");
    
    // Try to use ImageMagick convert if available
    if let Ok(output) = std::process::Command::new("convert")
        .arg(ppm_path)
        .arg(&png_path)
        .output()
    {
        if output.status.success() {
            // Delete PPM file
            let _ = std::fs::remove_file(ppm_path);
            return Ok(png_path);
        }
    }
    
    // If conversion fails, keep PPM
    Ok(ppm_path.clone())
}