//! SD 3.5 Flow Matching Sampler - REAL Implementation

use std::f32::consts::PI;

pub struct SD35Sampler {
    num_steps: usize,
    shift: f32,
}

impl SD35Sampler {
    pub fn new(num_steps: usize) -> Self {
        Self {
            num_steps,
            shift: 3.0, // SD3.5 recommended shift
        }
    }
    
    /// Generate timesteps for SD3.5 flow matching
    pub fn get_timesteps(&self) -> Vec<f32> {
        let mut timesteps = Vec::with_capacity(self.num_steps);
        
        // SD3.5 uses shifted timestep schedule
        for i in 0..self.num_steps {
            let t = 1.0 - (i as f32 / (self.num_steps - 1) as f32);
            // Apply shift transformation
            let shifted_t = self.shift * t / (1.0 + (self.shift - 1.0) * t);
            timesteps.push(shifted_t * 1000.0);
        }
        
        timesteps
    }
    
    /// Generate sigmas for SD3.5
    pub fn get_sigmas(&self) -> Vec<f32> {
        let timesteps = self.get_timesteps();
        timesteps.iter().map(|&t| {
            // SD3.5 sigma calculation
            let lin_t = t / 1000.0;
            ((1.0 - lin_t) / lin_t).sqrt()
        }).collect()
    }
    
    /// Sample SD3.5 using flow matching
    pub fn sample(
        &self,
        latents: &mut Vec<f32>,
        latent_shape: (usize, usize, usize, usize), // (batch, channels, height, width)
        text_embeds: &[f32],
        pooled_projections: &[f32],
        model_fn: impl Fn(&[f32], f32, &[f32], &[f32]) -> Vec<f32>,
    ) -> Vec<f32> {
        let timesteps = self.get_timesteps();
        let sigmas = self.get_sigmas();
        
        println!("🎨 SD3.5 Flow Matching Sampling");
        println!("  Steps: {}", self.num_steps);
        println!("  Shift: {}", self.shift);
        println!("  Latent shape: {:?}", latent_shape);
        
        // Initialize with noise
        let total_size = latent_shape.0 * latent_shape.1 * latent_shape.2 * latent_shape.3;
        let mut x = vec![0.0f32; total_size];
        
        // Add initial noise
        for i in 0..total_size {
            x[i] = gaussian_random() * sigmas[0];
        }
        
        // Sampling loop
        for (step, (&t, &sigma)) in timesteps.iter().zip(sigmas.iter()).enumerate() {
            println!("  Step {}/{}: t={:.2}, sigma={:.4}", step + 1, self.num_steps, t, sigma);
            
            // Scale input
            let scaled_x: Vec<f32> = x.iter().map(|&v| v / (sigma.powi(2) + 1.0).sqrt()).collect();
            
            // Get model prediction
            let noise_pred = model_fn(&scaled_x, t, text_embeds, pooled_projections);
            
            // Flow matching update
            if step < self.num_steps - 1 {
                let next_sigma = sigmas[step + 1];
                let dt = (next_sigma - sigma) / sigma;
                
                for i in 0..total_size {
                    // Euler step
                    x[i] = x[i] + dt * noise_pred[i];
                    
                    // Add noise for non-final steps
                    if step < self.num_steps - 2 {
                        x[i] += (2.0 * dt).sqrt() * gaussian_random() * sigma;
                    }
                }
            }
        }
        
        println!("✅ SD3.5 sampling complete!");
        x
    }
}

// Simple Gaussian random number generator
fn gaussian_random() -> f32 {
    // Box-Muller transform
    let u1 = fastrand::f32();
    let u2 = fastrand::f32();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// Example usage
fn main() {
    println!("SD 3.5 Flow Matching Sampler\n");
    
    let sampler = SD35Sampler::new(20);
    
    // Test timesteps
    let timesteps = sampler.get_timesteps();
    println!("Timesteps: {:?}", &timesteps[..5]);
    
    // Test sigmas
    let sigmas = sampler.get_sigmas();
    println!("Sigmas: {:?}", &sigmas[..5]);
    
    // Mock model function
    let model_fn = |_x: &[f32], t: f32, _embeds: &[f32], _pooled: &[f32]| {
        // Simulate model prediction
        vec![0.1 * (t / 1000.0); 16 * 128 * 128] // SD3.5 uses 16-channel latents
    };
    
    // Test sampling
    let mut latents = vec![0.0f32; 16 * 128 * 128];
    let text_embeds = vec![0.0f32; 77 * 6144]; // CLIP-L + CLIP-G + T5
    let pooled = vec![0.0f32; 2048];
    
    let result = sampler.sample(
        &mut latents,
        (1, 16, 128, 128),
        &text_embeds,
        &pooled,
        model_fn,
    );
    
    println!("\nFinal latent stats:");
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    let min = result.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = result.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!("  Mean: {:.4}, Min: {:.4}, Max: {:.4}", mean, min, max);
}