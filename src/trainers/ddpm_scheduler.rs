//! DDPM Noise Scheduler for SDXL training
//! Implements proper noise scheduling for diffusion training

use anyhow::Result;
use candle_core::{Device, DType, Tensor};

pub struct DDPMScheduler {
    num_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: String,
    
    // Precomputed values
    betas: Tensor,
    alphas: Tensor,
    alphas_cumprod: Tensor,
    sqrt_alphas_cumprod: Tensor,
    sqrt_one_minus_alphas_cumprod: Tensor,
    log_one_minus_alphas_cumprod: Tensor,
    sqrt_recip_alphas_cumprod: Tensor,
    sqrt_recipm1_alphas_cumprod: Tensor,
    posterior_variance: Tensor,
    posterior_log_variance_clipped: Tensor,
    posterior_mean_coef1: Tensor,
    posterior_mean_coef2: Tensor,
}

impl DDPMScheduler {
    pub fn new(
        num_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        device: &Device,
    ) -> Result<Self> {
        // Generate beta schedule
        let betas = match beta_schedule {
            "linear" => Self::linear_beta_schedule(num_timesteps, beta_start, beta_end, device)?,
            "scaled_linear" => Self::scaled_linear_beta_schedule(num_timesteps, beta_start, beta_end, device)?,
            "squaredcos_cap_v2" => Self::cosine_beta_schedule(num_timesteps, device)?,
            _ => Self::linear_beta_schedule(num_timesteps, beta_start, beta_end, device)?,
        };
        
        // Compute alphas
        let alphas = (1.0 - &betas)?;
        let mut alphas_cumprod_vec = vec![1.0f32];
        let alphas_vec: Vec<f32> = alphas.to_vec1()?;
        
        for &alpha in alphas_vec.iter() {
            alphas_cumprod_vec.push(alphas_cumprod_vec.last().unwrap() * alpha);
        }
        alphas_cumprod_vec.remove(0); // Remove the initial 1.0
        
        let alphas_cumprod = Tensor::from_vec(alphas_cumprod_vec.clone(), &[num_timesteps], device)?;
        
        // Precompute useful values
        let sqrt_alphas_cumprod = alphas_cumprod.sqrt()?;
        let sqrt_one_minus_alphas_cumprod = (1.0 - &alphas_cumprod)?.sqrt()?;
        let log_one_minus_alphas_cumprod = (1.0 - &alphas_cumprod)?.log()?;
        let sqrt_recip_alphas_cumprod = (1.0 / &alphas_cumprod)?.sqrt()?;
        let sqrt_recipm1_alphas_cumprod = ((1.0 / &alphas_cumprod)? - 1.0)?.sqrt()?;
        
        // Compute posterior variance
        let alphas_cumprod_prev = Self::append_zero(&alphas_cumprod)?;
        // Take the last num_timesteps elements to match betas size
        let alphas_cumprod_prev_truncated = alphas_cumprod_prev.narrow(0, 1, num_timesteps)?;
        let posterior_variance = (&betas * (1.0 - &alphas_cumprod_prev_truncated)? / (1.0 - &alphas_cumprod)?)?;
        
        // Clipped log calculation
        let posterior_log_variance_clipped = posterior_variance.maximum(&Tensor::full(1e-20f32, &[num_timesteps], device)?)?
            .log()?;
        
        // Posterior mean coefficients
        let posterior_mean_coef1 = (&betas * alphas_cumprod_prev_truncated.sqrt()? / (1.0 - &alphas_cumprod)?)?;
        let posterior_mean_coef2 = ((1.0 - &alphas_cumprod_prev_truncated)? * alphas.sqrt()? / (1.0 - &alphas_cumprod)?)?;
        
        Ok(Self {
            num_timesteps,
            beta_start,
            beta_end,
            beta_schedule: beta_schedule.to_string(),
            betas,
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            log_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_mean_coef1,
            posterior_mean_coef2,
        })
    }
    
    /// Add noise to samples
    pub fn add_noise(
        &self,
        original_samples: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        // Get batch size from timesteps
        let batch_size = timesteps.dims()[0];
        
        // Get sqrt_alpha_prod and sqrt_one_minus_alpha_prod for the given timesteps
        // index_select expects timesteps to be i64
        let timesteps_i64 = timesteps.to_dtype(DType::I64)?;
        let sqrt_alpha_prod = self.sqrt_alphas_cumprod.index_select(&timesteps_i64, 0)?;
        let sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.index_select(&timesteps_i64, 0)?;
        
        // Reshape to broadcast correctly over the latent dimensions
        // From [batch_size] to [batch_size, 1, 1, 1] for broadcasting with [batch_size, 4, H, W]
        let sqrt_alpha_prod = sqrt_alpha_prod.reshape(&[batch_size, 1, 1, 1])?;
        let sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.reshape(&[batch_size, 1, 1, 1])?;
        
        // Add noise: noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        // Convert to same dtype as input samples
        let sqrt_alpha_prod = sqrt_alpha_prod.to_dtype(original_samples.dtype())?;
        let sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to_dtype(original_samples.dtype())?;
        
        // Use broadcast_mul to ensure proper broadcasting
        let scaled_original = sqrt_alpha_prod.broadcast_mul(original_samples)?;
        let scaled_noise = sqrt_one_minus_alpha_prod.broadcast_mul(noise)?;
        let noisy_samples = (scaled_original + scaled_noise)?;
        
        Ok(noisy_samples)
    }
    
    /// Get SNR (Signal-to-Noise Ratio) for loss weighting
    pub fn get_snr(&self, timesteps: &Tensor) -> Result<Tensor> {
        let timesteps_i64 = timesteps.to_dtype(DType::I64)?;
        let alphas_cumprod = self.alphas_cumprod.index_select(&timesteps_i64, 0)?;
        let snr = (&alphas_cumprod / (1.0 - &alphas_cumprod)?)?;
        Ok(snr)
    }
    
    /// Compute v-prediction target
    pub fn get_velocity(&self, sample: &Tensor, noise: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        let batch_size = timesteps.dims()[0];
        
        let timesteps_i64 = timesteps.to_dtype(DType::I64)?;
        let sqrt_alpha_prod = self.sqrt_alphas_cumprod.index_select(&timesteps_i64, 0)?;
        let sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.index_select(&timesteps_i64, 0)?;
        
        // Reshape for broadcasting
        let sqrt_alpha_prod = sqrt_alpha_prod.reshape(&[batch_size, 1, 1, 1])?;
        let sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.reshape(&[batch_size, 1, 1, 1])?;
        
        // Convert to same dtype as inputs
        let sqrt_alpha_prod = sqrt_alpha_prod.to_dtype(sample.dtype())?;
        let sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to_dtype(sample.dtype())?;
        
        let velocity = ((sqrt_alpha_prod.broadcast_mul(noise))? - (sqrt_one_minus_alpha_prod.broadcast_mul(sample))?)?;
        Ok(velocity)
    }
    
    /// Sample random timesteps
    pub fn sample_timesteps(&self, batch_size: usize, device: &Device) -> Result<Tensor> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let timesteps: Vec<i64> = (0..batch_size)
            .map(|_| rng.gen_range(0..self.num_timesteps) as i64)
            .collect();
        
        Ok(Tensor::from_vec(timesteps, &[batch_size], device)?)
    }
    
    // Beta schedule functions
    fn linear_beta_schedule(num_timesteps: usize, beta_start: f32, beta_end: f32, device: &Device) -> Result<Tensor> {
        let betas: Vec<f32> = (0..num_timesteps)
            .map(|i| beta_start + (beta_end - beta_start) * (i as f32) / (num_timesteps as f32 - 1.0))
            .collect();
        
        Ok(Tensor::from_vec(betas, &[num_timesteps], device)?)
    }
    
    fn scaled_linear_beta_schedule(num_timesteps: usize, beta_start: f32, beta_end: f32, device: &Device) -> Result<Tensor> {
        let start = beta_start.sqrt();
        let end = beta_end.sqrt();
        let betas: Vec<f32> = (0..num_timesteps)
            .map(|i| {
                let t = start + (end - start) * (i as f32) / (num_timesteps as f32 - 1.0);
                t * t
            })
            .collect();
        
        Ok(Tensor::from_vec(betas, &[num_timesteps], device)?)
    }
    
    fn cosine_beta_schedule(num_timesteps: usize, device: &Device) -> Result<Tensor> {
        let s = 0.008;
        let steps: Vec<f32> = (0..=num_timesteps).map(|i| i as f32 / num_timesteps as f32).collect();
        
        let mut alphas_cumprod = Vec::new();
        for &t in steps.iter() {
            let alpha = ((t + s) / (1.0 + s) * std::f32::consts::PI / 2.0).cos().powi(2);
            alphas_cumprod.push(alpha);
        }
        
        // Normalize
        let alpha_0 = alphas_cumprod[0];
        for alpha in alphas_cumprod.iter_mut() {
            *alpha /= alpha_0;
        }
        
        // Compute betas
        let mut betas = Vec::new();
        for i in 1..alphas_cumprod.len() {
            let beta = 1.0 - (alphas_cumprod[i] / alphas_cumprod[i - 1]);
            betas.push(beta.min(0.999));
        }
        
        Ok(Tensor::from_vec(betas, &[num_timesteps], device)?)
    }
    
    fn append_zero(tensor: &Tensor) -> Result<Tensor> {
        let device = tensor.device();
        let zero = Tensor::zeros(&[1], DType::F32, device)?;
        Ok(Tensor::cat(&[&zero, tensor], 0)?)
    }
    
    /// Get the number of training timesteps
    pub fn num_train_timesteps(&self) -> usize {
        self.num_timesteps
    }
    
    /// Get alphas_cumprod tensor
    pub fn alphas_cumprod(&self) -> Result<Tensor> {
        Ok(self.alphas_cumprod.clone())
    }
}

/// Min-SNR loss weighting
pub fn compute_snr_loss_weights(snr: &Tensor, gamma: f32) -> Result<Tensor> {
    // Min-SNR-gamma weighting: min(snr, gamma) / snr
    let clipped_snr = snr.minimum(&Tensor::new(gamma, snr.device())?)?;
    let weights = (&clipped_snr / snr)?;
    Ok(weights)
}