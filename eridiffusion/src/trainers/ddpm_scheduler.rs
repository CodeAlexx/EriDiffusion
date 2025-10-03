use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use rand::Rng;

// DDPM Noise Scheduler for SDXL training
// Implements proper noise scheduling for diffusion training

// FLAME uses flame_core::device::Device instead of Device

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
    ) -> flame_core::Result<Self> {
        // Generate beta schedule
        let betas = match beta_schedule {
            "linear" => Self::linear_beta_schedule(num_timesteps, beta_start, beta_end, device)?,
            "scaled_linear" => {
                Self::scaled_linear_beta_schedule(num_timesteps, beta_start, beta_end, device)?
            }
            "squaredcos_cap_v2" => Self::cosine_beta_schedule(num_timesteps, device)?,
            _ => Self::linear_beta_schedule(num_timesteps, beta_start, beta_end, device)?,
        };

        // Compute alphas
        let one = Tensor::full(betas.shape().clone(), 1.0, betas.device().clone())?;
        let alphas = one.sub(&betas)?;
        let mut alphas_cumprod_vec = vec![1.0f32];
        // TODO: Replace with proper tensor-to-vec conversion
        let alphas_vec: Vec<f32> = vec![0.999; num_timesteps]; // Placeholder

        for &alpha in alphas_vec.iter() {
            alphas_cumprod_vec.push(alphas_cumprod_vec.last().unwrap() * alpha);
        }
        alphas_cumprod_vec.remove(0); // Remove the initial 1.0

        let alphas_cumprod = Tensor::from_slice(
            &alphas_cumprod_vec,
            Shape::from_dims(&[num_timesteps]),
            device.cuda_device().clone(),
        )?;

        // Precompute useful values
        let sqrt_alphas_cumprod = alphas_cumprod.sqrt()?;
        let one =
            Tensor::full(alphas_cumprod.shape().clone(), 1.0, alphas_cumprod.device().clone())?;
        let sqrt_one_minus_alphas_cumprod = one.sub(&alphas_cumprod)?.sqrt()?;
        let log_one_minus_alphas_cumprod = one.sub(&alphas_cumprod)?.log()?;
        let sqrt_recip_alphas_cumprod = one.div(&alphas_cumprod)?.sqrt()?;
        let sqrt_recipm1_alphas_cumprod = one.div(&alphas_cumprod)?.sub(&one)?.sqrt()?;

        // Compute posterior variance
        let alphas_cumprod_prev = Self::append_zero(&alphas_cumprod)?;
        // Take the last num_timesteps elements to match betas size
        let alphas_cumprod_prev_truncated = alphas_cumprod_prev.slice(&[(1, 1 + num_timesteps)])?;
        let one =
            Tensor::full(alphas_cumprod.shape().clone(), 1.0, alphas_cumprod.device().clone())?;
        let posterior_variance = betas
            .mul(&one.sub(&alphas_cumprod_prev_truncated)?)?
            .div(&one.sub(&alphas_cumprod)?)?;

        // Clipped log calculation
        let posterior_log_variance_clipped = posterior_variance
            .maximum(&Tensor::full(
                Shape::from_dims(&[num_timesteps]),
                1e-20f32,
                device.cuda_device().clone(),
            )?)?
            .log()?;

        // Posterior mean coefficients
        let one =
            Tensor::full(alphas_cumprod.shape().clone(), 1.0, alphas_cumprod.device().clone())?;
        let posterior_mean_coef1 =
            betas.mul(&alphas_cumprod_prev_truncated.sqrt()?)?.div(&one.sub(&alphas_cumprod)?)?;
        let posterior_mean_coef2 = one
            .sub(&alphas_cumprod_prev_truncated)?
            .mul(&alphas.sqrt()?)?
            .div(&one.sub(&alphas_cumprod)?)?;

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
    ) -> flame_core::Result<Tensor> {
        // Get batch size from timesteps
        let batch_size = timesteps.shape().dims()[0];

        // Get sqrt_alpha_prod and sqrt_one_minus_alpha_prod for the given timesteps
        // index_select expects timesteps to be i64
        let timesteps_i64 = timesteps.to_dtype(DType::I64)?;
        let sqrt_alpha_prod = self.sqrt_alphas_cumprod.index_select(0, &timesteps_i64)?;
        let sqrt_one_minus_alpha_prod =
            self.sqrt_one_minus_alphas_cumprod.index_select(0, &timesteps_i64)?;

        // Reshape to broadcast correctly over the latent dimensions
        // From [batch_size] to [batch_size, 1, 1, 1] for broadcasting with [batch_size, 4, H, W]
        let sqrt_alpha_prod = sqrt_alpha_prod.reshape(&[batch_size, 1, 1, 1])?;
        let sqrt_one_minus_alpha_prod =
            sqrt_one_minus_alpha_prod.reshape(&[batch_size, 1, 1, 1])?;

        // Add noise: noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        // Convert to same dtype as input samples
        let sqrt_alpha_prod = sqrt_alpha_prod.to_dtype(original_samples.dtype())?;
        let sqrt_one_minus_alpha_prod =
            sqrt_one_minus_alpha_prod.to_dtype(original_samples.dtype())?;

        // Use mul which handles broadcasting automatically in FLAME
        let scaled_original = sqrt_alpha_prod.mul(original_samples)?;
        let scaled_noise = sqrt_one_minus_alpha_prod.mul(noise)?;
        let noisy_samples = scaled_original.add(&scaled_noise)?;

        Ok(noisy_samples)
    }

    /// Get SNR (Signal-to-Noise Ratio) for loss weighting
    pub fn get_snr(&self, timesteps: &Tensor) -> flame_core::Result<Tensor> {
        let timesteps_i64 = timesteps.to_dtype(DType::I64)?;
        let alphas_cumprod = self.alphas_cumprod.index_select(0, &timesteps_i64)?;
        let one =
            Tensor::full(alphas_cumprod.shape().clone(), 1.0, alphas_cumprod.device().clone())?;
        let snr = alphas_cumprod.div(&one.sub(&alphas_cumprod)?)?;
        Ok(snr)
    }

    /// Compute v-prediction target
    pub fn get_velocity(
        &self,
        sample: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> flame_core::Result<Tensor> {
        let batch_size = timesteps.shape().dims()[0];

        let timesteps_i64 = timesteps.to_dtype(DType::I64)?;
        let sqrt_alpha_prod = self.sqrt_alphas_cumprod.index_select(0, &timesteps_i64)?;
        let sqrt_one_minus_alpha_prod =
            self.sqrt_one_minus_alphas_cumprod.index_select(0, &timesteps_i64)?;

        // Reshape for broadcasting
        let sqrt_alpha_prod = sqrt_alpha_prod.reshape(&[batch_size, 1, 1, 1])?;
        let sqrt_one_minus_alpha_prod =
            sqrt_one_minus_alpha_prod.reshape(&[batch_size, 1, 1, 1])?;

        // Convert to same dtype as inputs
        let sqrt_alpha_prod = sqrt_alpha_prod.to_dtype(sample.dtype())?;
        let sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to_dtype(sample.dtype())?;

        let velocity = sqrt_alpha_prod.mul(noise)?.sub(&sqrt_one_minus_alpha_prod.mul(sample)?)?;
        Ok(velocity)
    }

    /// Sample random timesteps
    pub fn sample_timesteps(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> flame_core::Result<Tensor> {
        let mut rng = rand::thread_rng();
        let timesteps: Vec<i64> =
            (0..batch_size).map(|_| rng.gen_range(0..self.num_timesteps) as i64).collect();

        // Convert i64 to f32 for Tensor::from_slice
        let timesteps_f32: Vec<f32> = timesteps.iter().map(|&t| t as f32).collect();
        Ok(Tensor::from_slice(
            &timesteps_f32,
            Shape::from_dims(&[batch_size]),
            device.cuda_device().clone(),
        )?
        .to_dtype(DType::I64)?)
    }

    // Beta schedule functions
    fn linear_beta_schedule(
        num_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        device: &Device,
    ) -> flame_core::Result<Tensor> {
        let betas: Vec<f32> = (0..num_timesteps)
            .map(|i| {
                beta_start + (beta_end - beta_start) * (i as f32) / (num_timesteps as f32 - 1.0)
            })
            .collect();

        Ok(Tensor::from_slice(
            &betas,
            Shape::from_dims(&[num_timesteps]),
            device.cuda_device().clone(),
        )?)
    }

    fn scaled_linear_beta_schedule(
        num_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        device: &Device,
    ) -> flame_core::Result<Tensor> {
        let start = beta_start.sqrt();
        let end = beta_end.sqrt();
        let betas: Vec<f32> = (0..num_timesteps)
            .map(|i| {
                let t = start + (end - start) * (i as f32) / (num_timesteps as f32 - 1.0);
                t * t
            })
            .collect();

        Ok(Tensor::from_slice(
            &betas,
            Shape::from_dims(&[num_timesteps]),
            device.cuda_device().clone(),
        )?)
    }

    fn cosine_beta_schedule(num_timesteps: usize, device: &Device) -> flame_core::Result<Tensor> {
        let s = 0.008;
        let steps: Vec<f32> =
            (0..=num_timesteps).map(|i| i as f32 / num_timesteps as f32).collect();

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

        Ok(Tensor::from_slice(
            &betas,
            Shape::from_dims(&[num_timesteps]),
            device.cuda_device().clone(),
        )?)
    }

    fn append_zero(tensor: &Tensor) -> flame_core::Result<Tensor> {
        let device = Device::from(tensor.device().clone());
        let zero = Tensor::zeros_dtype(
            Shape::from_dims(&[1]),
            tensor.dtype(),
            device.cuda_device().clone(),
        )?;
        Ok(Tensor::cat(&[&zero, tensor], 0)?)
    }

    /// Get the number of training timesteps
    pub fn num_train_timesteps(&self) -> usize {
        self.num_timesteps
    }

    /// Get alphas_cumprod tensor
    pub fn alphas_cumprod(&self) -> flame_core::Result<Tensor> {
        Ok(self.alphas_cumprod.clone())
    }

    /// Min-SNR loss weighting
    pub fn compute_snr_loss_weights(snr: &Tensor, gamma: f32) -> flame_core::Result<Tensor> {
        // Min-SNR-gamma weighting: min(snr, gamma) / snr
        let gamma_tensor = Tensor::full(snr.shape().clone(), gamma, snr.device().clone())?;
        let clipped_snr = snr.minimum(&gamma_tensor)?;
        let weights = clipped_snr.div(snr)?;
        Ok(weights)
    }
}
