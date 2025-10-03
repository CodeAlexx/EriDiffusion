use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};

// SNR (Signal-to-Noise Ratio) weighting for diffusion training
// Implements Min-SNR weighting from https://arxiv.org/abs/2303.09556

// FLAME uses flame_core::device::Device instead of Device

/// Compute SNR weighting for loss calculation
pub struct SNRWeighting {
    gamma: f32,
    min_snr_gamma: Option<f32>,
}

impl SNRWeighting {
    pub fn new(gamma: f32, min_snr_gamma: Option<f32>) -> Self {
        Self { gamma, min_snr_gamma }
    }

    /// Calculate SNR weight for given timesteps
    /// For DDPM scheduler with linear beta schedule
    pub fn calculate_snr_weight(
        &self,
        timesteps: &Tensor,
        num_train_timesteps: usize,
    ) -> flame_core::Result<Tensor> {
        // Convert timesteps to alphas
        let device = Device::from(timesteps.device().clone());
        let alphas_cumprod = self.get_alphas_cumprod(num_train_timesteps, &device)?;

        // Get alpha values for the given timesteps
        let timesteps_i64 = timesteps.to_dtype(DType::I64)?;
        let alphas = alphas_cumprod.index_select(0, &timesteps_i64)?;

        // Calculate SNR = alpha^2 / (1 - alpha^2)
        let alphas_sq = alphas.square()?;
        let one = Tensor::full(alphas_sq.shape().clone(), 1.0, alphas_sq.device().clone())?;
        let one_minus_alphas_sq = one.sub(&alphas_sq)?;
        let snr = alphas_sq.div(&one_minus_alphas_sq)?;

        // Apply gamma weighting: weight = min(snr, gamma) / snr
        let gamma_tensor = Tensor::full(snr.shape().clone(), self.gamma, snr.device().clone())?;

        let snr_weight = if let Some(min_gamma) = self.min_snr_gamma {
            // Clamp SNR to minimum value before computing weight
            let min_gamma_tensor =
                Tensor::full(snr.shape().clone(), min_gamma, snr.device().clone())?;
            let clamped_snr = snr.maximum(&min_gamma_tensor)?;
            gamma_tensor.minimum(&clamped_snr)?.div(&clamped_snr)?
        } else {
            gamma_tensor.minimum(&snr)?.div(&snr)?
        };

        Ok(snr_weight)
    }

    /// Apply SNR weighting to loss
    pub fn apply_snr_weighting(
        &self,
        loss: &Tensor,
        timesteps: &Tensor,
        num_train_timesteps: usize,
    ) -> flame_core::Result<Tensor> {
        let snr_weight = self.calculate_snr_weight(timesteps, num_train_timesteps)?;

        // Ensure weight has same dtype as loss
        let snr_weight = snr_weight.to_dtype(loss.dtype())?;

        // Apply per-sample weighting
        let weighted_loss = loss.mul(&snr_weight)?;

        Ok(weighted_loss)
    }

    /// Get alphas_cumprod for DDPM scheduler
    fn get_alphas_cumprod(
        &self,
        num_train_timesteps: usize,
        device: &Device,
    ) -> flame_core::Result<Tensor> {
        // Linear beta schedule
        let beta_start = 0.00085f32;
        let beta_end = 0.012f32;

        // Create linspace manually
        let mut beta_values = Vec::with_capacity(num_train_timesteps);
        for i in 0..num_train_timesteps {
            let t = i as f32 / (num_train_timesteps - 1) as f32;
            beta_values.push(beta_start + t * (beta_end - beta_start));
        }
        let betas = Tensor::from_slice(
            &beta_values,
            Shape::from_dims(&[num_train_timesteps]),
            device.cuda_device().clone(),
        )?;
        let one = Tensor::full(betas.shape().clone(), 1.0, betas.device().clone())?;
        let alphas = one.sub(&betas)?;

        // Cumprod of alphas
        let mut alphas_cumprod_vec = Vec::with_capacity(num_train_timesteps);
        // TODO: Implement proper tensor-to-vec conversion
        let alphas_vec: Vec<f32> = beta_values.iter().map(|b| 1.0 - b).collect();

        let mut cumprod = 1.0f32;
        for alpha in alphas_vec {
            cumprod *= alpha;
            alphas_cumprod_vec.push(cumprod);
        }

        Ok(Tensor::from_slice(
            &alphas_cumprod_vec,
            Shape::from_dims(&[num_train_timesteps]),
            device.cuda_device().clone(),
        )?)
    }

    /// V-parameterization for diffusion models
    pub fn compute_v_prediction(
        noise_pred: &Tensor,
        sample: &Tensor,
        noise: &Tensor,
        alphas_cumprod: &Tensor,
        timesteps: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // v = sqrt(alpha) * noise - sqrt(1 - alpha) * sample
        let timesteps_i64 = timesteps.to_dtype(DType::I64)?;
        let alpha_t = alphas_cumprod.index_select(0, &timesteps_i64)?;

        let sqrt_alpha_t = alpha_t.sqrt()?;
        // affine(a, b) = a * x + b, so affine(-1.0, 1.0) = -x + 1 = 1 - x
        let one = Tensor::full(alpha_t.shape().clone(), 1.0, alpha_t.device().clone())?;
        let sqrt_one_minus_alpha_t = one.sub(&alpha_t)?.sqrt()?;

        // Reshape for broadcasting
        let batch_size = sample.shape().dims()[0];
        let sqrt_alpha_t = sqrt_alpha_t.reshape(&[batch_size, 1, 1, 1])?;
        let sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.reshape(&[batch_size, 1, 1, 1])?;

        // v-target
        let v_target = sqrt_alpha_t.mul(noise)?.sub(&sqrt_one_minus_alpha_t.mul(sample)?)?;

        Ok(v_target)
    }
}
