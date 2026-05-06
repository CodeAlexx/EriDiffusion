//! Training pipeline traits: noise injection, loss computation.
//! Each model type (DDPM, flow-matching) has its own implementation.

use flame_core::Tensor;
use crate::Result;

/// Pre-packaged inputs after noise injection
pub struct PreparedInputs {
    pub noisy: Tensor,
    pub timestep: Tensor,
    pub context: Vec<Tensor>,
    pub pooled: Option<Tensor>,
}

/// Pre-packaged targets for loss
pub struct Targets {
    pub target: Tensor,
}

/// Model-specific training pipeline.
pub trait TrainingPipeline: Send + Sync {
    /// Add noise to latents, produce (noisy_latent, timestep, context) + target
    fn prepare_inputs(
        &self,
        latents: &Tensor,
        text_embeddings: &[Tensor],
        pooled: Option<&Tensor>,
    ) -> Result<(PreparedInputs, Targets)>;

    /// Compute loss between prediction and target
    fn compute_loss(
        &self,
        pred: &Tensor,
        target: &Tensor,
        timestep_idx: Option<usize>,
    ) -> Result<Tensor>;

    /// Get the noise schedule alpha-bar values (for DDPM)
    fn alphas_cumprod(&self) -> Option<&[f32]> { None }

    /// Get the number of timesteps
    fn num_timesteps(&self) -> usize { 1000 }
}

/// Flow matching training pipeline (Flux, SD3, Klein, Z-Image, etc.)
pub struct FlowMatchingPipeline {
    pub shift: f32,
    pub seed: u64,
}

impl FlowMatchingPipeline {
    /// Sample a logit-normal timestep in (0,1), shifted, clamped to (eps, 1-eps)
    pub fn sample_timestep(seed: &mut u64, shift: f32) -> f32 {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(*seed);
        *seed = seed.wrapping_add(1);

        // Box-Muller for logit-normal
        let u1 = rng.gen::<f32>().max(1e-6);
        let u2 = rng.gen::<f32>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let t_raw = 1.0 / (1.0 + (-z).exp());

        // Resolution shift
        let t = if (shift - 1.0).abs() < 1e-6 {
            t_raw
        } else {
            shift * t_raw / ((shift - 1.0) * t_raw + 1.0)
        };

        t.clamp(1e-4, 1.0 - 1e-4)
    }
}

impl TrainingPipeline for FlowMatchingPipeline {
    fn prepare_inputs(
        &self,
        latents: &Tensor,
        text_embeddings: &[Tensor],
        pooled: Option<&Tensor>,
    ) -> Result<(PreparedInputs, Targets)> {
        let mut seed = self.seed;
        let t = Self::sample_timestep(&mut seed, self.shift);
        let noise = Tensor::randn(latents.shape().clone(), 0.0, 1.0, latents.device().clone())?;

        let scale_t = 1.0 - t;
        let noisy = latents.mul_scalar(scale_t)?.add(&noise.mul_scalar(t)?)?;
        let target = noise.sub(latents)?;

        let timestep = Tensor::from_vec(vec![t], flame_core::Shape::from_dims(&[1]), latents.device().clone())?;

        Ok((
            PreparedInputs { noisy, timestep, context: text_embeddings.to_vec(), pooled: pooled.cloned() },
            Targets { target },
        ))
    }

    fn compute_loss(
        &self,
        pred: &Tensor,
        target: &Tensor,
        _timestep_idx: Option<usize>,
    ) -> Result<Tensor> {
        // MSE in BF16 with FP32 reduction
        let diff = pred.sub(target)?;
        let squared = diff.square()?;
        Ok(squared.mean()?)
    }
}

/// DDPM training pipeline (SD1.x, SD2.x, SDXL)
pub struct DDPMPipeline {
    pub alphas_cumprod: Vec<f32>,
    pub v_prediction: bool,
    pub seed: u64,
}

impl DDPMPipeline {
    pub fn new(num_steps: usize, v_prediction: bool, seed: u64) -> Self {
        // Cosine schedule
        let mut alphas = vec![1.0f32; num_steps];
        for i in 0..num_steps {
            let t = i as f32 / (num_steps as f32 - 1.0);
            alphas[i] = (t * std::f32::consts::PI / 2.0).cos().powi(2);
        }
        Self { alphas_cumprod: alphas, v_prediction, seed }
    }
}

impl TrainingPipeline for DDPMPipeline {
    fn prepare_inputs(
        &self,
        latents: &Tensor,
        _text_embeddings: &[Tensor],
        _pooled: Option<&Tensor>,
    ) -> Result<(PreparedInputs, Targets)> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let ti = rng.gen_range(0..self.alphas_cumprod.len());

        let alpha = self.alphas_cumprod[ti];
        let noise = Tensor::randn(latents.shape().clone(), 0.0, 1.0, latents.device().clone())?;

        // x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * noise
        let sqrt_alpha = alpha.sqrt();
        let sqrt_1m_alpha = (1.0 - alpha).sqrt();
        let noisy = latents.mul_scalar(sqrt_alpha)?.add(&noise.mul_scalar(sqrt_1m_alpha)?)?;

        // Target depends on prediction type
        let target = if self.v_prediction {
            // v = sqrt(alpha) * noise - sqrt(1-alpha) * x_0
            noise.mul_scalar(sqrt_alpha)?.sub(&latents.mul_scalar(sqrt_1m_alpha)?)?
        } else {
            noise
        };

        let timestep = Tensor::from_vec(
            vec![ti as f32],
            flame_core::Shape::from_dims(&[1]),
            latents.device().clone(),
        )?;

        Ok((
            PreparedInputs { noisy, timestep, context: vec![], pooled: None },
            Targets { target },
        ))
    }

    fn compute_loss(&self, pred: &Tensor, target: &Tensor, _ti: Option<usize>) -> Result<Tensor> {
        let diff = pred.sub(target)?;
        Ok(diff.square()?.mean()?)
    }

    fn alphas_cumprod(&self) -> Option<&[f32]> { Some(&self.alphas_cumprod) }
    fn num_timesteps(&self) -> usize { self.alphas_cumprod.len() }
}
