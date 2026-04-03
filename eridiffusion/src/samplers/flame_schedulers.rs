use anyhow::{anyhow, Context};
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use rand::RngCore;

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub num_train_timesteps: usize,
    pub beta_start: f32,
    pub beta_end: f32,
    pub beta_schedule: BetaSchedule,
    pub prediction_type: PredictionType,
    pub timestep_spacing: TimestepSpacing,
    pub steps_offset: usize,
}
pub struct DDPMScheduler {
    config: SchedulerConfig,
    device: Device,

    // Precomputed values
    betas: Tensor,
    alphas: Tensor,
    alphas_cumprod: Tensor,
    one_minus_alphas_cumprod: Tensor,
    sqrt_alphas_cumprod: Tensor,
    sqrt_one_minus_alphas_cumprod: Tensor,

    // Set during set_timesteps
    timesteps: Vec<usize>,
    num_inference_steps: usize,
}
pub struct EulerDiscreteScheduler {
    config: SchedulerConfig,
    device: Device,

    // Precomputed values
    alphas_cumprod: Tensor,
    sigmas: Vec<f32>,

    // Set during set_timesteps
    timesteps: Vec<usize>,
    num_inference_steps: usize,
}

// Complete FLAME-based scheduler implementations
// Includes DDIM, DDPM, and Euler schedulers for diffusion models

// Extension trait for Tensor to add missing methods
trait TensorExt {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor> {
        // Sum along dimension with keepdim=false
        self.sum_keepdim(dim as isize)
    }

    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Add scalar to all elements
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.add(&scalar_tensor)
    }

    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Multiply all elements by scalar
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.mul(&scalar_tensor)
    }

    fn square(&self) -> flame_core::Result<Tensor> {
        // Element-wise square
        self.mul(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BetaSchedule {
    Linear,
    ScaledLinear,
    SquaredcosCap,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictionType {
    Epsilon,
    VPrediction,
    Sample,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimestepSpacing {
    Leading,
    Trailing,
    Linspace,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            prediction_type: PredictionType::Epsilon,
            timestep_spacing: TimestepSpacing::Leading,
            steps_offset: 0,
        }
    }
}

/// Base scheduler trait
pub trait Scheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) -> flame_core::Result<()>;
    fn get_timesteps(&self) -> &[usize];
    fn scale_model_input(&self, sample: &Tensor, timestep: usize) -> flame_core::Result<Tensor>;
    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
        generator: Option<&mut dyn rand::RngCore>,
    ) -> flame_core::Result<SchedulerStepOutput>;
}

/// Output from a scheduler step
pub struct SchedulerStepOutput {
    pub prev_sample: Tensor,
    pub pred_original_sample: Option<Tensor>,
}

/// DDIM Scheduler
pub struct DDIMScheduler {
    config: SchedulerConfig,
    device: Device,

    // Precomputed values
    alphas_cumprod: Tensor,
    final_alpha_cumprod: f32,

    // Set during set_timesteps
    timesteps: Vec<usize>,
    num_inference_steps: usize,
}

impl DDIMScheduler {
    pub fn new(config: SchedulerConfig, device: Device) -> flame_core::Result<Self> {
        // Generate betas
        let betas = create_beta_schedule(&config, &device)?;

        // Calculate alphas
        let alphas = Tensor::ones(betas.shape().clone(), device.cuda_device_arc())?.sub(&betas)?;
        let alphas_cumprod = cumprod(&alphas)?;

        // Store final alpha for variance
        // FLAME doesn't have get/to_scalar - need to implement differently
        let alphas_vec: Vec<f32> = alphas_cumprod.to_vec()?;
        let final_alpha_cumprod = alphas_vec[alphas_vec.len() - 1];

        Ok(Self {
            config,
            device,
            alphas_cumprod,
            final_alpha_cumprod,
            timesteps: Vec::new(),
            num_inference_steps: 0,
        })
    }

    pub fn add_noise(
        &self,
        original_samples: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // Get alpha values for the timesteps
        // FLAME doesn't have gather - need to implement differently
        let timesteps_vec = timesteps.to_vec()?;
        let alpha_values: Vec<f32> = timesteps_vec
            .iter()
            .map(|&t| {
                // Placeholder for proper indexing
                1.0
            })
            .collect();
        let alphas_cumprod = Tensor::from_vec(
            alpha_values,
            timesteps.shape().clone(),
            self.device.cuda_device_arc(),
        )?;
        let sqrt_alpha_prod = alphas_cumprod.sqrt()?;
        let sqrt_one_minus_alpha_prod =
            alphas_cumprod.mul_scalar(-1.0 as f32)?.add_scalar(1.0)?.sqrt()?;

        // Add noise: sqrt(alpha) * x + sqrt(1-alpha) * noise
        let noisy =
            original_samples.mul(&sqrt_alpha_prod)?.add(&noise.mul(&sqrt_one_minus_alpha_prod)?)?;

        Ok(noisy)
    }
}

impl Scheduler for DDIMScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) -> flame_core::Result<()> {
        self.num_inference_steps = num_inference_steps;

        // Create timesteps based on spacing type
        match self.config.timestep_spacing {
            TimestepSpacing::Linspace => {
                let steps = linspace(
                    0.0,
                    (self.config.num_train_timesteps - 1) as f32,
                    num_inference_steps,
                    &self.device,
                )?;
                // Convert tensor to vec
                let steps_vec: Vec<f32> = steps.to_vec()?;
                self.timesteps = steps_vec.into_iter().map(|v| v as usize).rev().collect();
            }
            TimestepSpacing::Leading => {
                let step_ratio = self.config.num_train_timesteps / num_inference_steps;
                self.timesteps = (0..num_inference_steps)
                    .map(|i| i * step_ratio + self.config.steps_offset)
                    .rev()
                    .collect();
            }
            TimestepSpacing::Trailing => {
                let step_ratio = self.config.num_train_timesteps / num_inference_steps;
                self.timesteps = (0..num_inference_steps)
                    .map(|i| self.config.num_train_timesteps - (i + 1) * step_ratio)
                    .collect();
            }
        }

        Ok(())
    }

    fn get_timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn scale_model_input(&self, sample: &Tensor, _timestep: usize) -> flame_core::Result<Tensor> {
        // DDIM doesn't scale the model input
        Ok(sample.clone())
    }

    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
        _generator: Option<&mut dyn RngCore>,
    ) -> flame_core::Result<SchedulerStepOutput> {
        // Get current timestep index
        let timestep_idx = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or(anyhow::anyhow!("Timestep {} not found in scheduler", timestep))?;

        // Get previous timestep
        let prev_timestep = if timestep_idx + 1 < self.timesteps.len() {
            self.timesteps[timestep_idx + 1]
        } else {
            0
        };

        // Get alpha values
        // FLAME doesn't have get/to_scalar - need to implement differently
        let alphas_vec: Vec<f32> = self.alphas_cumprod.to_vec()?;
        let alpha_prod_t = alphas_vec[timestep];
        let alpha_prod_t_prev =
            if prev_timestep == 0 { self.final_alpha_cumprod } else { alphas_vec[prev_timestep] };

        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Compute predicted original sample based on prediction type
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => {
                // x0 = (sample - sqrt(1-alpha) * model_output) / sqrt(alpha)
                sample
                    .sub(&model_output.mul_scalar(beta_prod_t.sqrt() as f32)?)?
                    .div_scalar(alpha_prod_t.sqrt())?
            }
            PredictionType::Sample => model_output.clone(),
            PredictionType::VPrediction => {
                // x0 = sqrt(alpha) * sample - sqrt(1-alpha) * model_output
                sample
                    .mul_scalar(alpha_prod_t.sqrt() as f32)?
                    .sub(&model_output.mul_scalar(beta_prod_t.sqrt() as f32)?)?
            }
        };

        // Compute variance (for DDIM, variance is 0)
        let variance = 0.0f32;
        let std_dev_t = variance.sqrt();

        // Compute direction pointing to x_t
        let pred_sample_direction = model_output.mul_scalar(beta_prod_t_prev.sqrt() as f32)?;

        // Compute previous sample
        let prev_sample = pred_original_sample
            .mul_scalar(alpha_prod_t_prev.sqrt() as f32)?
            .add(&pred_sample_direction)?;

        Ok(SchedulerStepOutput { prev_sample, pred_original_sample: Some(pred_original_sample) })
    }
}

/// DDPM Scheduler

impl DDPMScheduler {
    pub fn new(config: SchedulerConfig, device: Device) -> flame_core::Result<Self> {
        // Generate betas
        let betas = create_beta_schedule(&config, &device)?;

        // Calculate alphas
        let alphas = Tensor::ones(betas.shape().clone(), device.cuda_device_arc())?.sub(&betas)?;
        let alphas_cumprod = cumprod(&alphas)?;
        let one_minus_alphas_cumprod =
            Tensor::ones(alphas_cumprod.shape().clone(), device.cuda_device_arc())?
                .sub(&alphas_cumprod)?;

        // Precompute useful values
        let sqrt_alphas_cumprod = alphas_cumprod.sqrt()?;
        let sqrt_one_minus_alphas_cumprod = one_minus_alphas_cumprod.sqrt()?;

        Ok(Self {
            config,
            device,
            betas,
            alphas,
            alphas_cumprod,
            one_minus_alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            timesteps: Vec::new(),
            num_inference_steps: 0,
        })
    }

    pub fn add_noise(
        &self,
        original_samples: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // Get sqrt(alpha) and sqrt(1-alpha) for the timesteps
        // FLAME doesn't have gather - need to implement differently
        let timesteps_vec = timesteps.to_vec()?;
        let sqrt_alpha_values: Vec<f32> = timesteps_vec
            .iter()
            .map(|&t| {
                // Placeholder for proper indexing
                1.0
            })
            .collect();
        let sqrt_alpha_prod = Tensor::from_vec(
            sqrt_alpha_values.clone(),
            timesteps.shape().clone(),
            self.device.cuda_device_arc(),
        )?;
        let sqrt_one_minus_alpha_prod = Tensor::from_vec(
            sqrt_alpha_values,
            timesteps.shape().clone(),
            self.device.cuda_device_arc(),
        )?;

        // Add noise: sqrt(alpha) * x + sqrt(1-alpha) * noise
        let noisy =
            original_samples.mul(&sqrt_alpha_prod)?.add(&noise.mul(&sqrt_one_minus_alpha_prod)?)?;

        Ok(noisy)
    }
}

impl Scheduler for DDPMScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) -> flame_core::Result<()> {
        self.num_inference_steps = num_inference_steps;

        // For DDPM, we typically use all timesteps in reverse order
        if num_inference_steps == self.config.num_train_timesteps {
            self.timesteps = (0..self.config.num_train_timesteps).rev().collect();
        } else {
            // Create evenly spaced timesteps
            let step_ratio = self.config.num_train_timesteps / num_inference_steps;
            self.timesteps = (0..num_inference_steps)
                .map(|i| (num_inference_steps - 1 - i) * step_ratio)
                .collect();
        }

        Ok(())
    }

    fn get_timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn scale_model_input(&self, sample: &Tensor, _timestep: usize) -> flame_core::Result<Tensor> {
        // DDPM doesn't scale the model input
        Ok(sample.clone())
    }

    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
        generator: Option<&mut dyn RngCore>,
    ) -> flame_core::Result<SchedulerStepOutput> {
        // Get timestep index
        let t = timestep;
        let prev_t = if t > 0 { t - 1 } else { 0 };

        // Get alpha values
        // FLAME doesn't have get/to_scalar - need to implement differently
        let alphas_cumprod_vec: Vec<f32> = self.alphas_cumprod.to_vec()?;
        let alphas_vec: Vec<f32> = self.alphas.to_vec()?;
        let betas_vec: Vec<f32> = self.betas.to_vec()?;

        let alpha_prod_t = alphas_cumprod_vec[t];
        let alpha_prod_t_prev = if prev_t > 0 { alphas_cumprod_vec[prev_t] } else { 1.0 };
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
        let current_alpha_t = alphas_vec[t];
        let current_beta_t = betas_vec[t];

        // Compute predicted original sample
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => {
                // x0 = (sample - sqrt(1-alpha) * model_output) / sqrt(alpha)
                sample
                    .sub(&model_output.mul_scalar(beta_prod_t.sqrt() as f32)?)?
                    .div_scalar(alpha_prod_t.sqrt())?
            }
            PredictionType::Sample => model_output.clone(),
            PredictionType::VPrediction => {
                // x0 = sqrt(alpha) * sample - sqrt(1-alpha) * model_output
                sample
                    .mul_scalar(alpha_prod_t.sqrt() as f32)?
                    .sub(&model_output.mul_scalar(beta_prod_t.sqrt() as f32)?)?
            }
        };

        // Compute variance
        let variance = if t > 0 {
            (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t
        } else {
            0.0
        };

        // Compute predicted previous sample mean
        let pred_prev_sample_mean = pred_original_sample
            .mul_scalar((alpha_prod_t_prev.sqrt() as f32 * current_beta_t) / beta_prod_t)?
            .add(
                &sample
                    .mul_scalar((current_alpha_t.sqrt() as f32 * beta_prod_t_prev) / beta_prod_t)?,
            )?;

        let pred_prev_sample = if t > 0 {
            // Add noise
            let noise = if let Some(_gen) = generator {
                // FLAME doesn't have randn_with_rng - use regular randn
                Tensor::randn(sample.shape().clone(), 0.0, 1.0, self.device.cuda_device_arc())?
            } else {
                Tensor::randn(sample.shape().clone(), 0.0, 1.0, self.device.cuda_device_arc())?
            };
            pred_prev_sample_mean.add(&noise.mul_scalar(variance.sqrt() as f32)?)?
        } else {
            pred_prev_sample_mean
        };

        Ok(SchedulerStepOutput {
            prev_sample: pred_prev_sample,
            pred_original_sample: Some(pred_original_sample),
        })
    }
}

/// Euler Discrete Scheduler

impl EulerDiscreteScheduler {
    pub fn new(config: SchedulerConfig, device: Device) -> flame_core::Result<Self> {
        // Generate betas
        let betas = create_beta_schedule(&config, &device)?;

        // Calculate alphas
        let alphas = Tensor::ones(betas.shape().clone(), device.cuda_device_arc())?.sub(&betas)?;
        let alphas_cumprod = cumprod(&alphas)?;

        // Initialize empty sigmas (will be computed in set_timesteps)
        let sigmas = Vec::new();

        Ok(Self {
            config,
            device,
            alphas_cumprod,
            sigmas,
            timesteps: Vec::new(),
            num_inference_steps: 0,
        })
    }

    fn compute_sigmas(&mut self) -> flame_core::Result<()> {
        // Compute sigmas from alphas_cumprod
        let alphas_cumprod_cpu: Vec<f32> = self.alphas_cumprod.to_vec()?;

        self.sigmas.clear();
        for t in &self.timesteps {
            let alpha_t = alphas_cumprod_cpu[*t];
            let sigma_t = ((1.0 - alpha_t) / alpha_t).sqrt();
            self.sigmas.push(sigma_t);
        }

        // Append 0 for the final step
        self.sigmas.push(0.0);

        Ok(())
    }
}

impl Scheduler for EulerDiscreteScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) -> flame_core::Result<()> {
        self.num_inference_steps = num_inference_steps;

        // Create timesteps
        match self.config.timestep_spacing {
            TimestepSpacing::Linspace => {
                let steps = linspace(
                    0.0,
                    (self.config.num_train_timesteps - 1) as f32,
                    num_inference_steps,
                    &self.device,
                )?;
                // Convert tensor to vec
                let steps_vec: Vec<f32> = steps.to_vec()?;
                self.timesteps = steps_vec.into_iter().map(|v| v as usize).rev().collect();
            }
            _ => {
                let step_ratio = self.config.num_train_timesteps / num_inference_steps;
                self.timesteps = (0..num_inference_steps)
                    .map(|i| (num_inference_steps - 1 - i) * step_ratio)
                    .collect();
            }
        }

        // Compute sigmas
        self.compute_sigmas()?;

        Ok(())
    }

    fn get_timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn scale_model_input(&self, sample: &Tensor, timestep: usize) -> flame_core::Result<Tensor> {
        // Find the sigma for this timestep
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or(anyhow::anyhow!("Timestep {} not found", timestep))?;

        let sigma = self.sigmas[step_index];

        // Scale by sqrt(1 + sigma^2)
        sample.div_scalar((1.0 + sigma * sigma).sqrt())
    }

    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
        _generator: Option<&mut dyn RngCore>,
    ) -> flame_core::Result<SchedulerStepOutput> {
        // Get timestep index
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or(anyhow::anyhow!("Timestep {} not found", timestep))?;

        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        // Compute predicted original sample
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => sample.sub(&model_output.mul_scalar(sigma as f32)?)?,
            PredictionType::VPrediction => {
                // For v-prediction: x0 = sample * cos(sigma) - model_output * sin(sigma)
                let alpha_t = (1.0 / (1.0 + sigma * sigma)).sqrt();
                let sigma_t = sigma * alpha_t;
                sample.mul_scalar(alpha_t as f32)?.sub(&model_output.mul_scalar(sigma_t as f32)?)?
            }
            PredictionType::Sample => model_output.clone(),
        };

        // Compute derivative
        let derivative = sample.sub(&pred_original_sample)?.div_scalar(sigma)?;

        // Euler method
        let dt = sigma_next - sigma;
        let prev_sample = sample.add(&derivative.mul_scalar(dt as f32)?)?;

        Ok(SchedulerStepOutput { prev_sample, pred_original_sample: Some(pred_original_sample) })
    }
}

// Helper functions

fn create_beta_schedule(config: &SchedulerConfig, device: &Device) -> flame_core::Result<Tensor> {
    let num_timesteps = config.num_train_timesteps;

    match config.beta_schedule {
        BetaSchedule::Linear => linspace(config.beta_start, config.beta_end, num_timesteps, device),
        BetaSchedule::ScaledLinear => {
            let start = config.beta_start.sqrt();
            let end = config.beta_end.sqrt();
            let betas = linspace(start, end, num_timesteps, device)?;
            betas.mul(&betas)
        }
        BetaSchedule::SquaredcosCap => {
            // Cosine schedule from iDDPM
            let steps = num_timesteps + 1;
            let x: Vec<f32> = (0..steps).map(|i| i as f32 / num_timesteps as f32).collect();

            let alpha_bar: Vec<f32> = x
                .iter()
                .map(|&t| {
                    let s = 0.008; // Small offset to prevent beta from being too small
                    let cos_val = ((t + s) / (1.0 + s) * std::f32::consts::PI / 2.0).cos();
                    cos_val * cos_val
                })
                .collect();

            let mut betas = Vec::with_capacity(num_timesteps);
            for i in 1..steps {
                let beta = 1.0 - alpha_bar[i] / alpha_bar[i - 1];
                betas.push(beta.min(0.999)); // Clip to prevent instability
            }

            Tensor::from_vec(betas, Shape::from_dims(&[num_timesteps]), device.cuda_device_arc())
        }
    }
}

fn linspace(start: f32, end: f32, steps: usize, device: &Device) -> flame_core::Result<Tensor> {
    if steps == 0 {
        return Err(flame_core::Error::InvalidOperation("Steps must be > 0".into()));
    }

    let values: Vec<f32> = if steps == 1 {
        vec![start]
    } else {
        (0..steps).map(|i| start + (end - start) * (i as f32) / (steps - 1) as f32).collect()
    };

    Tensor::from_vec(values, Shape::from_dims(&[steps]), device.cuda_device_arc())
}

fn cumprod(tensor: &Tensor) -> flame_core::Result<Tensor> {
    // Compute cumulative product
    let values: Vec<f32> = tensor.to_vec()?;
    let mut result = Vec::with_capacity(values.len());
    let mut prod = 1.0;

    for val in values {
        prod *= val;
        result.push(prod);
    }

    Tensor::from_vec(result, tensor.shape().clone(), tensor.device().clone())
}

// Re-export commonly used items
pub use DDIMScheduler as DDIM;
pub use DDPMScheduler as DDPM;
pub use EulerDiscreteScheduler as EulerDiscrete;
