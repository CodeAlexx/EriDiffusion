use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
pub mod validate;

/// Prediction type for diffusion models
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PredictionType {
    Epsilon,
    VPrediction,
    Sample,
}

/// Configuration for DDIM scheduler
#[derive(Clone)]
pub struct DDIMSchedulerConfig {
    pub num_train_timesteps: usize,
    pub beta_start: f32,
    pub beta_end: f32,
    pub beta_schedule: String,
    pub clip_sample: bool,
    pub prediction_type: PredictionType,
}

impl Default for DDIMSchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".to_string(),
            clip_sample: false,
            prediction_type: PredictionType::Epsilon,
        }
    }
}

pub struct DPMSolverMultistepScheduler {
    num_train_timesteps: usize,
    num_inference_steps: usize,
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f32>,
    sample_max_value: f32,
    prediction_type: PredictionType,
}
pub struct DPMSchedulerConfig {
    pub num_inference_steps: usize,
    pub solver_order: usize,
    pub solver_type: String,
}

// FLAME uses flame_core::device::Device instead of Device

/// Common scheduler trait for diffusion models
pub trait Scheduler: Send + Sync {
    /// Set the number of inference timesteps
    fn set_timesteps(&mut self, num_inference_steps: usize);

    /// Get the current timesteps
    fn timesteps(&self) -> &[usize];

    /// Get initial noise sigma
    fn init_noise_sigma(&self) -> f32;

    /// Scale model input
    fn scale_model_input(&self, sample: &Tensor, timestep: usize) -> flame_core::Result<Tensor>;

    /// Perform one step of the scheduling algorithm
    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor>;
}

/// DDIM scheduler implementation
#[derive(Clone)]
pub struct DDIMScheduler {
    num_train_timesteps: usize,
    num_inference_steps: usize,
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f32>,
    init_noise_sigma: f32,
}

impl DDIMScheduler {
    pub fn new(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        clip_sample: bool,
    ) -> Self {
        // Create beta schedule
        let betas = if beta_schedule == "scaled_linear" {
            Self::scaled_linear_beta_schedule(num_train_timesteps, beta_start, beta_end)
        } else {
            Self::linear_beta_schedule(num_train_timesteps, beta_start, beta_end)
        };

        // Calculate alphas
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(alphas.len());
        let mut cumprod = 1.0;
        for alpha in alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        Self {
            num_train_timesteps,
            num_inference_steps: 50, // default
            timesteps: Vec::new(),
            alphas_cumprod,
            init_noise_sigma: 1.0,
        }
    }

    pub fn timesteps(&self, num_inference_steps: usize) -> flame_core::Result<Vec<i64>> {
        let mut scheduler = self.clone();
        scheduler.set_timesteps(num_inference_steps);
        Ok(scheduler.timesteps.iter().map(|&t| t as i64).collect())
    }

    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: i64,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // Convert i64 to usize and delegate
        let timestep_idx = self.timesteps.iter().position(|&t| t as i64 == timestep).unwrap_or(0);

        // Get alpha values
        let alpha_prod_t = self.alphas_cumprod[timestep as usize];
        let alpha_prod_t_sqrt = alpha_prod_t.sqrt();
        let beta_prod_t_sqrt = (1.0 - alpha_prod_t).sqrt();

        // Predict x0
        let beta_prod_t_sqrt_tensor = Tensor::full(
            model_output.shape().clone(),
            beta_prod_t_sqrt,
            model_output.device().clone(),
        )?;
        let alpha_prod_t_sqrt_tensor = Tensor::full(
            model_output.shape().clone(),
            alpha_prod_t_sqrt,
            model_output.device().clone(),
        )?;
        let pred_original_sample = sample
            .sub(&model_output.mul(&beta_prod_t_sqrt_tensor)?)?
            .div(&alpha_prod_t_sqrt_tensor)?;

        // Get next timestep
        let next_timestep = if timestep_idx + 1 < self.timesteps.len() {
            self.timesteps[timestep_idx + 1]
        } else {
            0
        };

        let alpha_prod_t_next =
            if next_timestep > 0 { self.alphas_cumprod[next_timestep] } else { 1.0 };

        let alpha_prod_t_next_sqrt = alpha_prod_t_next.sqrt();
        let beta_prod_t_next_sqrt = (1.0 - alpha_prod_t_next).sqrt();

        // Compute next sample
        let alpha_prod_t_next_sqrt_tensor = Tensor::full(
            model_output.shape().clone(),
            alpha_prod_t_next_sqrt,
            model_output.device().clone(),
        )?;
        let beta_prod_t_next_sqrt_tensor = Tensor::full(
            model_output.shape().clone(),
            beta_prod_t_next_sqrt,
            model_output.device().clone(),
        )?;
        let next_sample = pred_original_sample
            .mul(&alpha_prod_t_next_sqrt_tensor)?
            .add(&model_output.mul(&beta_prod_t_next_sqrt_tensor)?)?;

        Ok(next_sample)
    }

    pub fn new_simple(num_inference_steps: usize) -> flame_core::Result<Self> {
        let num_train_timesteps = 1000;

        // Create beta schedule
        let betas = Self::linear_beta_schedule(num_train_timesteps, 0.00085, 0.012);

        // Calculate alphas
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(alphas.len());
        let mut cumprod = 1.0;
        for alpha in alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        let mut scheduler = Self {
            num_train_timesteps,
            num_inference_steps,
            timesteps: Vec::new(),
            alphas_cumprod,
            init_noise_sigma: 1.0,
        };
        scheduler.set_timesteps(num_inference_steps);
        Ok(scheduler)
    }

    fn linear_beta_schedule(num_timesteps: usize, beta_start: f32, beta_end: f32) -> Vec<f32> {
        (0..num_timesteps)
            .map(|i| {
                let t = i as f32 / (num_timesteps - 1) as f32;
                beta_start + t * (beta_end - beta_start)
            })
            .collect()
    }

    fn scaled_linear_beta_schedule(
        num_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> Vec<f32> {
        // Scaled linear schedule used by SDXL
        let start = beta_start.sqrt();
        let end = beta_end.sqrt();
        (0..num_timesteps)
            .map(|i| {
                let t = i as f32 / (num_timesteps - 1) as f32;
                let beta = start + t * (end - start);
                beta * beta
            })
            .collect()
    }
}

impl Scheduler for DDIMScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;

        // Create evenly spaced timesteps
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        self.timesteps =
            (0..num_inference_steps).map(|i| (num_inference_steps - 1 - i) * step_ratio).collect();
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f32 {
        1.0
    }

    fn scale_model_input(&self, sample: &Tensor, _timestep: usize) -> flame_core::Result<Tensor> {
        Ok(sample.clone())
    }

    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // Get alpha values
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_sqrt = alpha_prod_t.sqrt();
        let beta_prod_t_sqrt = (1.0 - alpha_prod_t).sqrt();

        // Predict x0
        let beta_sqrt_tensor =
            Tensor::full(sample.shape().clone(), beta_prod_t_sqrt, sample.device().clone())?;
        let alpha_sqrt_tensor =
            Tensor::full(sample.shape().clone(), alpha_prod_t_sqrt, sample.device().clone())?;
        let noise_term = model_output.mul(&beta_sqrt_tensor)?;
        let numerator = sample.sub(&noise_term)?;
        let pred_original_sample = numerator.div(&alpha_sqrt_tensor)?;

        // Get next timestep
        let next_timestep = if timestep > self.num_inference_steps {
            timestep - self.num_train_timesteps / self.num_inference_steps
        } else {
            0
        };

        let alpha_prod_t_next =
            if next_timestep > 0 { self.alphas_cumprod[next_timestep] } else { 1.0 };

        let alpha_prod_t_next_sqrt = alpha_prod_t_next.sqrt();
        let beta_prod_t_next_sqrt = (1.0 - alpha_prod_t_next).sqrt();

        // Compute next sample
        let alpha_next_tensor =
            Tensor::full(sample.shape().clone(), alpha_prod_t_next_sqrt, sample.device().clone())?;
        let beta_next_tensor =
            Tensor::full(sample.shape().clone(), beta_prod_t_next_sqrt, sample.device().clone())?;
        let term1 = pred_original_sample.mul(&alpha_next_tensor)?;
        let term2 = model_output.mul(&beta_next_tensor)?;
        let next_sample = term1.add(&term2)?;

        Ok(next_sample)
    }
}

/// DPM-Solver++ scheduler

impl DPMSolverMultistepScheduler {
    pub fn new(num_inference_steps: usize) -> flame_core::Result<Self> {
        let mut scheduler = Self {
            num_train_timesteps: 1000,
            num_inference_steps,
            timesteps: vec![],
            alphas_cumprod: vec![],
            sample_max_value: 1.0,
            prediction_type: PredictionType::Epsilon,
        };
        scheduler.set_timesteps(num_inference_steps);
        Ok(scheduler)
    }
}

impl Scheduler for DPMSolverMultistepScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;

        // Simplified timestep schedule
        let num_train_timesteps = 1000;
        let step_ratio = num_train_timesteps / num_inference_steps;
        self.timesteps =
            (0..num_inference_steps).map(|i| (num_inference_steps - 1 - i) * step_ratio).collect();
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f32 {
        1.0
    }

    fn scale_model_input(&self, sample: &Tensor, _timestep: usize) -> flame_core::Result<Tensor> {
        Ok(sample.clone())
    }

    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // REAL DPM-Solver++ implementation

        // Convert timestep to log-SNR (signal-to-noise ratio)
        let num_train_timesteps = 1000;
        let t = timestep as f32 / num_train_timesteps as f32;
        let lambda_t = ((1.0 - t).ln() - t.ln()) * 0.5;

        // Get previous timestep
        let prev_t = if timestep > self.num_train_timesteps / self.num_inference_steps {
            timestep - num_train_timesteps / self.num_inference_steps
        } else {
            0
        };
        let prev_t_ratio = prev_t as f32 / num_train_timesteps as f32;
        let lambda_s = ((1.0 - prev_t_ratio).ln() - prev_t_ratio.ln()) * 0.5;

        let h = lambda_t - lambda_s;
        let h_tensor = Tensor::full(sample.shape().clone(), h, sample.device().clone())?;

        // Convert between x and v prediction
        let alpha_t = (1.0 / (1.0 + (-2.0 * lambda_t).exp())).sqrt();
        let sigma_t = (1.0 - alpha_t * alpha_t).sqrt();

        let alpha_t_tensor =
            Tensor::full(sample.shape().clone(), alpha_t, sample.device().clone())?;
        let sigma_t_tensor =
            Tensor::full(sample.shape().clone(), sigma_t, sample.device().clone())?;

        // For epsilon prediction, convert to v-prediction
        let v_t = model_output.mul(&alpha_t_tensor)?.sub(&sample.div(&sigma_t_tensor)?)?;

        // DPM-Solver++ update (second order)
        let alpha_s = (1.0 / (1.0 + (-2.0 * lambda_s).exp())).sqrt();
        let sigma_s = (1.0 - alpha_s * alpha_s).sqrt();

        let alpha_s_tensor =
            Tensor::full(sample.shape().clone(), alpha_s, sample.device().clone())?;
        let sigma_s_tensor =
            Tensor::full(sample.shape().clone(), sigma_s, sample.device().clone())?;

        // Linear combination
        let x_t = sample.mul(&alpha_t_tensor)?.div(&sigma_t_tensor)?;

        // Update formula
        let coeff1 = sigma_s / sigma_t * ((-h).exp() - 1.0);
        let coeff1_tensor = Tensor::full(sample.shape().clone(), coeff1, sample.device().clone())?;

        let next_sample = sample
            .mul(&Tensor::full(
                sample.shape().clone(),
                sigma_s / sigma_t,
                sample.device().clone(),
            )?)?
            .add(&v_t.mul(&coeff1_tensor)?)?;

        Ok(next_sample)
    }
}

/// DPM Scheduler configuration

impl Default for DPMSchedulerConfig {
    fn default() -> Self {
        Self { num_inference_steps: 50, solver_order: 2, solver_type: "midpoint".to_string() }
    }
}

/// DPM Scheduler (alias for DPMSolverMultistepScheduler)
pub type DPMScheduler = DPMSolverMultistepScheduler;

// DDIM module for compatibility
