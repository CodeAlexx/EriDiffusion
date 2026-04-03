use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Tensor};

#[derive(Clone, Copy, Debug)]
pub enum SchedulerType {
    DDPM,
    DDIM,
    Euler,
    EulerA,
    DPMSolver,
}

/// Common trait for all schedulers
pub trait Scheduler {
    fn timesteps(&self) -> &[usize];
    fn add_noise(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timestep: usize,
    ) -> flame_core::Result<Tensor>;
    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor>;
}

/// Enum wrapper for different scheduler implementations
pub enum SchedulerWrapper {
    DDPM(DDPMScheduler),
    DDIM(DDIMScheduler),
}

impl Scheduler for SchedulerWrapper {
    fn timesteps(&self) -> &[usize] {
        match self {
            SchedulerWrapper::DDPM(s) => s.timesteps(),
            SchedulerWrapper::DDIM(s) => s.timesteps(),
        }
    }

    fn add_noise(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timestep: usize,
    ) -> flame_core::Result<Tensor> {
        match self {
            SchedulerWrapper::DDPM(s) => s.add_noise(original, noise, timestep),
            SchedulerWrapper::DDIM(s) => s.add_noise(original, noise, timestep),
        }
    }

    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor> {
        match self {
            SchedulerWrapper::DDPM(s) => s.step(model_output, timestep, sample),
            SchedulerWrapper::DDIM(s) => s.step(model_output, timestep, sample),
        }
    }
}

/// DDPM (Denoising Diffusion Probabilistic Models) Scheduler
pub struct DDPMScheduler {
    num_inference_steps: usize,
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f32>,
    betas: Vec<f32>,
}

impl Scheduler for DDPMScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn add_noise(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timestep: usize,
    ) -> flame_core::Result<Tensor> {
        let alpha_cumprod = self.alphas_cumprod[timestep];
        let sqrt_alpha_cumprod = alpha_cumprod.sqrt();
        let sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod).sqrt();

        // noisy = sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise
        let scaled_original = original.mul_scalar(sqrt_alpha_cumprod)?;
        let scaled_noise = noise.mul_scalar(sqrt_one_minus_alpha_cumprod)?;

        scaled_original.add(&scaled_noise)
    }

    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor> {
        let t = timestep;
        let prev_t = if t > 0 {
            self.timesteps[self.timesteps.iter().position(|&x| x == t).unwrap() + 1]
        } else {
            0
        };

        let alpha_t = self.alphas_cumprod[t];
        let alpha_prev = if prev_t > 0 { self.alphas_cumprod[prev_t] } else { 1.0 };
        let beta_t = self.betas[t];

        // Compute predicted original sample
        let pred_original = sample
            .sub(&model_output.mul_scalar((1.0 - alpha_t).sqrt() as f32)?)?
            .div_scalar(alpha_t.sqrt() as f32)?;

        // Compute variance
        let variance = (1.0 - alpha_prev) / (1.0 - alpha_t) * beta_t;
        let std_dev = variance.sqrt();

        // Compute predicted previous sample mean
        let pred_prev_mean = pred_original
            .mul_scalar(((alpha_prev * beta_t) / (1.0 - alpha_t)).sqrt() as f32)?
            .add(
                &sample
                    .mul_scalar(((1.0 - alpha_prev - variance) / (1.0 - alpha_t)).sqrt() as f32)?,
            )?;

        // Add noise for non-zero timesteps
        if t > 0 {
            let noise = Tensor::randn(sample.shape().clone(), 0.0, 1.0, sample.device().clone())?;
            pred_prev_mean.add(&noise.mul_scalar(std_dev as f32)?)
        } else {
            Ok(pred_prev_mean)
        }
    }
}

impl DDPMScheduler {
    pub fn new(num_inference_steps: usize) -> Self {
        let num_train_steps = 1000;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .map(|i| (i * num_train_steps) / num_inference_steps)
            .rev()
            .collect();

        // Linear beta schedule
        let beta_start = 0.00085;
        let beta_end = 0.012;
        let betas: Vec<f32> = (0..num_train_steps)
            .map(|i| {
                let t = i as f32 / (num_train_steps - 1) as f32;
                beta_start + t * (beta_end - beta_start)
            })
            .collect();

        // Calculate alphas_cumprod
        let mut alphas_cumprod = Vec::with_capacity(num_train_steps);
        let mut cumprod = 1.0;
        for beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        Self { num_inference_steps, timesteps, alphas_cumprod, betas }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn add_noise(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timestep: usize,
    ) -> flame_core::Result<Tensor> {
        let alpha_cumprod = self.alphas_cumprod[timestep];
        let sqrt_alpha_cumprod = alpha_cumprod.sqrt();
        let sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod).sqrt();

        // noisy = sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise
        let scaled_original = original.mul_scalar(sqrt_alpha_cumprod)?;
        let scaled_noise = noise.mul_scalar(sqrt_one_minus_alpha_cumprod)?;

        scaled_original.add(&scaled_noise)
    }

    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor> {
        let t = timestep;
        let prev_t = if t > 0 {
            self.timesteps[self.timesteps.iter().position(|&x| x == t).unwrap() + 1]
        } else {
            0
        };

        let alpha_t = self.alphas_cumprod[t];
        let alpha_prev = if prev_t > 0 { self.alphas_cumprod[prev_t] } else { 1.0 };
        let beta_t = self.betas[t];

        // Compute predicted original sample
        let pred_original = sample
            .sub(&model_output.mul_scalar((1.0 - alpha_t).sqrt() as f32)?)?
            .div_scalar(alpha_t.sqrt() as f32)?;

        // Compute variance
        let variance = (1.0 - alpha_prev) / (1.0 - alpha_t) * beta_t;
        let std_dev = variance.sqrt();

        // Compute predicted previous sample mean
        let pred_prev_mean = pred_original
            .mul_scalar(((alpha_prev * beta_t) / (1.0 - alpha_t)).sqrt() as f32)?
            .add(
                &sample
                    .mul_scalar(((1.0 - alpha_prev - variance) / (1.0 - alpha_t)).sqrt() as f32)?,
            )?;

        // Add noise for non-zero timesteps
        if t > 0 {
            let noise = Tensor::randn(sample.shape().clone(), 0.0, 1.0, sample.device().clone())?;
            pred_prev_mean.add(&noise.mul_scalar(std_dev as f32)?)
        } else {
            Ok(pred_prev_mean)
        }
    }
}

/// DDIM (Denoising Diffusion Implicit Models) Scheduler
pub struct DDIMScheduler {
    num_inference_steps: usize,
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f32>,
    eta: f32,
}

impl Scheduler for DDIMScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn add_noise(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timestep: usize,
    ) -> flame_core::Result<Tensor> {
        let alpha_cumprod = self.alphas_cumprod[timestep];
        let sqrt_alpha_cumprod = alpha_cumprod.sqrt();
        let sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod).sqrt();

        let scaled_original = original.mul_scalar(sqrt_alpha_cumprod)?;
        let scaled_noise = noise.mul_scalar(sqrt_one_minus_alpha_cumprod)?;

        scaled_original.add(&scaled_noise)
    }

    fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor> {
        let t = timestep;
        let prev_t = if t > 0 {
            self.timesteps[self.timesteps.iter().position(|&x| x == t).unwrap() + 1]
        } else {
            0
        };

        let alpha_t = self.alphas_cumprod[t];
        let alpha_prev = if prev_t > 0 { self.alphas_cumprod[prev_t] } else { 1.0 };

        // Compute predicted original sample
        let pred_original = sample
            .sub(&model_output.mul_scalar((1.0 - alpha_t).sqrt() as f32)?)?
            .div_scalar(alpha_t.sqrt() as f32)?;

        // Compute variance with eta
        let variance =
            self.eta * ((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev));
        let std_dev = variance.sqrt();

        // Compute direction pointing to x_t
        let pred_sample_direction = model_output
            .mul_scalar(((1.0 - alpha_prev - variance) / (1.0 - alpha_t)).sqrt() as f32)?;

        // Compute predicted previous sample
        let pred_prev_sample = pred_original
            .mul_scalar((alpha_prev - variance as f32).sqrt() as f32)?
            .add(&pred_sample_direction)?;

        // Add noise if eta > 0 and not last step
        if variance > 0.0 && t > 0 {
            let noise = Tensor::randn(sample.shape().clone(), 0.0, 1.0, sample.device().clone())?;
            pred_prev_sample.add(&noise.mul_scalar(std_dev as f32)?)
        } else {
            Ok(pred_prev_sample)
        }
    }
}

impl DDIMScheduler {
    pub fn new(num_inference_steps: usize, eta: f32) -> Self {
        let num_train_steps = 1000;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .map(|i| (i * num_train_steps) / num_inference_steps)
            .rev()
            .collect();

        // Linear beta schedule
        let beta_start = 0.00085;
        let beta_end = 0.012;
        let betas: Vec<f32> = (0..num_train_steps)
            .map(|i| {
                let t = i as f32 / (num_train_steps - 1) as f32;
                beta_start + t * (beta_end - beta_start)
            })
            .collect();

        // Calculate alphas_cumprod
        let mut alphas_cumprod = Vec::with_capacity(num_train_steps);
        let mut cumprod = 1.0;
        for beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        Self { num_inference_steps, timesteps, alphas_cumprod, eta }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn add_noise(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timestep: usize,
    ) -> flame_core::Result<Tensor> {
        let alpha_cumprod = self.alphas_cumprod[timestep];
        let sqrt_alpha_cumprod = alpha_cumprod.sqrt();
        let sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod).sqrt();

        let scaled_original = original.mul_scalar(sqrt_alpha_cumprod)?;
        let scaled_noise = noise.mul_scalar(sqrt_one_minus_alpha_cumprod)?;

        scaled_original.add(&scaled_noise)
    }

    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> flame_core::Result<Tensor> {
        let t = timestep;
        let prev_t = if t > 0 {
            self.timesteps[self.timesteps.iter().position(|&x| x == t).unwrap() + 1]
        } else {
            0
        };

        let alpha_t = self.alphas_cumprod[t];
        let alpha_prev = if prev_t > 0 { self.alphas_cumprod[prev_t] } else { 1.0 };

        // Compute predicted original sample
        let pred_original = sample
            .sub(&model_output.mul_scalar((1.0 - alpha_t).sqrt() as f32)?)?
            .div_scalar(alpha_t.sqrt() as f32)?;

        // Compute variance with eta
        let variance =
            self.eta * ((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev));
        let std_dev = variance.sqrt();

        // Compute direction pointing to x_t
        let pred_sample_direction = model_output
            .mul_scalar(((1.0 - alpha_prev - variance) / (1.0 - alpha_t)).sqrt() as f32)?;

        // Compute predicted previous sample
        let pred_prev_sample = pred_original
            .mul_scalar((alpha_prev - variance).sqrt() as f32)?
            .add(&pred_sample_direction)?;

        // Add noise if eta > 0 and not last step
        if variance > 0.0 && t > 0 {
            let noise = Tensor::randn(sample.shape().clone(), 0.0, 1.0, sample.device().clone())?;
            pred_prev_sample.add(&noise.mul_scalar(std_dev as f32)?)
        } else {
            Ok(pred_prev_sample)
        }
    }
}
