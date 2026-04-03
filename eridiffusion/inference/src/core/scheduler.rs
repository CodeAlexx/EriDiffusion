use anyhow::Result;
use flame_core::Tensor;

/// Minimal Euler scheduler configuration.
#[derive(Clone, Copy, Debug)]
pub struct EulerSchedulerConfig {
    pub num_train_steps: usize,
    pub num_inference_steps: usize,
    pub sigma_min: f32,
    pub sigma_max: f32,
}

/// Simple Euler scheduler mirroring Candle’s behaviour.
#[derive(Clone, Debug)]
pub struct EulerScheduler {
    sigmas: Vec<f32>,
}

impl EulerScheduler {
    pub fn new(cfg: EulerSchedulerConfig) -> Self {
        let EulerSchedulerConfig { num_inference_steps, sigma_min, sigma_max, .. } = cfg;
        // Karras sigma schedule
        let sigmas = karras_sigmas(num_inference_steps, sigma_min, sigma_max);
        Self { sigmas }
    }

    pub fn len(&self) -> usize {
        self.sigmas.len()
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn step(&self, latents: &Tensor, eps: &Tensor, idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas[idx];
        let sigma_next = self.sigmas[idx + 1];
        let dt = sigma_next - sigma;
        Ok(latents.add(&eps.mul_scalar(dt)?)?)
    }
}

fn karras_sigmas(steps: usize, sigma_min: f32, sigma_max: f32) -> Vec<f32> {
    if steps < 2 {
        return vec![sigma_max];
    }
    let rho = 7.0f32;
    let min_inv = sigma_min.powf(1.0 / rho);
    let max_inv = sigma_max.powf(1.0 / rho);
    let ramp = (max_inv - min_inv) / ((steps - 1) as f32);
    (0..steps).map(|i| (max_inv - i as f32 * ramp).max(0.0).powf(rho)).collect()
}
