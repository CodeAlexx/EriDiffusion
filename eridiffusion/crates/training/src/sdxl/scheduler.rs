use eridiffusion_core::{Device, Error, Result};
use eridiffusion_models::devtensor::tensor_from_vec_on;
use flame_core::{DType, Shape, Tensor};

#[derive(Clone, Copy, Debug)]
pub enum ScheduleKind {
    Karras,
    LogLinear,
}

#[derive(Clone, Debug)]
pub struct SchedulerCfg {
    pub steps: usize,
    pub sigma_min: f32,
    pub sigma_max: f32,
    pub rho: f32,
    pub kind: ScheduleKind,
}

impl Default for SchedulerCfg {
    fn default() -> Self {
        Self { steps: 30, sigma_min: 0.029, sigma_max: 14.0, rho: 7.0, kind: ScheduleKind::Karras }
    }
}

#[derive(Clone, Debug)]
pub struct Sigmas {
    values: Vec<f32>,
}

impl Sigmas {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn get(&self, index: usize) -> f32 {
        self.values[index]
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        self.values.iter().copied()
    }
}

pub fn make_sigmas(cfg: &SchedulerCfg) -> Sigmas {
    match cfg.kind {
        ScheduleKind::Karras => karras_sigmas(cfg.steps, cfg.sigma_min, cfg.sigma_max, cfg.rho),
        ScheduleKind::LogLinear => loglinear_sigmas(cfg.steps, cfg.sigma_min, cfg.sigma_max),
    }
}

fn karras_sigmas(steps: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Sigmas {
    if steps == 0 {
        return Sigmas { values: vec![sigma_max, sigma_min] };
    }
    let denom = steps as f32;
    let ramp: Vec<f32> = (0..=steps).map(|i| i as f32 / denom).collect();
    let min_inv = sigma_min.powf(1.0 / rho);
    let max_inv = sigma_max.powf(1.0 / rho);
    let values = ramp.into_iter().map(|t| (max_inv + t * (min_inv - max_inv)).powf(rho)).collect();
    Sigmas { values }
}

fn loglinear_sigmas(steps: usize, sigma_min: f32, sigma_max: f32) -> Sigmas {
    if steps == 0 {
        return Sigmas { values: vec![sigma_max, sigma_min] };
    }
    let log_min = sigma_min.ln();
    let log_max = sigma_max.ln();
    let denom = steps as f32;
    let values = (0..=steps)
        .map(|i| {
            let r = i as f32 / denom;
            (log_max + r * (log_min - log_max)).exp()
        })
        .collect();
    Sigmas { values }
}

/// Build a [batch]-length FP32 tensor filled with `sigma`.
pub fn sigma_to_timestep(batch: usize, sigma: f32, device: &Device) -> Result<Tensor> {
    let data = vec![sigma; batch];
    tensor_from_vec_on(data, Shape::from_dims(&[batch]), device, DType::F32)
        .map_err(|e| Error::InvalidInput(format!("sigma_to_timestep: {e}")))
}
