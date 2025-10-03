//! Deterministic samplers (Euler and Heun) with classifier-free guidance.
//! Caller supplies a `denoise` closure returning the model output in the configured prediction domain.

use eridiffusion_core::{Device, Error, Result};
use flame_core::Tensor;

use super::pred_convert::model_out_to_eps;
use super::scheduler::{sigma_to_timestep, Sigmas};

#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub scale: f32,
}

impl Default for Cfg {
    fn default() -> Self {
        Self { scale: 7.0 }
    }
}

fn cfg_mix(uncond: &Tensor, cond: &Tensor, cfg: Cfg) -> Result<Tensor> {
    let delta = cond.sub(uncond).map_err(Error::from)?;
    let scaled = delta.mul_scalar(cfg.scale).map_err(Error::from)?;
    uncond.add(&scaled).map_err(Error::from)
}

/// Deterministic Euler sampler in sigma parameterisation.
/// `denoise(x, t, ctx)` must return model output in `pred_type` space ("eps" or "v").
pub fn sample_euler<F>(
    mut x: Tensor,
    sigmas: &Sigmas,
    pred_type: &str,
    cfg: Cfg,
    device: &Device,
    cond: &Tensor,
    uncond: &Tensor,
    mut denoise: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
{
    if sigmas.len() < 2 {
        return Ok(x);
    }
    let batch = x.shape()[0] as usize;
    for i in 0..(sigmas.len() - 1) {
        let sigma = sigmas.get(i);
        let t = sigma_to_timestep(batch, sigma, device)?;
        let out_c = denoise(&x, &t, cond)?;
        let out_u = denoise(&x, &t, uncond)?;
        let eps_c = model_out_to_eps(pred_type, &x, &out_c, sigma)?;
        let eps_u = model_out_to_eps(pred_type, &x, &out_u, sigma)?;
        let eps = cfg_mix(&eps_u, &eps_c, cfg)?;
        let ds = sigmas.get(i + 1) - sigma;
        let step = eps.mul_scalar(ds).map_err(Error::from)?;
        x = x.add(&step).map_err(Error::from)?;
    }
    Ok(x)
}

/// Heun's second-order sampler (deterministic). Often yields higher quality than Euler at the same step count.
pub fn sample_heun<F>(
    mut x: Tensor,
    sigmas: &Sigmas,
    pred_type: &str,
    cfg: Cfg,
    device: &Device,
    cond: &Tensor,
    uncond: &Tensor,
    mut denoise: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
{
    if sigmas.len() < 2 {
        return Ok(x);
    }
    let batch = x.shape()[0] as usize;
    for i in 0..(sigmas.len() - 1) {
        let s0 = sigmas.get(i);
        let t0 = sigma_to_timestep(batch, s0, device)?;
        let out_c0 = denoise(&x, &t0, cond)?;
        let out_u0 = denoise(&x, &t0, uncond)?;
        let eps_c0 = model_out_to_eps(pred_type, &x, &out_c0, s0)?;
        let eps_u0 = model_out_to_eps(pred_type, &x, &out_u0, s0)?;
        let eps0 = cfg_mix(&eps_u0, &eps_c0, cfg)?;

        let ds = sigmas.get(i + 1) - s0;
        let step0 = eps0.clone().mul_scalar(ds).map_err(Error::from)?;
        let x_euler = x.add(&step0).map_err(Error::from)?;

        let s1 = sigmas.get(i + 1);
        let t1 = sigma_to_timestep(batch, s1, device)?;
        let out_c1 = denoise(&x_euler, &t1, cond)?;
        let out_u1 = denoise(&x_euler, &t1, uncond)?;
        let eps_c1 = model_out_to_eps(pred_type, &x_euler, &out_c1, s1)?;
        let eps_u1 = model_out_to_eps(pred_type, &x_euler, &out_u1, s1)?;
        let eps1 = cfg_mix(&eps_u1, &eps_c1, cfg)?;

        let half0 = eps0.mul_scalar(0.5).map_err(Error::from)?;
        let half1 = eps1.mul_scalar(0.5).map_err(Error::from)?;
        let avg = half0.add(&half1).map_err(Error::from)?;
        let step = avg.mul_scalar(ds).map_err(Error::from)?;
        x = x.add(&step).map_err(Error::from)?;
    }
    Ok(x)
}
