use flame_core::{Result as FlameResult, Tensor};

/// Euler A sampler step for velocity/epsilon predictions.
/// x: current sample, t: current timestep scalar, dt: step size, predict: model closure producing velocity/epsilon
/// Classifier-free guidance: if uncond is Some, apply cfg: uncond + gs*(cond-uncond), else use cond as-is.
pub fn euler_a_step<F>(
    x: &Tensor,
    t: &Tensor,
    dt: f32,
    guidance_scale: Option<f32>,
    predict: F,
) -> FlameResult<Tensor>
where
    F: Fn(&Tensor, &Tensor, bool) -> FlameResult<Tensor>, // (x, t, is_uncond)
{
    let pred = if let Some(gs) = guidance_scale {
        let uncond = predict(x, t, true)?;
        let cond = predict(x, t, false)?;
        // cond_guided = uncond + gs*(cond - uncond)
        cond.sub(&uncond)?.mul_scalar(gs)?.add(&uncond)?
    } else {
        predict(x, t, false)?
    };
    // Euler update: x_{k+1} = x_k + dt * v(x_k, t_k)
    x.add(&pred.mul_scalar(dt)?)
}
