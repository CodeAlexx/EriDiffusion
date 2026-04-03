//! Prediction conversions between epsilon and v-parameterization.

use eridiffusion_core::{Error, Result};
use flame_core::Tensor;

/// Convert model output into epsilon space depending on the prediction type (`"eps"` or `"v"`).
pub fn model_out_to_eps(pred_type: &str, x: &Tensor, model_out: &Tensor, sigma: f32) -> Result<Tensor> {
    match pred_type {
        "eps" => Ok(model_out.clone()),
        "v" => {
            let c1 = 1.0f32 / (1.0 + sigma * sigma).sqrt();
            let term = x.mul_scalar(c1).map_err(Error::from)?;
            term.sub(model_out).map_err(Error::from)
        }
        other => Err(Error::from_msg(format!("unknown pred_type={other}"))),
    }
}
