//! SNR (Signal-to-Noise Ratio) weighting for diffusion training

use anyhow::Result;
use candle_core::{Tensor, Device};

/// Compute v-prediction from noise prediction and other components
pub fn compute_v_prediction(
    noise_pred: &Tensor,
    latents: &Tensor,
    noise: &Tensor,
    alphas_cumprod: &Tensor,
    timesteps: &Tensor,
) -> Result<Tensor> {
    // Get alpha values for the given timesteps
    let timestep_values = timesteps.to_vec1::<i64>()?;
    let batch_size = timestep_values.len();
    
    // Gather alpha values for each timestep
    let mut alpha_values = Vec::with_capacity(batch_size);
    let alphas = alphas_cumprod.to_vec1::<f32>()?;
    
    for &t in &timestep_values {
        let idx = t.min(alphas.len() as i64 - 1).max(0) as usize;
        alpha_values.push(alphas[idx]);
    }
    
    // Create tensors for alpha and sigma
    let alpha_t = Tensor::from_slice(&alpha_values, &[batch_size], timesteps.device())?;
    let alpha_t = alpha_t.reshape(&[batch_size, 1, 1, 1])?;
    let sigma_t = (1.0 - &alpha_t)?.sqrt()?;
    
    // v = alpha_t * noise - sigma_t * latents
    let v_target = (&alpha_t * noise)? - (&sigma_t * latents)?;
    
    Ok(v_target)
}