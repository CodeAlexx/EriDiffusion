//! Flux 1 sampling — FlowMatchEulerDiscreteScheduler with sequence-length-dependent shift.
//!
//! Reference: BFL `flux/sampling.py:get_schedule`,
//! `flame-diffusion/flux1-trainer/src/sampling.rs:euler_schedule`,
//! `/home/alex/upstream Python/modules/model/FluxModel.py:create_noise_scheduler`.
//!
//! Linearly-interpolated mu(image_seq_len): mu(256)=0.5, mu(4096)=1.15.
//! For 512²: image_seq_len = (512/8/2)² = 1024, mu ≈ 0.63.
//!
//! Time-shift: `shifted = exp(mu) / (exp(mu) + (1/t - 1)^sigma_exp)` with sigma_exp=1.

use flame_core::{Result, Tensor};

/// VAE spatial compression × patch size = 8 × 2 = 16 px per latent token.
pub const VAE_SPATIAL_SCALE: usize = 8;
pub const PATCH_SIZE: usize = 2;

/// Linearly interpolate mu based on packed-token count.
/// mu(256) = 0.5  (512×512)
/// mu(4096) = 1.15 (2048×2048)
pub fn shift_mu_for_resolution(width: usize, height: usize) -> f32 {
    let image_seq_len = (height / VAE_SPATIAL_SCALE / PATCH_SIZE)
        * (width / VAE_SPATIAL_SCALE / PATCH_SIZE);
    let t = ((image_seq_len as f32 - 256.0) / (4096.0 - 256.0)).clamp(0.0, 1.0);
    0.5 + t * (1.15 - 0.5)
}

/// Time-shift function, matches sd-scripts flux_train_utils.time_shift.
fn time_shift(mu: f32, sigma_exp: f32, t: f32) -> f32 {
    let exp_mu = mu.exp();
    exp_mu / (exp_mu + (1.0 / t.max(1e-6) - 1.0).powf(sigma_exp))
}

/// Build sigma schedule of length `num_steps + 1` from 1.0 → 0.0 with
/// resolution-dependent shift applied to each interior point.
/// Endpoints are exact: `sigmas[0] = 1.0`, `sigmas[num_steps] = 0.0`.
pub fn schedule(num_steps: usize, width: usize, height: usize) -> Vec<f32> {
    let mu = shift_mu_for_resolution(width, height);
    let mut s = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        if i == num_steps {
            s.push(0.0);
            continue;
        }
        let t = 1.0 - (i as f32 / num_steps as f32);
        s.push(time_shift(mu, 1.0, t));
    }
    s
}

/// Euler ODE step: `x_next = x + (sigma_next - sigma) * pred`.
pub fn euler_step(x: &Tensor, pred: &Tensor, sigma: f32, sigma_next: f32) -> Result<Tensor> {
    let dt = sigma_next - sigma;
    x.add(&pred.mul_scalar(dt)?)
}

/// Pack [B, 16, H, W] VAE latent → [B, N_img, 64] Flux DiT input.
/// BFL: `rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)`.
pub fn pack_latents(latents: &Tensor) -> Result<(Tensor, usize, usize)> {
    let dims = latents.shape().dims().to_vec();
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let h_tok = h / PATCH_SIZE;
    let w_tok = w / PATCH_SIZE;
    let patch_dim = c * PATCH_SIZE * PATCH_SIZE;
    let x = latents.reshape(&[b, c, h_tok, PATCH_SIZE, w_tok, PATCH_SIZE])?;
    let x = x.permute(&[0, 2, 4, 1, 3, 5])?;
    let x = x.reshape(&[b, h_tok * w_tok, patch_dim])?;
    Ok((x, h_tok, w_tok))
}

/// Unpack [B, N_img, 64] → [B, 16, H, W] (inverse of pack_latents).
pub fn unpack_latents(x: &Tensor, h_tok: usize, w_tok: usize) -> Result<Tensor> {
    let b = x.shape().dims()[0];
    let x = x.reshape(&[b, h_tok, w_tok, 16, PATCH_SIZE, PATCH_SIZE])?;
    let x = x.permute(&[0, 3, 1, 4, 2, 5])?;
    x.reshape(&[b, 16, h_tok * PATCH_SIZE, w_tok * PATCH_SIZE])
}

/// Build image position IDs [N_img, 3] for RoPE.
/// BFL: `img_ids[:, 1] = row, img_ids[:, 2] = col`.
pub fn build_img_ids(
    h_tok: usize,
    w_tok: usize,
    device: std::sync::Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    use flame_core::Shape;
    let n = h_tok * w_tok;
    let mut data = vec![0.0f32; n * 3];
    for row in 0..h_tok {
        for col in 0..w_tok {
            let idx = row * w_tok + col;
            data[idx * 3 + 1] = row as f32;
            data[idx * 3 + 2] = col as f32;
        }
    }
    Tensor::from_vec(data, Shape::from_dims(&[n, 3]), device)
}

/// Build text position IDs [N_txt, 3] for RoPE — all zeros (BFL convention).
pub fn build_txt_ids(
    txt_len: usize,
    device: std::sync::Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    use flame_core::Shape;
    let data = vec![0.0f32; txt_len * 3];
    Tensor::from_vec(data, Shape::from_dims(&[txt_len, 3]), device)
}
