//! SDXL sampler — DDIM and Euler-Ancestral schedulers on the standard
//! scaled-linear DDPM schedule.
//!
//! Reference: sd-scripts `sdxl_train.py` (training schedule), Song et al.
//! 2020 DDIM eq. 12 (deterministic σ=0 case), Karras et al. 2022
//! Euler-Ancestral. SDXL preset (`#sdxl 1.0 LoRA.json`) does not set
//! `force_v_prediction`, so the default target is **epsilon**; we expose a
//! `Prediction` enum so the caller can opt into v-prediction for cosmos-style
//! checkpoints. SDXL audit H4: OT preset default scheduler is **Euler-A
//! with 30 steps**; DDIM is retained as a legacy / determinism opt-in.
//!
//! Schedule:
//!   - num_train_timesteps = 1000
//!   - β_start = 0.00085, β_end = 0.012, schedule = "scaled_linear"
//!     i.e. β_t = (sqrt(β_s) + t*(sqrt(β_e) - sqrt(β_s)))²
//!   - α_t = 1 - β_t,  ᾱ_t = ∏ α_t
//!
//! Sampler exposes:
//!   - `compute_alpha_bar()` — full ᾱ table (length 1001 to allow t=1000 lookup)
//!   - `Prediction` enum
//!   - `SchedulerKind` enum (DDIM / EulerA)
//!   - `timesteps()` — diffusers-compatible descending schedule with
//!                     `steps_offset=1`
//!   - `ddim_step()` — one deterministic ODE step
//!   - `euler_a_step()` — one Euler-Ancestral step with injected noise

use flame_core::{Result, Tensor};

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;

#[derive(Copy, Clone, Debug)]
pub enum Prediction {
    /// Default — model predicts noise ε. SDXL preset.
    Epsilon,
    /// Model predicts velocity v = √ᾱ·ε - √(1-ᾱ)·x₀. Some SDXL fine-tunes.
    V,
}

/// Sampler dispatch. Default (SDXL audit H4) is Euler-Ancestral, matching
/// upstream Python's `SampleConfig._get_model_defaults` for SDXL.
#[derive(Copy, Clone, Debug)]
pub enum SchedulerKind {
    /// Deterministic DDIM. Includes diffusers' `steps_offset=1`.
    Ddim,
    /// Euler-Ancestral (default). Injects noise per step for stochastic
    /// diversity at low step counts.
    EulerA,
}

/// Build the cumulative α product table. Length 1001 — `alpha_bar[0]`
/// corresponds to "no noise" (which we approximate as 1.0 because ᾱ_0 in
/// the discrete DDPM schedule is technically α_0 = 1 - β_0 ≈ 0.99915, but
/// DDIM's "previous timestep before step 0" semantically wants 1.0).
pub fn compute_alpha_bar() -> Vec<f32> {
    let mut alpha_bar = Vec::with_capacity(NUM_TRAIN_TIMESTEPS + 1);
    let sqrt_start = BETA_START.sqrt();
    let sqrt_end = BETA_END.sqrt();
    let mut cum = 1.0f32;
    for i in 0..=NUM_TRAIN_TIMESTEPS {
        let t = (i as f64) / (NUM_TRAIN_TIMESTEPS as f64 - 1.0);
        let sqrt_beta = sqrt_start + t.min(1.0) * (sqrt_end - sqrt_start);
        let beta = (sqrt_beta * sqrt_beta) as f32;
        cum *= 1.0 - beta;
        alpha_bar.push(cum);
    }
    alpha_bar
}

/// Build the descending discrete timestep schedule for DDIM/Euler-A with
/// `num_inference_steps` steps. SDXL audit H3: matches diffusers'
/// `EulerDiscreteScheduler` / `DDIMScheduler` with `steps_offset=1`, so the
/// final step is `t=1` (not `t=0` where ᾱ ≈ 0.99915 would corrupt the last
/// denoising step).
///
/// For 30 steps: `[951, 918, 884, ..., 51, 18, 1]`-style schedule.
pub fn timesteps(num_inference_steps: usize) -> Vec<usize> {
    let step_size = NUM_TRAIN_TIMESTEPS / num_inference_steps;
    (0..num_inference_steps)
        .map(|i| {
            // SDXL audit H3: `+1` is the diffusers `steps_offset=1`.
            (NUM_TRAIN_TIMESTEPS + 1).saturating_sub((i + 1) * step_size)
        })
        .collect()
}

/// One deterministic DDIM step.
///
///   pred_x0 = (x_t - √(1-ᾱ_t)·ε_pred) / √ᾱ_t           (eps-pred)
///           = √ᾱ_t · x_t - √(1-ᾱ_t) · v_pred           (v-pred)
///   x_prev  = √ᾱ_prev · pred_x0  +  √(1-ᾱ_prev) · ε_pred
///
/// `ab_t`    = ᾱ at the current step
/// `ab_prev` = ᾱ at the next (smaller-t) step. Use 1.0 for the final step.
pub fn ddim_step(
    x: &Tensor,
    pred: &Tensor,
    ab_t: f32,
    ab_prev: f32,
    prediction: Prediction,
) -> Result<Tensor> {
    match prediction {
        Prediction::Epsilon => {
            // pred = ε
            let pred_x0 = x.sub(&pred.mul_scalar((1.0 - ab_t).sqrt())?)?
                .mul_scalar(1.0 / ab_t.sqrt())?;
            let dir = pred.mul_scalar((1.0 - ab_prev).sqrt())?;
            pred_x0.mul_scalar(ab_prev.sqrt())?.add(&dir)
        }
        Prediction::V => {
            // pred = v ⇒ ε = √ᾱ_t · v + √(1-ᾱ_t) · x_t
            //         pred_x0 = √ᾱ_t · x_t - √(1-ᾱ_t) · v
            let pred_x0 = x.mul_scalar(ab_t.sqrt())?
                .sub(&pred.mul_scalar((1.0 - ab_t).sqrt())?)?;
            let eps = pred.mul_scalar(ab_t.sqrt())?
                .add(&x.mul_scalar((1.0 - ab_t).sqrt())?)?;
            let dir = eps.mul_scalar((1.0 - ab_prev).sqrt())?;
            pred_x0.mul_scalar(ab_prev.sqrt())?.add(&dir)
        }
    }
}

/// One Euler-Ancestral step. Operates in σ-space — input `x` is the
/// σ-scaled latent (caller manages the scaling), output is the next
/// σ-scaled latent. Injects fresh noise at each step (the "ancestral" part).
///
/// SDXL audit H4: this is the OT preset default scheduler (`EULER_A`).
///
/// σ²(t) = (1 - ᾱ_t) / ᾱ_t. Given ε_pred (or v_pred) and the σ-scaled latent:
///   pred_x0 = x / sqrt(1+σ²) - σ * ε_pred / sqrt(1+σ²)        (eps)
///           = √(1/(1+σ²)) * x · √ᾱ_t - σ²/(σ·√(1+σ²)) * v_pred (v simplified)
/// Ancestral split:
///   σ²_up   = σ²_prev * (σ² - σ²_prev) / σ²
///   σ²_down = σ²_prev - σ²_up
///   x_next  = pred_x0 + σ_down · d  +  σ_up · noise
///   d       = (x_in_sigma - pred_x0) / σ
pub fn euler_a_step(
    x: &Tensor,
    pred: &Tensor,
    ab_t: f32,
    ab_prev: f32,
    noise: &Tensor,
    prediction: Prediction,
) -> Result<Tensor> {
    let sigma_t = ((1.0 - ab_t) / ab_t).sqrt();
    let sigma_prev = if ab_prev >= 1.0 {
        0.0
    } else {
        ((1.0 - ab_prev) / ab_prev).sqrt()
    };

    // Reconstruct x in σ-space (ε-pred and v-pred branches). Note: caller
    // hands us a `pred` that was generated from the model fed
    // `model_input = x * sqrt(ᾱ_t)`, so ε_pred is in standard noise space.
    let pred_x0 = match prediction {
        Prediction::Epsilon => {
            // x_in_unit_var = x · sqrt(ᾱ_t)        (σ-scaled → unit-variance latent)
            // pred_x0       = (x_in_unit_var - sqrt(1-ᾱ_t)·ε) / sqrt(ᾱ_t)
            // Equivalent in σ-space: pred_x0 = x / (1+σ²)^0.5 ... but we go
            // through ᾱ to share the formula with DDIM.
            x.mul_scalar(ab_t.sqrt())?
                .sub(&pred.mul_scalar((1.0 - ab_t).sqrt())?)?
                .mul_scalar(1.0 / ab_t.sqrt())?
        }
        Prediction::V => {
            // unit_var = x · sqrt(ᾱ_t); pred_x0 = sqrt(ᾱ_t)·unit_var - sqrt(1-ᾱ_t)·v
            x.mul_scalar(ab_t)?  // sqrt(ᾱ) · sqrt(ᾱ) · x
                .sub(&pred.mul_scalar((1.0 - ab_t).sqrt())?)?
        }
    };

    // Ancestral noise split (diffusers `EulerAncestralDiscreteScheduler`):
    //   σ²_up = σ_prev² * (σ_t² - σ_prev²) / σ_t²
    //   σ_down² = σ_prev² - σ_up²
    let sigma_t_sq = sigma_t * sigma_t;
    let sigma_prev_sq = sigma_prev * sigma_prev;
    let sigma_up_sq = if sigma_t_sq > 0.0 {
        sigma_prev_sq * (sigma_t_sq - sigma_prev_sq) / sigma_t_sq
    } else { 0.0 };
    let sigma_up = sigma_up_sq.max(0.0).sqrt();
    let sigma_down = (sigma_prev_sq - sigma_up_sq).max(0.0).sqrt();

    // d = (x - pred_x0) / σ_t
    let dx = x.sub(&pred_x0)?;
    let d = if sigma_t > 1e-8 { dx.mul_scalar(1.0 / sigma_t)? } else { dx };

    // x_next = pred_x0 + σ_down · d + σ_up · noise
    let mut next = pred_x0.add(&d.mul_scalar(sigma_down)?)?;
    if sigma_up > 0.0 {
        next = next.add(&noise.mul_scalar(sigma_up)?)?;
    }
    Ok(next)
}

/// Compute the SDXL `add_time_ids` 6-vector for one sample:
///   `(orig_h, orig_w, crop_top, crop_left, target_h, target_w)`.
/// Caller is responsible for embedding it into the 256·6=1536-dim time
/// vector and concatenating with the CLIP-G pool to form `y` (2816-dim).
pub fn build_time_ids(orig_h: u32, orig_w: u32, crop_top: u32, crop_left: u32, target_h: u32, target_w: u32) -> [f32; 6] {
    [orig_h as f32, orig_w as f32, crop_top as f32, crop_left as f32, target_h as f32, target_w as f32]
}

/// Sinusoidal embedding of one `add_time_id` value at dim=256, identical
/// in shape/scale to the time embedding sinusoidal so the SDXL `label_emb`
/// MLP sees the right distribution. Returns 256 floats.
///
/// Matches sd-scripts `sdxl_train_util.get_size_embeddings` (frequency
/// embedding identical to LDM time embedding, dim=256, max_period=10000).
pub fn sin_embed_256(value: f32) -> Vec<f32> {
    const DIM: usize = 256;
    let half = DIM / 2;
    let mut data = vec![0.0f32; DIM];
    for j in 0..half {
        let freq = (-(10000.0f64.ln()) * (j as f64) / (half as f64)).exp() as f32;
        let angle = value * freq;
        data[j] = angle.cos();
        data[half + j] = angle.sin();
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_bar_table_size() {
        let ab = compute_alpha_bar();
        assert_eq!(ab.len(), NUM_TRAIN_TIMESTEPS + 1);
        // Monotonically decreasing
        for i in 1..ab.len() {
            assert!(ab[i] <= ab[i - 1]);
        }
        // ᾱ_999 should be small (deep noise)
        assert!(ab[999] < 0.01);
    }

    #[test]
    fn timestep_schedule_is_descending_with_steps_offset() {
        let ts = timesteps(20);
        assert_eq!(ts.len(), 20);
        for i in 1..ts.len() {
            assert!(ts[i] < ts[i - 1]);
        }
        // SDXL audit H3: with steps_offset=1, last step is t=1 (not t=0).
        assert_eq!(*ts.last().unwrap(), 1);
        // First step at 1001 - 50 = 951 for 20 steps.
        assert_eq!(ts[0], 951);
    }

    #[test]
    fn euler_a_30_step_schedule() {
        let ts = timesteps(30);
        assert_eq!(ts.len(), 30);
        // Final t should be > 0 (steps_offset=1).
        assert!(*ts.last().unwrap() > 0);
    }
}
