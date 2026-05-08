//! Caption dropout — randomly replace conditioning embeddings with the cached
//! unconditional embedding to teach the model classifier-free guidance.
//!
//! Phase: 1
//! Config flag: `caption_dropout_probability: f32` (default 0.0)
//!
//! Reference:
//!   - SimpleTuner `helpers/training/state_tracker.py` and
//!     `helpers/training/diffusion_model.py` (search `caption_dropout`)
//!   - OneTrainer `modules/modelLoader/StableDiffusion3ModelLoader.py`,
//!     `modules/modelSetup/BaseStableDiffusion3Setup.py` (search `dropout_probability`)
//!
//! Behavior:
//!   With probability `prob`, swap the conditional caption embedding for the
//!   cached unconditional ("") embedding. When `prob <= 0.0` (or the Bernoulli
//!   trial fails), the function returns `cond.clone()` — byte-identical to
//!   "do nothing".

use flame_core::{Result, Tensor};
use rand::rngs::StdRng;
use rand::Rng;

/// Drop caption with probability `prob`, returning the cached unconditional
/// embedding instead. When `prob <= 0.0`, returns `cond` unchanged with no
/// rng draw (so default-off does NOT consume rng state).
///
/// Caller is responsible for ensuring `uncond` has a shape compatible with
/// the model's expected caption-embedding shape (broadcast at the model
/// boundary, not here).
pub fn maybe_drop_caption(
    cond: &Tensor,
    uncond: &Tensor,
    prob: f32,
    rng: &mut StdRng,
) -> Result<Tensor> {
    if prob <= 0.0 {
        return Ok(cond.clone());
    }
    if rng.r#gen::<f32>() < prob {
        Ok(uncond.clone())
    } else {
        Ok(cond.clone())
    }
}

/// Legacy skeleton signature kept for any caller wired in Phase 0. Forwards
/// to [`maybe_drop_caption`] with the same semantics.
pub fn drop_caption(
    uncond: &Tensor,
    cond: &Tensor,
    prob: f32,
    rng: &mut StdRng,
) -> Result<Tensor> {
    maybe_drop_caption(cond, uncond, prob, rng)
}
