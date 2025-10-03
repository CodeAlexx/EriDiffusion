//! SDXL conditioning glue.
//!
//! Provides two entry points for building the classic **2816-D conditioning vector** used in SDXL:
//!
//! - [`sdxl_cond_2816_from_raw_ts`] — accepts raw timestep tensors (I32/I64 or F32, or even sigma values).
//!   Internally calls [`timestep_helpers::preprocess_timesteps`] to cast → F32 and
//!   clamp to a safe range before producing sinusoidal embeddings. Recommended for training and
//!   inference where timesteps may come in as integers or from a noise scheduler.
//!
//! - [`sdxl_cond_2816_from_sin`] — backwards-compatible path that assumes you already have a `[N]` F32
//!   tensor of timesteps in the correct range. Useful if you want to skip preprocessing.
//!
//! Both functions concatenate:
//!
//! * `[N,1280]` from `label_emb.0.{0,2}` (pooled text embedding path), with
//! * `[N,1536]` sinusoidal timestep embedding (Diffusers-style, `max_period = 10000.0` typical),
//!
//! producing `[N,2816]` for per-block AdaLN modulation.
//!
//! ## Example
//! ```rust
//! use eridiffusion_training::conditioning::sdxl_cond_2816_from_raw_ts;
//!
//! // raw timesteps can be I32, I64, or F32
//! # use flame_core::{Tensor, DType};
//! # use eridiffusion_core::Device;
//! # let device = Device::try_from("cuda:0").unwrap();
//! # let raw_ts = Tensor::from_vec_dtype(vec![0f32, 500.0, 1000.0], flame_core::Shape::from_dims(&[3]), device.cuda_device().clone(), DType::F32).unwrap();
//! # let label1280 = raw_ts.reshape(&[3, 1]).unwrap().repeat(&[1, 1280]).unwrap();
//! let cond2816 = sdxl_cond_2816_from_raw_ts(&label1280, &raw_ts, 0.0, 1000.0, 10000.0)?;
//! # assert_eq!(cond2816.shape().dims(), &[3, 2816]);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Ensure you `pub mod timestep_helpers;` in this crate so [`timestep_helpers::preprocess_timesteps`] is visible.

pub mod feature_toggle;
pub mod make_conditioning;
pub mod time_ids;
pub mod timestep_embedding;
pub mod timestep_helpers;

use eridiffusion_core::{Error, Result};
use flame_core::{DType, Tensor};

use self::timestep_embedding::timestep_embedding;
pub use feature_toggle::{make_conditioning, CondArgs};
pub use make_conditioning::{
    make_conditioning_with_sinusoidal, make_conditioning_with_time_ids, SdxlCond,
};

/// Build a 2816-D SDXL-style conditioning by concatenating label_emb_1280 with sinusoidal time 1536.
/// Accepts `raw_ts` which may be I32/I64 or F32; it will be cast & clamped to `[min_t, max_t]` before embedding.
pub fn sdxl_cond_2816_from_raw_ts(
    pooled_1280: &Tensor,
    raw_ts: &Tensor,
    min_t: f32,
    max_t: f32,
    max_period: f32,
) -> Result<Tensor> {
    let shape = pooled_1280.shape();
    if shape.dims().last().copied().unwrap_or(0) != 1280 {
        return Err(Error::InvalidInput(
            "sdxl_cond_2816_from_raw_ts: expected pooled text to be [N,1280]".into(),
        ));
    }
    let SdxlCond { cond_2816, .. } =
        make_conditioning_with_sinusoidal(pooled_1280, raw_ts, min_t, max_t, max_period)?;
    Ok(cond_2816)
}

/// Backwards-compatible helper for callers that already have clamped F32 timesteps.
pub fn sdxl_cond_2816_from_sin(
    pooled_1280: &Tensor,
    timesteps_f32: &Tensor,
    max_period: f32,
) -> Result<Tensor> {
    let dims = pooled_1280.shape().dims();
    if dims.last().copied().unwrap_or(0) != 1280 {
        return Err(Error::InvalidInput(
            "sdxl_cond_2816_from_sin: expected pooled text to be [N,1280]".into(),
        ));
    }
    if timesteps_f32.dtype() != DType::F32 {
        return Err(Error::InvalidInput(
            "sdxl_cond_2816_from_sin: timesteps tensor must be F32".into(),
        ));
    }
    let time_1536 = timestep_embedding(timesteps_f32, 1536, max_period, false, Some(1.0))?;
    Tensor::cat(&[pooled_1280, &time_1536], 1).map_err(Error::from)
}
