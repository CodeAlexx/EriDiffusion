//! SDXL make_conditioning glue
//! Place at: crates/training/src/conditioning/make_conditioning.rs
//!
//! Two supported paths to build **cond_2816** used by LabelEmb (2816→1280→1280):
//!   A) **time_ids MLP path** (recommended for checkpoint parity):
//!       time_ids [N,6] --(TimeIdsMLP: 6→256)--> [N,256]
//!       time_proj_1536 = concat([pooled_1280], [time_ids_256])      // [N,1536]
//!       cond_2816       = concat([label_emb_input_1280], [time_proj_1536])
//!      Typically label_emb_input_1280 == pooled_1280, so pooled_1280 is used twice by design.
//!
//!   B) **sinusoidal path** (no learned time weights):
//!       time_proj_1536 = sinusoidal(timesteps_f32, 1536)
//!       cond_2816       = concat([label_emb_input_1280], [time_proj_1536])
//!
//! Return struct also exposes the intermediate `time_proj_1536` for debugging or alternate wiring.

use anyhow::{bail, Result};
use flame_core::{DType, Tensor};

use super::timestep_embedding::timestep_embedding;
use super::timestep_helpers::preprocess_timesteps;

pub struct SdxlCond {
    pub cond_2816: Tensor,      // to LabelEmb::forward_from_2816
    pub time_proj_1536: Tensor, // useful for debugging/alt routing
}

/// Path A: pooled 1280 + time_ids MLP 256 → time_proj_1536; concat with label_1280 → cond_2816.
pub fn make_conditioning_with_time_ids(
    label_emb_in_1280: &Tensor, // [N,1280] (usually pooled_1280)
    pooled_1280: &Tensor,       // [N,1280]
    time_ids_6: &Tensor,        // [N,6] (F32)
    time_ids_mlp_fwd: &dyn Fn(&Tensor) -> Result<Tensor>, // returns [N,256]
) -> Result<SdxlCond> {
    let label_shape = label_emb_in_1280.shape().dims();
    if label_shape.len() != 2 || label_shape[1] != 1280 {
        bail!(
            "make_conditioning_with_time_ids: label_emb_in must be [N,1280], got {:?}",
            label_shape
        );
    }
    let pooled_shape = pooled_1280.shape().dims();
    if pooled_shape.len() != 2 || pooled_shape[1] != 1280 {
        bail!(
            "make_conditioning_with_time_ids: pooled_1280 must be [N,1280], got {:?}",
            pooled_shape
        );
    }
    let tid_shape = time_ids_6.shape().dims();
    if tid_shape.len() != 2 || tid_shape[1] != 6 {
        bail!("make_conditioning_with_time_ids: time_ids must be [N,6], got {:?}", tid_shape);
    }
    let tid_256 = time_ids_mlp_fwd(time_ids_6)?; // [N,256]
    let time_proj_1536 = Tensor::cat(&[pooled_1280, &tid_256], 1)?; // [N,1536]
    let cond_2816 = Tensor::cat(&[label_emb_in_1280, &time_proj_1536], 1)?; // [N,2816]
    let cond_2816 = to_bf16_if_needed(&cond_2816)?;
    let time_proj_1536 = to_bf16_if_needed(&time_proj_1536)?;
    Ok(SdxlCond { cond_2816, time_proj_1536 })
}

/// Path B: pooled label + sinusoidal timesteps → cond_2816
pub fn make_conditioning_with_sinusoidal(
    label_emb_in_1280: &Tensor, // [N,1280]
    timesteps_any: &Tensor,     // [N] (I32/I64/F32)
    min_t: f32,
    max_t: f32,      // clamp
    max_period: f32, // e.g., 10000.0
) -> Result<SdxlCond> {
    let label_shape = label_emb_in_1280.shape().dims();
    if label_shape.len() != 2 || label_shape[1] != 1280 {
        bail!(
            "make_conditioning_with_sinusoidal: label_emb_in must be [N,1280], got {:?}",
            label_shape
        );
    }
    let t_f32 = preprocess_timesteps(timesteps_any, min_t, max_t)?;
    let time_proj_1536 = timestep_embedding(&t_f32, 1536, max_period, false, Some(1.0))?; // [N,1536]
    let time_proj_1536 = to_bf16_if_needed(&time_proj_1536)?;
    let cond_2816 = Tensor::cat(&[label_emb_in_1280, &time_proj_1536], 1)?;
    let cond_2816 = to_bf16_if_needed(&cond_2816)?;
    Ok(SdxlCond { cond_2816, time_proj_1536 })
}

fn to_bf16_if_needed(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == DType::BF16 {
        t.clone_result().map_err(Into::into)
    } else {
        t.to_dtype(DType::BF16).map_err(Into::into)
    }
}
