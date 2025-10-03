//! SDXL time_ids constructor (no learned weights).
//! Produces the standard `[orig_h, orig_w, crop_y, crop_x, target_h, target_w]` vector.

use eridiffusion_core::{Device, Error, Result};
use eridiffusion_models::devtensor::{tensor_from_slice_on, F32_};
use flame_core::Shape;

/// Build a batched time_ids tensor by repeating the provided values for `batch` samples.
pub fn build_time_ids(
    batch: usize,
    orig_h: f32,
    orig_w: f32,
    crop_y: f32,
    crop_x: f32,
    target_h: f32,
    target_w: f32,
    device: &Device,
) -> Result<flame_core::Tensor> {
    let vals = [orig_h, orig_w, crop_y, crop_x, target_h, target_w];
    let base = tensor_from_slice_on(&vals, Shape::from_dims(&[1, 6]), device, F32_)?;
    base.repeat(&[batch, 1]).map_err(Error::from)
}
