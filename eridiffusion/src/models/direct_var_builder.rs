use crate::loaders::{PrefixedWeightLoader, WeightLoader};
use anyhow::anyhow;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
use std::collections::HashMap;

// Direct WeightLoader that creates tensors on the correct device without copies

/// Create a WeightLoader directly from tensors without unnecessary copies

pub fn create_direct_var_builder(
    tensors: HashMap<String, Tensor>,
    dtype: DType,
    device: &Device,
) -> Result<WeightLoader> {
    Ok(WeightLoader { weights: tensors, device: device.clone() })
}
