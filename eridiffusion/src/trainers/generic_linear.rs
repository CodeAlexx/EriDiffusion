use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::Parameter;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};

pub struct WeightLoaderTraining {
    device: Device,
    dtype: DType,
}
pub struct WeightLoaderTrainingPrefixed<'a> {
    inner: &'a WeightLoaderTraining,
    prefix: String,
}

// Generic Linear layer implementation that works with both Tensor and Parameter
// This enables gradient checkpointing without WeightLoader issues

// FLAME uses flame_core::device::Device instead of Device

/// Generic Linear layer that can hold either Tensor or Parameter
pub struct Linear<T> {
    pub weight: T,
    pub bias: Option<T>,
}

// Implementation for inference (T = Tensor)
impl Linear<Tensor> {
    pub fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        let w = self.weight.transpose_dims(0, 1)?;
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.add(bias),
        }
    }
}

// Implementation for training (T = Parameter)
impl Linear<Parameter> {
    pub fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        let w = self.weight.tensor()?.transpose_dims(0, 1)?;
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.add(&bias.tensor()?),
        }
    }
}

/// WeightLoader for training that returns Parameter instead of Tensor

impl WeightLoaderTraining {
    pub fn new(dtype: DType, device: Device) -> Self {
        Self { device, dtype }
    }

    pub fn get<S: Into<Shape>>(&self, shape: S, name: &str) -> flame_core::Result<Parameter> {
        // Create new trainable Parameter with proper initialization
        let shape = shape.into();
        let bound = 1.0 / (shape.dims().iter().product::<usize>() as f64).sqrt();

        // Create tensor first then wrap in Parameter
        let tensor = Tensor::randn(shape, 0.0, bound as f32, self.device.cuda_device().clone())?;
        let param = Parameter::new(tensor);

        // Note: In a real implementation, we'd want to track these params in our own map
        // For now, we'll just return the param
        Ok(param)
    }

    pub fn get_with_hint<S: Into<Shape>>(
        &self,
        shape: S,
        name: &str,
        hint: &str,
    ) -> flame_core::Result<Parameter> {
        let full_name =
            if hint.is_empty() { name.to_string() } else { format!("{}.{}", hint, name) };
        self.get(shape, &full_name)
    }

    pub fn pp(&self, prefix: &str) -> WeightLoaderTrainingPrefixed {
        WeightLoaderTrainingPrefixed { inner: self, prefix: prefix.to_string() }
    }
}

/// Prefixed version for hierarchical naming

impl<'a> WeightLoaderTrainingPrefixed<'a> {
    pub fn get<S: Into<Shape>>(&self, shape: S, name: &str) -> flame_core::Result<Parameter> {
        self.inner.get_with_hint(shape, name, &self.prefix)
    }

    pub fn pp(&self, suffix: &str) -> WeightLoaderTrainingPrefixed {
        WeightLoaderTrainingPrefixed {
            inner: self.inner,
            prefix: format!("{}.{}", self.prefix, suffix),
        }
    }
}

/// Helper to create Linear layers for training
pub fn linear_training_prefixed(
    in_dim: usize,
    out_dim: usize,
    vb: &WeightLoaderTraining,
) -> flame_core::Result<Linear<Parameter>> {
    let weight = vb.get(Shape::from_dims(&[out_dim, in_dim]), "weight")?;
    let bias = vb.get(Shape::from_dims(&[out_dim]), "bias").ok();

    Ok(Linear { weight, bias })
}

/// Helper to create Linear layers for inference
pub fn linear(
    in_dim: usize,
    out_dim: usize,
    vb: &crate::loaders::WeightLoader,
) -> flame_core::Result<Linear<Tensor>> {
    let weight = vb.tensor("weight", &[out_dim, in_dim])?;
    let bias = vb.tensor("bias", &[out_dim]).ok();

    Ok(Linear { weight, bias })
}

// Re-export the WeightLoader from gradient_checkpoint_generic
// pub use super::gradient_checkpoint_generic::WeightLoader;
