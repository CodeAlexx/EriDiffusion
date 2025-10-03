//! Vision utilities for FLAME

use flame_core::device::Device;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::sync::Arc;

/// Extension trait for Tensor to add missing methods
trait TensorExt {
    fn to_vec(&self) -> flame_core::Result<Vec<f32>>;
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn to_vec(&self) -> flame_core::Result<Vec<f32>> {
        // Convert tensor to Vec<f32>
        // This is a placeholder - actual implementation would depend on FLAME internals
        Ok(vec![0.0; self.shape().dims().iter().product::<usize>()])
    }

    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor> {
        // Sum along dimension
        self.sum_dim(dim)
    }

    fn square(&self) -> flame_core::Result<Tensor> {
        // Element-wise square
        self.mul(self)
    }
}

pub fn resize(tensor: &Tensor, new_size: &[usize]) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let (_, height, width) = (dims[0], dims[1], dims[2]);
    let new_height = new_size[0];
    let new_width = new_size[1];

    // Calculate scaling factors
    let scale_h = height as f32 / new_height as f32;
    let scale_w = width as f32 / new_width as f32;

    // Create output tensor
    let channels = tensor.shape().dims()[0];
    let mut output = Tensor::zeros(
        Shape::from_dims(&[channels, new_height, new_width]),
        tensor.device().clone(),
    )?;

    // Bilinear interpolation
    for y in 0..new_height {
        for x in 0..new_width {
            let src_y = y as f32 * scale_h;
            let src_x = x as f32 * scale_w;

            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(height - 1);
            let x1 = (x0 + 1).min(width - 1);

            let dy = src_y - y0 as f32;
            let dx = src_x - x0 as f32;

            // Bilinear weights
            let w00 = (1.0 - dx) * (1.0 - dy);
            let w01 = dx * (1.0 - dy);
            let w10 = (1.0 - dx) * dy;
            let w11 = dx * dy;

            // Interpolate
            let channels = tensor.shape().dims()[0];
            for c in 0..channels {
                // TODO: FLAME doesn't have a get() method for individual tensor elements
                // This would need a proper implementation with slice operations
                let v00 = 0.0; // tensor.get(&[c, y0, x0])?.item()?;
                let v01 = 0.0; // tensor.get(&[c, y0, x1])?.item()?;
                let v10 = 0.0; // tensor.get(&[c, y1, x0])?.item()?;
                let v11 = 0.0; // tensor.get(&[c, y1, x1])?.item()?;

                let value = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
                // TODO: FLAME doesn't have a set() method for individual tensor elements
                // output.set(&[c, y, x], value)?;
            }
        }
    }

    Ok(output)
}

/// Horizontal flip
pub fn horizontal_flip(tensor: &Tensor, device: &CudaDevice) -> Result<Tensor> {
    let shape = tensor.shape();
    let width = shape.dims()[2];

    // Create indices for flipping
    let indices: Vec<f32> = (0..width).rev().map(|i| i as f32).collect();
    let indices = Tensor::from_vec(indices, Shape::from_dims(&[width]), tensor.device().clone())?;

    // Gather along width dimension
    tensor.index_select(2, &indices)
}

/// Random crop
pub fn random_crop(tensor: &Tensor, crop_size: usize) -> Result<Tensor> {
    let shape = tensor.shape();
    let height = shape.dims()[1];
    let width = shape.dims()[2];

    if height < crop_size || width < crop_size {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Image too small for crop: {}x{} < {}",
            height, width, crop_size
        )));
    }

    let y = rand::random::<usize>() % (height - crop_size);
    let x = rand::random::<usize>() % (width - crop_size);

    tensor.slice(&[
        (0, tensor.shape().dims()[0]), // Keep all channels
        (y, y + crop_size),
        (x, x + crop_size),
    ])
}

/// Center crop
pub fn center_crop(tensor: &Tensor, crop_size: usize) -> Result<Tensor> {
    let shape = tensor.shape();
    let height = shape.dims()[1];
    let width = shape.dims()[2];

    let y = (height - crop_size) / 2;
    let x = (width - crop_size) / 2;

    tensor.slice(&[
        (0, tensor.shape().dims()[0]), // Keep all channels
        (y, y + crop_size),
        (x, x + crop_size),
    ])
}

/// Normalize tensor with mean and std
pub fn normalize(
    tensor: &Tensor,
    mean: &[f32],
    std: &[f32],
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    // Convert mean and std to tensors
    let mean =
        Tensor::from_vec(mean.to_vec(), Shape::from_dims(&[mean.len(), 1, 1]), device.clone())?;
    let std = Tensor::from_vec(std.to_vec(), Shape::from_dims(&[std.len(), 1, 1]), device.clone())?;

    // Normalize: (x - mean) / std
    tensor.sub(&mean)?.div(&std)
}

/// Random rotation
pub fn random_rotation(tensor: &Tensor, max_angle: f32) -> flame_core::Result<Tensor> {
    let angle = (rand::random::<f32>() - 0.5) * 2.0 * max_angle;
    rotate(tensor, angle)
}

/// Rotate tensor by angle (in radians)
pub fn rotate(tensor: &Tensor, angle: f32) -> flame_core::Result<Tensor> {
    // Simplified rotation - would need proper affine transformation
    // For now, just return a clone of the original
    Ok(tensor.clone())
}

/// Color jitter
pub fn color_jitter(
    tensor: &Tensor,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
) -> flame_core::Result<Tensor> {
    let mut result = tensor.clone();

    // Brightness
    if brightness > 0.0 {
        let factor = 1.0 + (rand::random::<f32>() - 0.5) * 2.0 * brightness;
        result = result.mul_scalar(factor)?;
    }

    // Contrast
    if contrast > 0.0 {
        let factor = 1.0 + (rand::random::<f32>() - 0.5) * 2.0 * contrast;
        let mean = result.mean()?;
        result = result.sub(&mean)?.mul_scalar(factor)?.add(&mean)?;
    }

    // Clamp to valid range
    result.clamp(-1.0, 1.0)
}
