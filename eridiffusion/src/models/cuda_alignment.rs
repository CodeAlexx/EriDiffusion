use flame_core::{DType, Device, Result, Shape, Tensor};

/// CUDA memory alignment utilities
pub struct CudaAlignmentUtils;

impl CudaAlignmentUtils {
    /// CUDA memory alignment boundary (4MB for optimal performance)
    const CUDA_ALIGNMENT_BYTES: usize = 4 * 1024 * 1024; // 4MB
    const CUDA_ALIGNMENT_ELEMENTS_F32: usize = Self::CUDA_ALIGNMENT_BYTES / 4; // 1M elements for F32
    const CUDA_ALIGNMENT_ELEMENTS_BF16: usize = Self::CUDA_ALIGNMENT_BYTES / 2; // 2M elements for BF16

    /// Check if tensor size is CUDA-aligned
    pub fn is_aligned(tensor: &Tensor, dtype: DType) -> bool {
        let total_elements = tensor.shape().dims().iter().product::<usize>();
        let alignment_elements = match dtype {
            DType::F32 => Self::CUDA_ALIGNMENT_ELEMENTS_F32,
            DType::BF16 => Self::CUDA_ALIGNMENT_ELEMENTS_BF16,
            _ => Self::CUDA_ALIGNMENT_ELEMENTS_F32, // Default to F32 alignment
        };

        (total_elements * Self::dtype_size(dtype)) % Self::CUDA_ALIGNMENT_BYTES == 0
    }

    /// Get the size in bytes for a dtype
    fn dtype_size(dtype: DType) -> usize {
        match dtype {
            DType::F32 => 4,
            DType::BF16 => 2,
            DType::F16 => 2,
            DType::U32 => 4,
            DType::I64 => 8,
            _ => 4, // Default to 4 bytes
        }
    }

    /// Pad tensor to meet CUDA alignment requirements
    pub fn align_tensor(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
        if Self::is_aligned(tensor, dtype) {
            return Ok(tensor.clone());
        }

        let dims = tensor.shape().dims();
        let total_elements = dims.iter().product::<usize>();
        let dtype_size = Self::dtype_size(dtype);
        let current_bytes = total_elements * dtype_size;

        // Calculate aligned size
        let aligned_bytes = ((current_bytes + Self::CUDA_ALIGNMENT_BYTES - 1)
            / Self::CUDA_ALIGNMENT_BYTES)
            * Self::CUDA_ALIGNMENT_BYTES;
        let aligned_elements = aligned_bytes / dtype_size;
        let padding_elements = aligned_elements - total_elements;

        if padding_elements == 0 {
            return Ok(tensor.clone());
        }

        println!(
            "Aligning tensor: {} bytes -> {} bytes (padding {} elements)",
            current_bytes, aligned_bytes, padding_elements
        );

        // Strategy 1: Pad the last dimension
        Self::pad_last_dimension(tensor, padding_elements)
    }

    /// Pad the last dimension to achieve alignment
    fn pad_last_dimension(tensor: &Tensor, padding_elements: usize) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        let last_dim = dims.len() - 1;
        let last_dim_size = dims[last_dim];

        // Calculate padding needed for last dimension
        let new_last_dim = last_dim_size + padding_elements;
        let mut new_dims = dims.to_vec();
        new_dims[last_dim] = new_last_dim;

        // Create padded tensor with zeros
        let padded = Tensor::zeros(Shape::from_dims(&new_dims), tensor.device().clone())?;

        // Copy original data to the beginning of padded tensor
        Self::copy_to_padded(tensor, &padded)
    }

    /// Copy original tensor data into padded tensor
    fn copy_to_padded(src: &Tensor, dst: &Tensor) -> Result<Tensor> {
        let src_dims = src.shape().dims();
        let dst_dims = dst.shape().dims();

        // Create slice indices for copying (copy only the original size)
        let mut slice_ranges = Vec::new();
        for (i, (&src_size, &_dst_size)) in src_dims.iter().zip(dst_dims.iter()).enumerate() {
            slice_ranges.push((0, src_size));
        }

        // Create a slice of the destination tensor and copy source data
        let mut dst_slice = dst.slice(&slice_ranges)?;
        dst_slice.copy_(src)?;

        Ok(dst.clone())
    }

    /// Alternative: Reshape to alignment-friendly dimensions
    pub fn reshape_for_alignment(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
        let dims = tensor.shape().dims();

        // For images, try to adjust height/width to be alignment-friendly
        if dims.len() == 4 {
            // [B, C, H, W]
            return Self::align_image_tensor(tensor, dtype);
        }

        // For other tensors, pad the last dimension
        Self::align_tensor(tensor, dtype)
    }

    /// Align image tensor by adjusting spatial dimensions
    fn align_image_tensor(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Calculate total elements needed for alignment
        let current_elements = b * c * h * w;
        let dtype_size = Self::dtype_size(dtype);
        let current_bytes = current_elements * dtype_size;
        let aligned_bytes = ((current_bytes + Self::CUDA_ALIGNMENT_BYTES - 1)
            / Self::CUDA_ALIGNMENT_BYTES)
            * Self::CUDA_ALIGNMENT_BYTES;
        let aligned_elements = aligned_bytes / dtype_size;

        if aligned_elements == current_elements {
            return Ok(tensor.clone());
        }

        // Try to distribute padding across height and width
        let spatial_elements = h * w;
        let needed_spatial = (aligned_elements + b * c - 1) / (b * c);

        // Find new height and width that give us the needed spatial elements
        let (new_h, new_w) = Self::find_aligned_dimensions(h, w, needed_spatial);

        println!(
            "Reshaping image tensor: {}x{}x{}x{} -> {}x{}x{}x{}",
            b, c, h, w, b, c, new_h, new_w
        );

        // Create padded tensor
        let padded =
            Tensor::zeros(Shape::from_dims(&[b, c, new_h, new_w]), tensor.device().clone())?;

        // Copy original data
        let slice_ranges = vec![(0, b), (0, c), (0, h), (0, w)];
        let mut dst_slice = padded.slice(&slice_ranges)?;
        dst_slice.copy_(tensor)?;

        Ok(padded)
    }

    /// Find aligned height and width dimensions
    fn find_aligned_dimensions(h: usize, w: usize, needed_elements: usize) -> (usize, usize) {
        // Try to keep aspect ratio similar while meeting alignment requirements
        let aspect_ratio = h as f32 / w as f32;

        // Find dimensions close to original aspect ratio
        let new_w = ((needed_elements as f32 / aspect_ratio).sqrt()).ceil() as usize;
        let new_h = (needed_elements + new_w - 1) / new_w;

        // Ensure dimensions are at least as large as original
        let new_h = new_h.max(h);
        let new_w = new_w.max(w);

        (new_h, new_w)
    }
}

/// Integration with VAE and other model components
pub trait AlignedTensorOps {
    /// Convert to aligned tensor with specific dtype
    fn to_aligned_dtype(&self, dtype: DType) -> Result<Tensor>;

    /// Ensure tensor is CUDA-aligned
    fn ensure_cuda_aligned(&self) -> Result<Tensor>;
}

impl AlignedTensorOps for Tensor {
    fn to_aligned_dtype(&self, dtype: DType) -> Result<Tensor> {
        let converted = self.to_dtype(dtype)?;
        CudaAlignmentUtils::align_tensor(&converted, dtype)
    }

    fn ensure_cuda_aligned(&self) -> Result<Tensor> {
        CudaAlignmentUtils::align_tensor(self, self.dtype())
    }
}

/// Alignment-aware weight loading for models
pub struct AlignedWeightLoader;

impl AlignedWeightLoader {
    /// Load and align weights for CUDA
    pub fn load_aligned_weights(
        weights: &std::collections::HashMap<String, Tensor>,
        target_dtype: DType,
    ) -> Result<std::collections::HashMap<String, Tensor>> {
        let mut aligned_weights = std::collections::HashMap::new();

        for (key, tensor) in weights {
            let aligned = tensor.to_aligned_dtype(target_dtype)?;
            aligned_weights.insert(key.clone(), aligned);
        }

        Ok(aligned_weights)
    }

    /// Check if all weights in a model are properly aligned
    pub fn validate_alignment(
        weights: &std::collections::HashMap<String, Tensor>,
        dtype: DType,
    ) -> Vec<String> {
        let mut unaligned_keys = Vec::new();

        for (key, tensor) in weights {
            if !CudaAlignmentUtils::is_aligned(tensor, dtype) {
                unaligned_keys.push(key.clone());
            }
        }

        unaligned_keys
    }
}

/// Practical usage examples
#[cfg(test)]
mod tests {
    use super::*;

    /// Test alignment for common image sizes
    #[test]
    fn test_image_alignment() {
        // 1024x1024x3 = 3,145,728 elements = 12,582,912 bytes (F32)
        // This is NOT 4MB aligned (4,194,304 bytes)

        let dims = vec![1, 3, 1024, 1024];
        let total_elements: usize = dims.iter().product();
        let bytes_f32 = total_elements * 4;
        let bytes_bf16 = total_elements * 2;

        println!("1024x1024x3 image:");
        println!("  F32: {} bytes", bytes_f32);
        println!("  BF16: {} bytes", bytes_bf16);
        println!("  4MB alignment: {} bytes", 4 * 1024 * 1024);
        println!("  F32 aligned: {}", bytes_f32 % (4 * 1024 * 1024) == 0);
        println!("  BF16 aligned: {}", bytes_bf16 % (4 * 1024 * 1024) == 0);
    }
}
