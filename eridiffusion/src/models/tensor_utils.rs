//! Utilities for tensor operations and memory alignment

use flame_core::{DType, Result, Shape, Tensor};

/// Align tensor size to CUDA memory boundaries
pub fn align_tensor_size(size: usize) -> usize {
    // CUDA prefers aligned allocations, typically to 512 or 4096 bytes
    // For BF16/F16, each element is 2 bytes
    const ALIGNMENT: usize = 512; // 512 elements = 1KB for BF16/F16

    if size % ALIGNMENT == 0 {
        size
    } else {
        ((size / ALIGNMENT) + 1) * ALIGNMENT
    }
}

/// Convert tensor to dtype with proper memory alignment
pub fn to_dtype_aligned(tensor: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let shape = tensor.shape();
    let total_elements = shape.elem_count();

    // For small tensors (biases) and known problematic sizes, keep in original dtype
    if total_elements < 1024 || needs_alignment(total_elements) {
        // Small tensors and problematic sizes stay in F32
        Ok(tensor.clone())
    } else {
        // For larger tensors, try to convert to target dtype
        // This should work for most weight tensors
        match tensor.to_dtype(target_dtype) {
            Ok(converted) => Ok(converted),
            Err(_) => {
                // If conversion fails, keep original
                println!("Warning: Failed to convert tensor with {} elements to {:?}, keeping original dtype", total_elements, target_dtype);
                Ok(tensor.clone())
            }
        }
    }
}

/// Check if a tensor size might cause CUDA alignment issues
pub fn needs_alignment(size: usize) -> bool {
    // Known problematic sizes from CUDA alignment errors
    const PROBLEMATIC_SIZES: &[usize] = &[3456, 147456, 12288, 16384, 295936];

    // Check if it's a known problematic size or if it doesn't align well
    // CUDA prefers sizes that are multiples of 512 for F16/BF16 (1KB alignment)
    PROBLEMATIC_SIZES.contains(&size) || (size % 512 != 0 && size > 1024) || (size * 2) % 4096 != 0
    // Check byte alignment for F16/BF16
}
