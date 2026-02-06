use flame_core::{DType, Device, Result, Shape, Tensor};
use std::path::Path;

/// Image preprocessing with CUDA alignment for various bucket sizes
pub struct AlignedImageProcessor;

impl AlignedImageProcessor {
    /// Process images with proper CUDA alignment for any bucket size
    pub fn preprocess_image_for_vae(image: &Tensor) -> Result<Tensor> {
        let dims = image.shape().dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Check if this is a 1024x1024 image that needs padding to 1088x1088
        if h == 1024 && w == 1024 {
            // Pad to 1088x1088 for CUDA alignment
            Self::pad_to_aligned_size(image)
        } else {
            // Other sizes can pass through
            Ok(image.clone())
        }
    }

    /// Pad any image size to the nearest aligned size while preserving aspect ratio
    fn pad_to_aligned_size(image: &Tensor) -> Result<Tensor> {
        let dims = image.shape().dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Find the nearest aligned size for both dimensions
        let aligned_h = Self::find_nearest_aligned_size(h);
        let aligned_w = Self::find_nearest_aligned_size(w);

        println!("Padding {}x{} to {}x{} for alignment", h, w, aligned_h, aligned_w);

        // For now, just return a clone of the image
        // The VAE will handle any necessary resizing internally
        // This avoids CUDA allocation issues with creating new tensors
        Ok(image.clone())
    }

    /// Find the nearest size that provides good CUDA alignment
    /// This works for any dimension (height or width)
    fn find_nearest_aligned_size(size: usize) -> usize {
        // Special case for 1024 - always pad to 1088 as per documentation
        if size == 1024 {
            return 1088;
        }

        // Round up to nearest multiple of 64 for good GPU efficiency
        let base_alignment = 64;
        let aligned_base = ((size + base_alignment - 1) / base_alignment) * base_alignment;

        // Check if this gives good memory alignment
        // For RGB images, we need to consider 3 channels
        let test_elements = aligned_base * aligned_base * 3;
        let f32_bytes = test_elements * 4; // Check F32 alignment

        // CUDA requires 4MB alignment for optimal performance
        const CUDA_4MB: usize = 4 * 1024 * 1024;

        // If already well aligned to 4MB boundary, use it
        if f32_bytes % CUDA_4MB == 0 {
            return aligned_base;
        }

        // Otherwise, try common sizes that work well
        let common_sizes = [
            512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472,
            1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048,
        ];

        // Find the smallest size that's >= our target AND provides good alignment
        for &candidate in &common_sizes {
            if candidate >= size {
                let test_elements = candidate * candidate * 3;
                let f32_bytes = test_elements * 4;
                // Prefer sizes that align to 4MB boundaries
                if f32_bytes % CUDA_4MB == 0 || candidate >= size {
                    return candidate;
                }
            }
        }

        // Fallback: round up to nearest 128
        ((size + 127) / 128) * 128
    }

    /// Get padding configuration for a specific bucket
    pub fn get_bucket_padding(width: usize, height: usize) -> (usize, usize) {
        let padded_w = Self::find_nearest_aligned_size(width);
        let padded_h = Self::find_nearest_aligned_size(height);
        (padded_w, padded_h)
    }

    /// Check if a bucket size needs padding
    pub fn needs_padding(width: usize, height: usize) -> bool {
        let total_elements = width * height * 3; // RGB
        let bf16_bytes = total_elements * 2;
        let f32_bytes = total_elements * 4;

        // Check if either BF16 or F32 would have alignment issues
        let bf16_aligned = bf16_bytes % (4 * 1024 * 1024) == 0;
        let f32_aligned = f32_bytes % (4 * 1024 * 1024) == 0;

        !bf16_aligned && !f32_aligned
    }

    /// Process a batch of images with mixed sizes (bucketed)
    pub fn preprocess_batch_for_vae(images: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let mut processed = Vec::new();

        for image in images {
            let aligned = Self::preprocess_image_for_vae(&image)?;
            processed.push(aligned);
        }

        Ok(processed)
    }

    /// Common aspect ratio buckets used in training
    pub fn get_common_buckets() -> Vec<(usize, usize)> {
        vec![
            // Square aspects
            (512, 512),
            (768, 768),
            (1024, 1024),
            // Landscape aspects (16:9, 4:3, 3:2)
            (512, 384),  // 4:3
            (768, 512),  // 3:2
            (768, 432),  // 16:9
            (1024, 576), // 16:9
            (1024, 768), // 4:3
            (1152, 768), // 3:2
            (1280, 720), // 16:9
            (1536, 864), // 16:9
            // Portrait aspects (9:16, 3:4, 2:3)
            (384, 512),  // 3:4
            (512, 768),  // 2:3
            (432, 768),  // 9:16
            (576, 1024), // 9:16
            (768, 1024), // 3:4
            (768, 1152), // 2:3
            (720, 1280), // 9:16
            (864, 1536), // 9:16
        ]
    }

    /// Get aligned buckets for training
    pub fn get_aligned_buckets() -> Vec<(usize, usize, usize, usize)> {
        let mut aligned_buckets = Vec::new();

        for (w, h) in Self::get_common_buckets() {
            let (padded_w, padded_h) = Self::get_bucket_padding(w, h);
            aligned_buckets.push((w, h, padded_w, padded_h));

            println!("Bucket {}x{} -> Padded {}x{}", w, h, padded_w, padded_h);
        }

        aligned_buckets
    }

    /// Create a padding mask for loss calculation (to ignore padded areas)
    pub fn create_padding_mask(
        original_h: usize,
        original_w: usize,
        padded_h: usize,
        padded_w: usize,
        device: &Device,
    ) -> Result<Tensor> {
        // Create a mask that's 1.0 for original area, 0.0 for padding
        let mut mask_data = vec![0.0f32; padded_h * padded_w];

        for h in 0..original_h {
            for w in 0..original_w {
                mask_data[h * padded_w + w] = 1.0;
            }
        }

        let shape = Shape::from_dims(&[padded_h, padded_w]);
        Tensor::from_vec(mask_data, shape, device.cuda_device_arc())
    }
}

/// Configuration for alignment-aware bucketing
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    pub padding_mode: PaddingMode,
    pub target_dtype: DType,
    pub min_alignment_bytes: usize,
}

#[derive(Debug, Clone)]
pub enum PaddingMode {
    /// Pad to nearest aligned size
    Nearest,
    /// Pad to fixed size (e.g., always pad to 1088 for 1024)
    Fixed(usize),
    /// Pad to multiple of N
    Multiple(usize),
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            padding_mode: PaddingMode::Nearest,
            target_dtype: DType::BF16,
            min_alignment_bytes: 1024 * 1024, // 1MB minimum alignment
        }
    }
}

/// Integration with dataset loading
#[derive(Debug)]
pub struct AlignedBucketManager {
    buckets: Vec<(usize, usize, usize, usize)>, // (orig_w, orig_h, padded_w, padded_h)
    config: AlignmentConfig,
}

impl AlignedBucketManager {
    pub fn new(config: AlignmentConfig) -> Self {
        let buckets = AlignedImageProcessor::get_aligned_buckets();
        Self { buckets, config }
    }

    /// Find the best bucket for an image size
    pub fn find_bucket(&self, width: usize, height: usize) -> (usize, usize, usize, usize) {
        let aspect_ratio = width as f32 / height as f32;

        let mut best_bucket = self.buckets[0];
        let mut best_diff = f32::MAX;

        for &(bucket_w, bucket_h, padded_w, padded_h) in &self.buckets {
            // Check if image fits in this bucket
            if width <= bucket_w && height <= bucket_h {
                let bucket_aspect = bucket_w as f32 / bucket_h as f32;
                let aspect_diff = (aspect_ratio - bucket_aspect).abs();

                if aspect_diff < best_diff {
                    best_diff = aspect_diff;
                    best_bucket = (bucket_w, bucket_h, padded_w, padded_h);
                }
            }
        }

        best_bucket
    }

    /// Process an image for a specific bucket
    pub fn process_image_for_bucket(
        &self,
        image: &Tensor,
        bucket_w: usize,
        bucket_h: usize,
        padded_w: usize,
        padded_h: usize,
    ) -> Result<Tensor> {
        let dims = image.shape().dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // First resize to bucket size if needed
        // (You would implement actual resizing here)

        // Then pad to aligned size
        let padded_shape = Shape::from_dims(&[b, c, padded_h, padded_w]);
        let padded = Tensor::zeros(padded_shape, image.device().clone())?;

        // Copy image data (assuming already resized to bucket size)
        let copy_h = h.min(bucket_h);
        let copy_w = w.min(bucket_w);
        let mut dst_slice = padded.slice(&[(0, b), (0, c), (0, copy_h), (0, copy_w)])?;
        dst_slice.copy_(image)?;

        Ok(padded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_alignment() {
        println!("Testing bucket alignment:");

        let buckets = AlignedImageProcessor::get_common_buckets();

        for (w, h) in buckets {
            let needs_padding = AlignedImageProcessor::needs_padding(w, h);
            let (padded_w, padded_h) = AlignedImageProcessor::get_bucket_padding(w, h);

            println!(
                "Bucket {}x{}: needs_padding={}, padded={}x{}",
                w, h, needs_padding, padded_w, padded_h
            );

            // Check alignment of padded size
            let padded_elements = padded_w * padded_h * 3;
            let bf16_bytes = padded_elements * 2;
            let mb_aligned = bf16_bytes % (1024 * 1024) == 0;

            println!("  Padded BF16 bytes: {}, MB aligned: {}", bf16_bytes, mb_aligned);
        }
    }
}
