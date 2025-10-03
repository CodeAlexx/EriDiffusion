use flame_core::{DType, Device, Result, Shape, Tensor};
use std::collections::HashMap;

/// Bucket-aware CUDA alignment system for different aspect ratios and resolutions
pub struct BucketAlignmentManager {
    /// Cache of alignment configs for different bucket sizes
    pub alignment_cache: HashMap<(usize, usize), AlignmentStrategy>,
    /// Target dtype for alignment
    pub target_dtype: DType,
    /// CUDA alignment boundary (4MB)
    pub alignment_boundary: usize,
}

#[derive(Debug, Clone)]
pub struct AlignmentStrategy {
    pub aligned_height: usize,
    pub aligned_width: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub is_aligned: bool,
    pub memory_bytes: usize,
}

impl BucketAlignmentManager {
    const CUDA_ALIGNMENT_BYTES: usize = 4 * 1024 * 1024; // 4MB

    pub fn new(target_dtype: DType) -> Self {
        Self {
            alignment_cache: HashMap::new(),
            target_dtype,
            alignment_boundary: Self::CUDA_ALIGNMENT_BYTES,
        }
    }

    /// Get or compute alignment strategy for a bucket size
    pub fn get_alignment_strategy(&mut self, height: usize, width: usize) -> AlignmentStrategy {
        let key = (height, width);

        if let Some(strategy) = self.alignment_cache.get(&key) {
            return strategy.clone();
        }

        let strategy = self.compute_alignment_strategy(height, width);
        self.alignment_cache.insert(key, strategy.clone());
        strategy
    }

    /// Compute alignment strategy for specific dimensions
    fn compute_alignment_strategy(&self, height: usize, width: usize) -> AlignmentStrategy {
        let channels = 3; // RGB
        let dtype_size = match self.target_dtype {
            DType::F32 => 4,
            DType::BF16 => 2,
            DType::F16 => 2,
            _ => 4,
        };

        // Calculate current memory usage
        let current_elements = height * width * channels;
        let current_bytes = current_elements * dtype_size;

        // Check if already aligned
        if current_bytes % self.alignment_boundary == 0 {
            return AlignmentStrategy {
                aligned_height: height,
                aligned_width: width,
                padding_h: 0,
                padding_w: 0,
                is_aligned: true,
                memory_bytes: current_bytes,
            };
        }

        // Find aligned dimensions
        let (aligned_h, aligned_w) =
            self.find_aligned_dimensions(height, width, channels, dtype_size);
        let aligned_elements = aligned_h * aligned_w * channels;
        let aligned_bytes = aligned_elements * dtype_size;

        AlignmentStrategy {
            aligned_height: aligned_h,
            aligned_width: aligned_w,
            padding_h: aligned_h.saturating_sub(height),
            padding_w: aligned_w.saturating_sub(width),
            is_aligned: aligned_bytes % self.alignment_boundary == 0,
            memory_bytes: aligned_bytes,
        }
    }

    /// Find optimal aligned dimensions while preserving aspect ratio as much as possible
    fn find_aligned_dimensions(
        &self,
        h: usize,
        w: usize,
        channels: usize,
        dtype_size: usize,
    ) -> (usize, usize) {
        let current_bytes = h * w * channels * dtype_size;
        let target_bytes = ((current_bytes + self.alignment_boundary - 1)
            / self.alignment_boundary)
            * self.alignment_boundary;
        let target_elements = target_bytes / dtype_size;
        let target_spatial = target_elements / channels;

        // Strategy 1: Minimal padding - find smallest dimensions that work
        let candidates = self.generate_dimension_candidates(h, w, target_spatial);

        // Pick the candidate with minimal memory overhead and aspect ratio preservation
        let mut best_candidate = (h, w);
        let mut best_score = f64::INFINITY;
        let original_aspect = h as f64 / w as f64;

        for (cand_h, cand_w) in candidates {
            if cand_h < h || cand_w < w {
                continue; // Must be at least as large as original
            }

            let spatial_elements = cand_h * cand_w;
            if spatial_elements < target_spatial {
                continue; // Must meet minimum spatial requirements
            }

            // Score based on memory overhead and aspect ratio preservation
            let memory_overhead = (spatial_elements as f64) / (h * w) as f64;
            let aspect_diff = ((cand_h as f64 / cand_w as f64) - original_aspect).abs();
            let score = memory_overhead + aspect_diff * 0.1; // Weight aspect ratio preservation

            if score < best_score {
                best_score = score;
                best_candidate = (cand_h, cand_w);
            }
        }

        best_candidate
    }

    /// Generate candidate dimensions around the target
    fn generate_dimension_candidates(
        &self,
        h: usize,
        w: usize,
        target_spatial: usize,
    ) -> Vec<(usize, usize)> {
        let mut candidates = Vec::new();

        // Strategy 1: Proportional scaling
        let scale_factor = (target_spatial as f64 / (h * w) as f64).sqrt();
        let scaled_h = (h as f64 * scale_factor).ceil() as usize;
        let scaled_w = (w as f64 * scale_factor).ceil() as usize;
        candidates.push((scaled_h, scaled_w));

        // Strategy 2: Pad width more (common for landscape images)
        let pad_w = ((target_spatial + h - 1) / h).max(w);
        candidates.push((h, pad_w));

        // Strategy 3: Pad height more (common for portrait images)
        let pad_h = ((target_spatial + w - 1) / w).max(h);
        candidates.push((pad_h, w));

        // Strategy 4: Round up to multiples of 32/64 (CUDA-friendly)
        for multiple in [32, 64, 128] {
            let round_h = ((h + multiple - 1) / multiple) * multiple;
            let round_w = ((w + multiple - 1) / multiple) * multiple;
            if round_h * round_w >= target_spatial {
                candidates.push((round_h, round_w));
            }
        }

        // Strategy 5: Common bucket sizes with padding
        let common_sizes = [512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152];
        for &size_h in &common_sizes {
            for &size_w in &common_sizes {
                if size_h >= h && size_w >= w && size_h * size_w >= target_spatial {
                    candidates.push((size_h, size_w));
                }
            }
        }

        candidates.sort_by_key(|&(ch, cw)| ch * cw); // Sort by total size
        candidates.dedup();
        candidates
    }

    /// Apply alignment to a tensor based on computed strategy
    pub fn align_tensor(&mut self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 4 {
            return Err(flame_core::Error::InvalidOperation(
                "Expected 4D tensor [B, C, H, W]".to_string(),
            ));
        }

        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);

        if channels != 3 {
            println!("Warning: Expected 3 channels (RGB), got {}", channels);
        }

        let strategy = self.get_alignment_strategy(height, width);

        println!(
            "Bucket {}x{} -> {}x{} (padding: {}x{}, aligned: {}, memory: {:.2}MB)",
            height,
            width,
            strategy.aligned_height,
            strategy.aligned_width,
            strategy.padding_h,
            strategy.padding_w,
            strategy.is_aligned,
            strategy.memory_bytes as f64 / (1024.0 * 1024.0)
        );

        if strategy.padding_h == 0 && strategy.padding_w == 0 {
            // No padding needed
            return Ok(tensor.clone());
        }

        // Apply padding
        self.pad_tensor(tensor, &strategy)
    }

    /// Pad tensor according to alignment strategy
    fn pad_tensor(&self, tensor: &Tensor, strategy: &AlignmentStrategy) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);

        // Create padded tensor
        let padded_shape =
            Shape::from_dims(&[batch, channels, strategy.aligned_height, strategy.aligned_width]);
        let padded = Tensor::zeros(padded_shape, tensor.device().clone())?;

        // Copy original data to top-left corner (or center if preferred)
        let copy_ranges = vec![(0, batch), (0, channels), (0, height), (0, width)];

        let mut dst_slice = padded.slice(&copy_ranges)?;
        dst_slice.copy_(tensor)?;

        Ok(padded)
    }

    /// Precompute alignment strategies for common bucket sizes
    pub fn precompute_common_buckets(&mut self) -> Result<()> {
        // Common aspect ratios and their bucket sizes
        let common_buckets = vec![
            // Square
            (512, 512),
            (576, 576),
            (640, 640),
            (704, 704),
            (768, 768),
            (832, 832),
            (896, 896),
            (1024, 1024),
            // Landscape 4:3
            (512, 683),
            (576, 768),
            (640, 853),
            (704, 939),
            (768, 1024),
            // Landscape 16:9
            (512, 910),
            (576, 1024),
            (640, 1138),
            // Portrait 3:4
            (683, 512),
            (768, 576),
            (853, 640),
            (939, 704),
            (1024, 768),
            // Portrait 9:16
            (910, 512),
            (1024, 576),
            (1138, 640),
            // Wide landscape 21:9
            (512, 1195),
            (576, 1344),
            // Tall portrait 9:21
            (1195, 512),
            (1344, 576),
        ];

        println!(
            "Precomputing alignment strategies for {} common buckets...",
            common_buckets.len()
        );

        for (h, w) in common_buckets {
            let strategy = self.get_alignment_strategy(h, w);
            println!(
                "  {}x{} -> {}x{} ({:.1}% overhead)",
                h,
                w,
                strategy.aligned_height,
                strategy.aligned_width,
                ((strategy.aligned_height * strategy.aligned_width) as f64 / (h * w) as f64 - 1.0)
                    * 100.0
            );
        }

        Ok(())
    }

    /// Get alignment statistics for debugging
    pub fn get_alignment_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        let total_buckets = self.alignment_cache.len();
        let aligned_buckets = self.alignment_cache.values().filter(|s| s.is_aligned).count();

        stats.insert("total_buckets".to_string(), total_buckets as f64);
        stats.insert("aligned_buckets".to_string(), aligned_buckets as f64);
        stats.insert("alignment_rate".to_string(), aligned_buckets as f64 / total_buckets as f64);

        let avg_overhead: f64 =
            self.alignment_cache.values().map(|s| s.memory_bytes as f64).sum::<f64>()
                / total_buckets as f64;
        stats.insert("avg_memory_mb".to_string(), avg_overhead / (1024.0 * 1024.0));

        stats
    }
}

/// Integration with bucketed data loading
pub struct BucketAwareDataLoader {
    alignment_manager: BucketAlignmentManager,
    device: Device,
}

impl BucketAwareDataLoader {
    pub fn new(device: Device, target_dtype: DType) -> Self {
        let mut alignment_manager = BucketAlignmentManager::new(target_dtype);

        // Precompute common bucket alignments
        if let Err(e) = alignment_manager.precompute_common_buckets() {
            println!("Warning: Failed to precompute bucket alignments: {:?}", e);
        }

        Self { alignment_manager, device }
    }

    /// Load and align image for any bucket size
    pub fn load_aligned_image(
        &mut self,
        image_path: &std::path::Path,
        target_height: usize,
        target_width: usize,
    ) -> Result<Tensor> {
        // Load your image (existing code)
        let tensor = self.load_image_tensor(image_path, target_height, target_width)?;

        // Apply alignment
        let aligned = self.alignment_manager.align_tensor(&tensor)?;

        // Convert to target dtype
        let final_tensor = aligned.to_dtype(self.alignment_manager.target_dtype)?;

        Ok(final_tensor)
    }

    /// Placeholder for your existing image loading code
    fn load_image_tensor(
        &self,
        path: &std::path::Path,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        // TODO: Implement actual image loading
        return Err(flame_core::Error::InvalidOperation(format!(
            "Image loading not implemented for {:?}. Expected size {}x{}",
            path, height, width
        )));
    }

    /// Process a batch of different-sized images (common in bucketed training)
    pub fn process_mixed_batch(
        &mut self,
        images: Vec<(std::path::PathBuf, usize, usize)>,
    ) -> Result<Vec<Tensor>> {
        let mut aligned_images = Vec::new();

        for (path, height, width) in images {
            let aligned = self.load_aligned_image(&path, height, width)?;
            aligned_images.push(aligned);
        }

        Ok(aligned_images)
    }

    /// Get alignment statistics for monitoring
    pub fn print_alignment_stats(&self) {
        let stats = self.alignment_manager.get_alignment_stats();
        println!("=== Bucket Alignment Statistics ===");
        for (key, value) in stats {
            println!("  {}: {:.2}", key, value);
        }
    }
}

/// Example usage with different bucket sizes
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_alignment() {
        let mut manager = BucketAlignmentManager::new(DType::BF16);

        // Test various bucket sizes
        let test_buckets = vec![
            (512, 512),   // Square
            (1024, 1024), // Large square (problematic)
            (512, 768),   // 2:3 ratio
            (768, 512),   // 3:2 ratio
            (512, 1024),  // 1:2 ratio
            (1024, 512),  // 2:1 ratio
            (640, 640),   // Medium square
            (896, 896),   // Large square (better aligned)
        ];

        for (h, w) in test_buckets {
            let strategy = manager.get_alignment_strategy(h, w);
            println!(
                "{}x{}: {} -> {}x{} (aligned: {})",
                h,
                w,
                h * w * 3 * 2, // BF16 bytes
                strategy.aligned_height,
                strategy.aligned_width,
                strategy.is_aligned
            );
        }
    }
}
