use anyhow::Context;
use flame_core::device::Device;
use flame_core::GradientMap;
use flame_core::Result;
use flame_core::{DType, Error, Shape, Tensor};
// backward is a method on Tensor, not a standalone function
use flame_core::optimizers::{Adam, SGD};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

// Simple data batch structure
pub struct DataBatch {
    pub pixel_values: Tensor,
    pub input_ids: Tensor,
}

// Simple enhanced data loader stub
pub struct EnhancedDataLoader {
    batch_size: usize,
    current_step: usize,
}

impl EnhancedDataLoader {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size, current_step: 0 }
    }

    pub fn next_batch(&mut self) -> Result<DataBatch> {
        // Stub implementation
        let device = Device::cuda(0)?;
        let pixel_values = Tensor::zeros(
            Shape::from_dims(&[self.batch_size, 3, 1024, 1024]),
            device.cuda_device().clone(),
        )?;
        let input_ids = Tensor::zeros_dtype(
            Shape::from_dims(&[self.batch_size, 77]),
            DType::I64,
            device.cuda_device().clone(),
        )?;
        self.current_step += 1;
        Ok(DataBatch { pixel_values, input_ids })
    }
}

// Enhanced data loader with empty prompt file support
//
// Based on bghira's feedback from SimpleTuner:
// - Supports empty prompt files for proper dropout
// - Implements duplicate concept balancing (10%/30% rule)
// - Handles model-specific unconditional prompts correctly

/// Configuration for enhanced data loading
#[derive(Debug, Clone)]
pub struct EnhancedDataConfig {
    /// Path to empty prompt file (for dropout)
    pub empty_prompt_file: Option<std::path::PathBuf>,

    /// Caption dropout rate (0.0 - 1.0)
    pub caption_dropout_rate: f32,

    /// Whether to use empty prompt file for dropout
    pub use_empty_prompt_for_dropout: bool,

    /// Duplicate balancing threshold (e.g., 0.1 for 10%)
    pub duplicate_threshold: f32,

    /// Duplicate balancing limit (e.g., 0.3 for 30%)
    pub duplicate_limit: f32,

    /// Model type for proper unconditional handling
    pub model_type: ModelType,

    /// Aspect ratio bucketing configuration
    pub aspect_ratio_buckets: AspectRatioBucketConfig,
}

#[derive(Debug, Clone)]
pub enum ModelType {
    SDXL, // Uses zeros for unconditional
    Flux, // Uses empty string for unconditional
    SD35, // Uses empty string for unconditional
}

/// Aspect ratio bucketing configuration
#[derive(Debug, Clone)]
pub struct AspectRatioBucketConfig {
    /// Whether bucketing is enabled
    pub enabled: bool,

    /// List of (width, height) buckets
    pub buckets: Vec<(usize, usize)>,

    /// Maximum aspect ratio allowed
    pub max_aspect_ratio: f32,

    /// Minimum aspect ratio allowed
    pub min_aspect_ratio: f32,
}

impl Default for AspectRatioBucketConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            buckets: vec![(1024, 1024)],
            max_aspect_ratio: 2.0,
            min_aspect_ratio: 0.5,
        }
    }
}

impl AspectRatioBucketConfig {
    /// Generate standard buckets for a given resolution
    pub fn generate_buckets(resolution: usize, step: usize) -> Vec<(usize, usize)> {
        let mut buckets = Vec::new();
        let min_size = resolution / 2;
        let max_size = resolution * 2;

        // Generate buckets with different aspect ratios
        for width in (min_size..=max_size).step_by(step) {
            for height in (min_size..=max_size).step_by(step) {
                let pixel_count = width * height;
                let target_count = resolution * resolution;

                // Keep buckets close to target pixel count
                if ((pixel_count as f32 - target_count as f32).abs() / target_count as f32) < 0.1 {
                    buckets.push((width, height));
                }
            }
        }

        // Always include square resolution
        if !buckets.contains(&(resolution, resolution)) {
            buckets.push((resolution, resolution));
        }

        buckets.sort();
        buckets.dedup();
        buckets
    }

    /// Find the best bucket for given image dimensions
    pub fn find_bucket(&self, width: usize, height: usize) -> (usize, usize) {
        if !self.enabled || self.buckets.is_empty() {
            return (self.buckets[0].0, self.buckets[0].1);
        }

        let aspect_ratio = width as f32 / height as f32;

        // Find bucket with closest aspect ratio and pixel count
        let mut best_bucket = self.buckets[0];
        let mut best_score = f32::MAX;

        for &(bw, bh) in &self.buckets {
            let bucket_ratio = bw as f32 / bh as f32;
            let ratio_diff = (aspect_ratio - bucket_ratio).abs();

            let pixel_diff = ((width * height) as f32 - (bw * bh) as f32).abs();
            let pixel_ratio = pixel_diff / (width * height) as f32;

            // Combined score: aspect ratio similarity + pixel count similarity
            let score = ratio_diff + pixel_ratio * 0.5;

            if score < best_score {
                best_score = score;
                best_bucket = (bw, bh);
            }
        }

        best_bucket
    }
}
/// Enhanced caption handling with proper dropout
pub struct EnhancedCaptionHandler {
    config: EnhancedDataConfig,
    empty_prompt_content: Option<String>,
    concept_counts: std::collections::HashMap<String, usize>,
    total_samples: usize,
}

impl EnhancedCaptionHandler {
    /// Create new caption handler
    pub fn new(config: EnhancedDataConfig) -> flame_core::Result<Self> {
        // Load empty prompt file if specified
        let empty_prompt_content = if let Some(path) = &config.empty_prompt_file {
            if path.exists() {
                Some(
                    fs::read_to_string(path)
                        .map_err(|e| {
                            flame_core::Error::InvalidOperation(format!(
                                "Failed to read empty prompt file: {}",
                                path.display()
                            ))
                        })?
                        .trim()
                        .to_string(),
                )
            } else {
                println!("Warning: Empty prompt file specified but not found: {}", path.display());
                None
            }
        } else {
            None
        };

        Ok(Self { config, empty_prompt_content, concept_counts: HashMap::new(), total_samples: 0 })
    }

    /// Process caption with proper dropout and model-specific handling
    pub fn process_caption(
        &mut self,
        caption: &str,
        concept: Option<&str>,
        rng: &mut impl Rng,
    ) -> String {
        // Update concept counts if provided
        if let Some(concept) = concept {
            *self.concept_counts.entry(concept.to_string()).or_insert(0) += 1;
            self.total_samples += 1;
        }

        // Check if we should apply dropout
        if self.should_apply_dropout(rng) {
            return self.get_unconditional_prompt();
        }

        // Check if this concept should be balanced
        if let Some(concept) = concept {
            if self.should_balance_concept(concept, rng) {
                return self.get_unconditional_prompt();
            }
        }

        // Return original caption
        caption.to_string()
    }

    /// Check if dropout should be applied
    fn should_apply_dropout(&self, rng: &mut impl Rng) -> bool {
        self.config.caption_dropout_rate > 0.0
            && rng.gen::<f32>() < self.config.caption_dropout_rate
    }

    /// Check if concept should be balanced (10%/30% rule)
    fn should_balance_concept(&self, concept: &str, rng: &mut impl Rng) -> bool {
        if self.total_samples == 0 {
            return false;
        }

        let concept_ratio = self
            .concept_counts
            .get(concept)
            .map(|&count| count as f32 / self.total_samples as f32)
            .unwrap_or(0.0);

        // If concept appears in more than threshold (e.g., 10%) of samples
        if concept_ratio > self.config.duplicate_threshold {
            // Randomly drop to limit (e.g., 30%)
            let drop_probability = (concept_ratio - self.config.duplicate_limit)
                / (concept_ratio - self.config.duplicate_threshold);
            return rng.gen::<f32>() < drop_probability;
        }

        false
    }

    /// Get unconditional prompt based on model type
    fn get_unconditional_prompt(&self) -> String {
        // First check if we have an empty prompt file
        if self.config.use_empty_prompt_for_dropout {
            if let Some(ref content) = self.empty_prompt_content {
                return content.clone();
            }
        }

        // Otherwise use model-specific defaults
        match self.config.model_type {
            ModelType::SDXL => {
                // SDXL uses zeros, but we return empty string here
                // The text encoder will handle converting to zeros;
                "".to_string()
            }
            ModelType::Flux | ModelType::SD35 => {
                // Flux and SD3.5 use empty strings
                "".to_string()
            }
        }
    }

    /// Get statistics about concept distribution
    pub fn get_concept_stats(&self) -> HashMap<String, f32> {
        if self.total_samples == 0 {
            return HashMap::new();
        }
        self.concept_counts
            .iter()
            .map(|(concept, &count)| (concept.clone(), count as f32 / self.total_samples as f32))
            .collect()
    }

    /// Helper to detect concepts in captions
    pub fn extract_concept_from_caption(caption: &str, trigger_words: &[String]) -> Option<String> {
        // Look for trigger words in caption
        for trigger in trigger_words {
            if caption.contains(trigger) {
                return Some(trigger.clone());
            }
        }

        // If no trigger words, try to extract main subject
        // This is a simple heuristic - can be improved
        if caption.contains("woman") || caption.contains("girl") {
            return Some("person".to_string());
        }
        if caption.contains("man") || caption.contains("boy") {
            return Some("person".to_string());
        }
        if caption.contains("dog") || caption.contains("cat") {
            return Some("animal".to_string());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caption_processing() {
        // Test placeholder
    }
}
