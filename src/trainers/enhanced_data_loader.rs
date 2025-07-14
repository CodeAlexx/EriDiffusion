//! Enhanced data loader with empty prompt file support
//! 
//! Based on bghira's feedback from SimpleTuner:
//! - Supports empty prompt files for proper dropout
//! - Implements duplicate concept balancing (10%/30% rule)
//! - Handles model-specific unconditional prompts correctly

use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::fs;
use std::collections::HashMap;
use rand::Rng;

/// Configuration for enhanced data loading
#[derive(Debug, Clone)]
pub struct EnhancedDataConfig {
    /// Path to empty prompt file (for dropout)
    pub empty_prompt_file: Option<PathBuf>,
    
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
}

#[derive(Debug, Clone)]
pub enum ModelType {
    SDXL,    // Uses zeros for unconditional
    Flux,    // Uses empty string for unconditional
    SD35,    // Uses empty string for unconditional
}

/// Enhanced caption handling with proper dropout
pub struct EnhancedCaptionHandler {
    config: EnhancedDataConfig,
    empty_prompt_content: Option<String>,
    concept_counts: HashMap<String, usize>,
    total_samples: usize,
}

impl EnhancedCaptionHandler {
    /// Create new caption handler
    pub fn new(config: EnhancedDataConfig) -> Result<Self> {
        // Load empty prompt file if specified
        let empty_prompt_content = if let Some(path) = &config.empty_prompt_file {
            if path.exists() {
                Some(fs::read_to_string(path)
                    .with_context(|| format!("Failed to read empty prompt file: {}", path.display()))?
                    .trim()
                    .to_string())
            } else {
                println!("Warning: Empty prompt file specified but not found: {}", path.display());
                None
            }
        } else {
            None
        };
        
        Ok(Self {
            config,
            empty_prompt_content,
            concept_counts: HashMap::new(),
            total_samples: 0,
        })
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
        self.config.caption_dropout_rate > 0.0 && 
        rng.gen::<f32>() < self.config.caption_dropout_rate
    }
    
    /// Check if concept should be balanced (10%/30% rule)
    fn should_balance_concept(&self, concept: &str, rng: &mut impl Rng) -> bool {
        if self.total_samples == 0 {
            return false;
        }
        
        let concept_ratio = self.concept_counts.get(concept)
            .map(|&count| count as f32 / self.total_samples as f32)
            .unwrap_or(0.0);
        
        // If concept appears in more than threshold (e.g., 10%) of samples
        if concept_ratio > self.config.duplicate_threshold {
            // Randomly drop to limit (e.g., 30%)
            let drop_probability = (concept_ratio - self.config.duplicate_limit) / 
                                   (concept_ratio - self.config.duplicate_threshold);
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
                // The text encoder will handle converting to zeros
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
            .map(|(concept, &count)| {
                (concept.clone(), count as f32 / self.total_samples as f32)
            })
            .collect()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    
    #[test]
    fn test_caption_dropout() {
        let config = EnhancedDataConfig {
            empty_prompt_file: None,
            caption_dropout_rate: 0.5,
            use_empty_prompt_for_dropout: false,
            duplicate_threshold: 0.1,
            duplicate_limit: 0.3,
            model_type: ModelType::Flux,
        };
        
        let mut handler = EnhancedCaptionHandler::new(config).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        
        let mut dropout_count = 0;
        for _ in 0..1000 {
            let caption = handler.process_caption("test caption", None, &mut rng);
            if caption.is_empty() {
                dropout_count += 1;
            }
        }
        
        // Should be roughly 50% dropout
        assert!((dropout_count as f32 / 1000.0 - 0.5).abs() < 0.05);
    }
    
    #[test]
    fn test_concept_balancing() {
        let config = EnhancedDataConfig {
            empty_prompt_file: None,
            caption_dropout_rate: 0.0,
            use_empty_prompt_for_dropout: false,
            duplicate_threshold: 0.1,
            duplicate_limit: 0.3,
            model_type: ModelType::SDXL,
        };
        
        let mut handler = EnhancedCaptionHandler::new(config).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        
        // Add many samples with same concept
        for i in 0..100 {
            if i < 40 {
                handler.process_caption("dog running", Some("dog"), &mut rng);
            } else {
                handler.process_caption("cat sleeping", Some("cat"), &mut rng);
            }
        }
        
        // Check concept stats
        let stats = handler.get_concept_stats();
        assert_eq!(stats.get("dog").copied().unwrap_or(0.0), 0.4);
        assert_eq!(stats.get("cat").copied().unwrap_or(0.0), 0.6);
        
        // Now process more dog captions - should start balancing
        let mut balanced_count = 0;
        for _ in 0..100 {
            let caption = handler.process_caption("dog playing", Some("dog"), &mut rng);
            if caption.is_empty() {
                balanced_count += 1;
            }
        }
        
        // Should have some balancing happening
        assert!(balanced_count > 0);
    }
}