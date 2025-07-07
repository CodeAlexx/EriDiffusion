//! Caption preprocessing and tokenization for different models

use eridiffusion_core::{Result, Error, ModelArchitecture};
use std::collections::{HashMap, HashSet};
use regex::Regex;
use tracing::{debug, warn};

/// Caption preprocessor for different architectures
pub struct CaptionPreprocessor {
    architecture: ModelArchitecture,
    config: CaptionConfig,
    tag_processor: Option<TagProcessor>,
    token_replacements: HashMap<String, String>,
}

impl CaptionPreprocessor {
    /// Create new caption preprocessor
    pub fn new(architecture: ModelArchitecture) -> Result<Self> {
        let config = CaptionConfig::for_architecture(&architecture);
        let tag_processor = if config.process_tags {
            Some(TagProcessor::new())
        } else {
            None
        };
        
        let token_replacements = Self::create_token_replacements();
        
        Ok(Self {
            architecture,
            config,
            tag_processor,
            token_replacements,
        })
    }
    
    /// Preprocess single caption
    pub fn preprocess(&self, caption: &str) -> Result<ProcessedCaption> {
        let mut caption = caption.to_string();
        
        // Basic cleaning
        caption = self.clean_caption(&caption);
        
        // Process tags if enabled
        if let Some(tag_processor) = &self.tag_processor {
            caption = tag_processor.process(&caption)?;
        }
        
        // Apply token replacements
        caption = self.apply_replacements(&caption);
        
        // Handle special tokens
        caption = self.handle_special_tokens(&caption);
        
        // Truncate if needed
        let (truncated_caption, was_truncated) = self.truncate_caption(&caption);
        
        // Split into chunks for long caption support
        let chunks = if self.config.supports_long_captions {
            self.split_into_chunks(&truncated_caption)
        } else {
            vec![truncated_caption.clone()]
        };
        
        Ok(ProcessedCaption {
            text: truncated_caption.clone(),
            chunks,
            was_truncated,
            token_count: self.estimate_token_count(&truncated_caption),
        })
    }
    
    /// Preprocess batch of captions
    pub fn preprocess_batch(&self, captions: &[String]) -> Result<Vec<ProcessedCaption>> {
        captions.iter()
            .map(|c| self.preprocess(c))
            .collect()
    }
    
    /// Clean caption
    fn clean_caption(&self, caption: &str) -> String {
        let mut caption = caption.trim().to_string();
        
        // Remove extra whitespace
        let whitespace_re = Regex::new(r"\s+").unwrap();
        caption = whitespace_re.replace_all(&caption, " ").to_string();
        
        // Fix common issues
        caption = caption.replace(" ,", ",");
        caption = caption.replace(" .", ".");
        caption = caption.replace(" !", "!");
        caption = caption.replace(" ?", "?");
        caption = caption.replace(" :", ":");
        caption = caption.replace(" ;", ";");
        
        // Remove duplicate punctuation
        let dup_punct_re = Regex::new(r"([,\.!?:;])\1+").unwrap();
        caption = dup_punct_re.replace_all(&caption, "$1").to_string();
        
        caption
    }
    
    /// Apply token replacements
    fn apply_replacements(&self, caption: &str) -> String {
        let mut caption = caption.to_string();
        
        for (from, to) in &self.token_replacements {
            caption = caption.replace(from, to);
        }
        
        caption
    }
    
    /// Handle special tokens
    fn handle_special_tokens(&self, caption: &str) -> String {
        match self.architecture {
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
                // SD3 uses T5 which has specific token handling
                let mut caption = caption.to_string();
                
                // Ensure proper spacing around special tokens
                caption = caption.replace("<", " <");
                caption = caption.replace(">", "> ");
                caption = caption.replace("  ", " ");
                
                caption.trim().to_string()
            }
            ModelArchitecture::Flux => {
                // Flux may have specific requirements
                caption.to_string()
            }
            _ => caption.to_string(),
        }
    }
    
    /// Truncate caption to max tokens
    fn truncate_caption(&self, caption: &str) -> (String, bool) {
        let estimated_tokens = self.estimate_token_count(caption);
        
        if estimated_tokens <= self.config.max_tokens {
            return (caption.to_string(), false);
        }
        
        // Simple word-based truncation
        let words: Vec<&str> = caption.split_whitespace().collect();
        let target_words = (self.config.max_tokens as f32 * 0.75) as usize;
        
        if words.len() <= target_words {
            return (caption.to_string(), false);
        }
        
        let truncated = words[..target_words].join(" ");
        
        // Add ellipsis if configured
        let truncated = if self.config.add_ellipsis_on_truncate {
            format!("{}...", truncated)
        } else {
            truncated
        };
        
        (truncated, true)
    }
    
    /// Split caption into chunks for long caption support
    fn split_into_chunks(&self, caption: &str) -> Vec<String> {
        if !self.config.supports_long_captions {
            return vec![caption.to_string()];
        }
        
        let words: Vec<&str> = caption.split_whitespace().collect();
        let chunk_size = (self.config.chunk_size as f32 * 0.75) as usize;
        
        if words.len() <= chunk_size {
            return vec![caption.to_string()];
        }
        
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        
        for word in words {
            current_chunk.push(word);
            
            if current_chunk.len() >= chunk_size {
                chunks.push(current_chunk.join(" "));
                current_chunk.clear();
            }
        }
        
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.join(" "));
        }
        
        chunks
    }
    
    /// Estimate token count
    fn estimate_token_count(&self, caption: &str) -> usize {
        // Simple estimation: words * 1.3
        let word_count = caption.split_whitespace().count();
        (word_count as f32 * 1.3) as usize
    }
    
    /// Create common token replacements
    fn create_token_replacements() -> HashMap<String, String> {
        let mut replacements = HashMap::new();
        
        // Common replacements for better tokenization
        replacements.insert("e.g.".to_string(), "for example".to_string());
        replacements.insert("i.e.".to_string(), "that is".to_string());
        replacements.insert("etc.".to_string(), "and so on".to_string());
        replacements.insert("vs.".to_string(), "versus".to_string());
        
        replacements
    }
}

/// Caption configuration for different architectures
#[derive(Debug, Clone)]
pub struct CaptionConfig {
    /// Maximum number of tokens
    pub max_tokens: usize,
    
    /// Whether to process tags
    pub process_tags: bool,
    
    /// Whether the model supports long captions
    pub supports_long_captions: bool,
    
    /// Chunk size for long captions
    pub chunk_size: usize,
    
    /// Whether to add ellipsis on truncation
    pub add_ellipsis_on_truncate: bool,
    
    /// Minimum caption length
    pub min_caption_length: usize,
}

impl CaptionConfig {
    /// Get config for architecture
    pub fn for_architecture(arch: &ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::SD15 | ModelArchitecture::SD15 => Self {
                max_tokens: 77,
                process_tags: true,
                supports_long_captions: false,
                chunk_size: 77,
                add_ellipsis_on_truncate: false,
                min_caption_length: 3,
            },
            ModelArchitecture::SDXL => Self {
                max_tokens: 77,
                process_tags: true,
                supports_long_captions: true,
                chunk_size: 77,
                add_ellipsis_on_truncate: false,
                min_caption_length: 3,
            },
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => Self {
                max_tokens: 256, // T5 can handle more
                process_tags: true,
                supports_long_captions: true,
                chunk_size: 256,
                add_ellipsis_on_truncate: false,
                min_caption_length: 3,
            },
            ModelArchitecture::Flux => Self {
                max_tokens: 512,
                process_tags: true,
                supports_long_captions: true,
                chunk_size: 256,
                add_ellipsis_on_truncate: false,
                min_caption_length: 3,
            },
            _ => Self {
                max_tokens: 77,
                process_tags: false,
                supports_long_captions: false,
                chunk_size: 77,
                add_ellipsis_on_truncate: true,
                min_caption_length: 3,
            },
        }
    }
}

/// Processed caption result
#[derive(Debug, Clone)]
pub struct ProcessedCaption {
    /// Full processed text
    pub text: String,
    
    /// Chunks for long caption support
    pub chunks: Vec<String>,
    
    /// Whether the caption was truncated
    pub was_truncated: bool,
    
    /// Estimated token count
    pub token_count: usize,
}

/// Tag processor for Danbooru-style tags
pub struct TagProcessor {
    quality_tags: HashSet<String>,
    rating_tags: HashSet<String>,
    special_tags: HashSet<String>,
}

impl TagProcessor {
    /// Create new tag processor
    pub fn new() -> Self {
        let mut quality_tags = HashSet::new();
        quality_tags.insert("masterpiece".to_string());
        quality_tags.insert("best quality".to_string());
        quality_tags.insert("high quality".to_string());
        quality_tags.insert("medium quality".to_string());
        quality_tags.insert("low quality".to_string());
        quality_tags.insert("worst quality".to_string());
        
        let mut rating_tags = HashSet::new();
        rating_tags.insert("safe".to_string());
        rating_tags.insert("sensitive".to_string());
        rating_tags.insert("nsfw".to_string());
        rating_tags.insert("explicit".to_string());
        
        let mut special_tags = HashSet::new();
        special_tags.insert("1girl".to_string());
        special_tags.insert("1boy".to_string());
        special_tags.insert("solo".to_string());
        special_tags.insert("multiple girls".to_string());
        special_tags.insert("multiple boys".to_string());
        
        Self {
            quality_tags,
            rating_tags,
            special_tags,
        }
    }
    
    /// Process tags
    pub fn process(&self, caption: &str) -> Result<String> {
        // Check if this is tag-based caption
        if !caption.contains(",") || caption.contains(".") {
            // Probably natural language, don't process
            return Ok(caption.to_string());
        }
        
        let tags: Vec<&str> = caption.split(',').map(|t| t.trim()).collect();
        
        let mut quality_tags = Vec::new();
        let mut special_tags = Vec::new();
        let mut general_tags = Vec::new();
        
        for tag in tags {
            let tag_lower = tag.to_lowercase();
            
            if self.quality_tags.contains(&tag_lower) {
                quality_tags.push(tag);
            } else if self.special_tags.contains(&tag_lower) {
                special_tags.push(tag);
            } else if !self.rating_tags.contains(&tag_lower) {
                // Skip rating tags
                general_tags.push(tag);
            }
        }
        
        // Reorder: quality -> special -> general
        let mut ordered_tags = Vec::new();
        ordered_tags.extend(quality_tags);
        ordered_tags.extend(special_tags);
        ordered_tags.extend(general_tags);
        
        Ok(ordered_tags.join(", "))
    }
}

/// Caption augmentation for training
pub struct CaptionAugmenter {
    dropout_rate: f32,
    shuffle_rate: f32,
    template_rate: f32,
    templates: Vec<String>,
}

impl CaptionAugmenter {
    /// Create new augmenter
    pub fn new(dropout_rate: f32, shuffle_rate: f32, template_rate: f32) -> Self {
        let templates = vec![
            "a photo of {}".to_string(),
            "an image of {}".to_string(),
            "a picture of {}".to_string(),
            "{}, high quality".to_string(),
            "{}, professional photography".to_string(),
            "artwork of {}".to_string(),
            "{} in the style of {}".to_string(),
        ];
        
        Self {
            dropout_rate,
            shuffle_rate,
            template_rate,
            templates,
        }
    }
    
    /// Augment caption
    pub fn augment(&self, caption: &str, rng: &mut impl rand::Rng) -> String {
        let mut caption = caption.to_string();
        
        // Apply dropout
        if rng.gen::<f32>() < self.dropout_rate {
            caption = self.apply_dropout(&caption, rng);
        }
        
        // Apply shuffle
        if rng.gen::<f32>() < self.shuffle_rate {
            caption = self.apply_shuffle(&caption, rng);
        }
        
        // Apply template
        if rng.gen::<f32>() < self.template_rate {
            caption = self.apply_template(&caption, rng);
        }
        
        caption
    }
    
    /// Apply token dropout
    fn apply_dropout(&self, caption: &str, rng: &mut impl rand::Rng) -> String {
        let words: Vec<&str> = caption.split_whitespace().collect();
        let kept_words: Vec<&str> = words.into_iter()
            .filter(|_| rng.gen::<f32>() > 0.1) // 10% dropout per token
            .collect();
        
        if kept_words.is_empty() {
            caption.to_string() // Keep original if all dropped
        } else {
            kept_words.join(" ")
        }
    }
    
    /// Apply token shuffle
    fn apply_shuffle(&self, caption: &str, rng: &mut impl rand::Rng) -> String {
        if caption.contains(",") {
            // Shuffle tags
            let mut tags: Vec<&str> = caption.split(',').map(|t| t.trim()).collect();
            use rand::seq::SliceRandom;
            tags.shuffle(rng);
            tags.join(", ")
        } else {
            // Don't shuffle natural language
            caption.to_string()
        }
    }
    
    /// Apply template
    fn apply_template(&self, caption: &str, rng: &mut impl rand::Rng) -> String {
        use rand::seq::SliceRandom;
        if let Some(template) = self.templates.choose(rng) {
            if template.contains("{}") {
                template.replace("{}", caption)
            } else {
                caption.to_string()
            }
        } else {
            caption.to_string()
        }
    }
}

// Re-export rand traits needed
pub use rand::{Rng, thread_rng};