//! Caption processing utilities

use eridiffusion_core::{Result, Error};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use regex::Regex;

/// Caption processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionConfig {
    pub max_length: Option<usize>,
    pub min_length: Option<usize>,
    pub dropout_rate: f32,
    pub tag_dropout_rate: f32,
    pub shuffle_tags: bool,
    pub tag_separator: String,
    pub use_weighted_tags: bool,
    pub template: Option<String>,
}

impl Default for CaptionConfig {
    fn default() -> Self {
        Self {
            max_length: Some(77),
            min_length: None,
            dropout_rate: 0.0,
            tag_dropout_rate: 0.0,
            shuffle_tags: false,
            tag_separator: ", ".to_string(),
            use_weighted_tags: false,
            template: None,
        }
    }
}

/// Caption processor
pub struct CaptionProcessor {
    config: CaptionConfig,
    tag_weights: HashMap<String, f32>,
    banned_tags: HashSet<String>,
    tag_aliases: HashMap<String, String>,
}

impl CaptionProcessor {
    /// Create new caption processor
    pub fn new(config: CaptionConfig) -> Self {
        Self {
            config,
            tag_weights: HashMap::new(),
            banned_tags: HashSet::new(),
            tag_aliases: HashMap::new(),
        }
    }
    
    /// Load tag weights from file
    pub async fn load_tag_weights(&mut self, path: &std::path::Path) -> Result<()> {
        let content = tokio::fs::read_to_string(path).await?;
        for line in content.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                if let Ok(weight) = parts[1].parse::<f32>() {
                    self.tag_weights.insert(parts[0].to_string(), weight);
                }
            }
        }
        Ok(())
    }
    
    /// Process caption
    pub fn process(&self, caption: &str) -> String {
        let mut rng = fastrand::Rng::new();
        
        // Apply dropout
        if rng.f32() < self.config.dropout_rate {
            return String::new();
        }
        
        // Parse tags if present
        let processed = if caption.contains(&self.config.tag_separator) {
            self.process_tags(caption, &mut rng)
        } else {
            self.process_sentence(caption)
        };
        
        // Apply template
        let processed = if let Some(ref template) = self.config.template {
            template.replace("{caption}", &processed)
        } else {
            processed
        };
        
        // Apply length constraints
        self.apply_length_constraints(&processed)
    }
    
    /// Process tag-based caption
    fn process_tags(&self, caption: &str, rng: &mut fastrand::Rng) -> String {
        let mut tags: Vec<String> = caption
            .split(&self.config.tag_separator)
            .map(|s| s.trim().to_string())
            .filter(|tag| !self.banned_tags.contains(tag))
            .collect();
        
        // Apply tag aliases
        for tag in &mut tags {
            if let Some(alias) = self.tag_aliases.get(tag) {
                *tag = alias.clone();
            }
        }
        
        // Apply tag dropout
        if self.config.tag_dropout_rate > 0.0 {
            tags.retain(|_| rng.f32() >= self.config.tag_dropout_rate);
        }
        
        // Sort by weight if using weighted tags
        if self.config.use_weighted_tags {
            tags.sort_by(|a, b| {
                let weight_a = self.tag_weights.get(a).unwrap_or(&1.0);
                let weight_b = self.tag_weights.get(b).unwrap_or(&1.0);
                weight_b.partial_cmp(weight_a).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        
        // Shuffle tags if configured
        if self.config.shuffle_tags {
            let mut shuffled = tags.clone();
            rng.shuffle(&mut shuffled);
            tags = shuffled;
        }
        
        tags.join(&self.config.tag_separator)
    }
    
    /// Process sentence-based caption
    fn process_sentence(&self, caption: &str) -> String {
        // Clean up caption
        let cleaned = caption
            .trim()
            .replace("  ", " ")
            .replace("\n", " ")
            .replace("\r", "");
        
        // Apply basic cleaning
        let re = Regex::new(r"[^\w\s,.!?-]").unwrap();
        re.replace_all(&cleaned, "").to_string()
    }
    
    /// Apply length constraints
    fn apply_length_constraints(&self, caption: &str) -> String {
        let mut result = caption.to_string();
        
        // Apply minimum length
        if let Some(min_len) = self.config.min_length {
            if result.len() < min_len {
                // Pad with template or generic text
                result = format!("{} {}", result, "artwork");
            }
        }
        
        // Apply maximum length
        if let Some(max_len) = self.config.max_length {
            if result.len() > max_len {
                // Truncate at word boundary
                if let Some(pos) = result[..max_len].rfind(' ') {
                    result.truncate(pos);
                } else {
                    result.truncate(max_len);
                }
            }
        }
        
        result
    }
    
    /// Add banned tags
    pub fn add_banned_tags(&mut self, tags: Vec<String>) {
        for tag in tags {
            self.banned_tags.insert(tag);
        }
    }
    
    /// Add tag aliases
    pub fn add_tag_aliases(&mut self, aliases: HashMap<String, String>) {
        self.tag_aliases.extend(aliases);
    }
}

/// Caption augmentation
pub struct CaptionAugmenter {
    synonyms: HashMap<String, Vec<String>>,
    style_modifiers: Vec<String>,
    quality_modifiers: Vec<String>,
}

impl CaptionAugmenter {
    pub fn new() -> Self {
        let mut synonyms = HashMap::new();
        synonyms.insert("girl".to_string(), vec![
            "woman".to_string(),
            "female".to_string(),
            "lady".to_string(),
        ]);
        synonyms.insert("boy".to_string(), vec![
            "man".to_string(),
            "male".to_string(),
            "guy".to_string(),
        ]);
        
        let style_modifiers = vec![
            "digital art".to_string(),
            "oil painting".to_string(),
            "watercolor".to_string(),
            "pencil sketch".to_string(),
            "concept art".to_string(),
        ];
        
        let quality_modifiers = vec![
            "highly detailed".to_string(),
            "masterpiece".to_string(),
            "best quality".to_string(),
            "4k".to_string(),
            "award winning".to_string(),
        ];
        
        Self {
            synonyms,
            style_modifiers,
            quality_modifiers,
        }
    }
    
    /// Augment caption with variations
    pub fn augment(&self, caption: &str, num_variations: usize) -> Vec<String> {
        let mut variations = vec![caption.to_string()];
        let mut rng = fastrand::Rng::new();
        
        for _ in 0..num_variations {
            let mut variant = caption.to_string();
            
            // Replace synonyms
            for (word, synonyms) in &self.synonyms {
                if variant.contains(word) && rng.f32() < 0.5 {
                    let replacement = &synonyms[rng.usize(0..synonyms.len())];
                    variant = variant.replace(word, replacement);
                }
            }
            
            // Add style modifier
            if rng.f32() < 0.3 {
                let style = &self.style_modifiers[rng.usize(0..self.style_modifiers.len())];
                variant = format!("{}, {}", variant, style);
            }
            
            // Add quality modifier
            if rng.f32() < 0.3 {
                let quality = &self.quality_modifiers[rng.usize(0..self.quality_modifiers.len())];
                variant = format!("{}, {}", variant, quality);
            }
            
            variations.push(variant);
        }
        
        variations
    }
}

/// BLIP-style caption templates
pub struct CaptionTemplates;

impl CaptionTemplates {
    pub fn question_answer(question: &str, answer: &str) -> String {
        format!("Question: {} Answer: {}", question, answer)
    }
    
    pub fn description(desc: &str) -> String {
        format!("A photo of {}", desc)
    }
    
    pub fn detailed(subject: &str, style: &str, quality: &str) -> String {
        format!("{}, {}, {}", subject, style, quality)
    }
    
    pub fn instruction(instruction: &str, response: &str) -> String {
        format!("### Instruction: {}\n### Response: {}", instruction, response)
    }
}

/// Caption filtering
pub struct CaptionFilter {
    min_words: usize,
    max_words: usize,
    required_words: HashSet<String>,
    forbidden_words: HashSet<String>,
}

impl CaptionFilter {
    pub fn new() -> Self {
        Self {
            min_words: 3,
            max_words: 100,
            required_words: HashSet::new(),
            forbidden_words: HashSet::new(),
        }
    }
    
    /// Check if caption passes filter
    pub fn passes(&self, caption: &str) -> bool {
        let words: Vec<&str> = caption.split_whitespace().collect();
        let word_count = words.len();
        
        // Check word count
        if word_count < self.min_words || word_count > self.max_words {
            return false;
        }
        
        // Check required words
        for required in &self.required_words {
            if !caption.contains(required) {
                return false;
            }
        }
        
        // Check forbidden words
        for forbidden in &self.forbidden_words {
            if caption.contains(forbidden) {
                return false;
            }
        }
        
        true
    }
}