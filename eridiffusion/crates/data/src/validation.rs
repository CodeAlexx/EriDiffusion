//! Data validation and filtering

use eridiffusion_core::{Result, Error};
use candle_core::Tensor;
use std::path::Path;
use serde::{Serialize, Deserialize};

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub check_image_validity: bool,
    pub check_caption_validity: bool,
    pub min_image_size: (usize, usize),
    pub max_image_size: (usize, usize),
    pub allowed_formats: Vec<String>,
    pub check_duplicates: bool,
    pub check_corruption: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_image_validity: true,
            check_caption_validity: true,
            min_image_size: (64, 64),
            max_image_size: (4096, 4096),
            allowed_formats: vec![
                "jpg".to_string(),
                "jpeg".to_string(),
                "png".to_string(),
                "webp".to_string(),
            ],
            check_duplicates: true,
            check_corruption: true,
        }
    }
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

#[derive(Debug)]
pub enum ValidationError {
    InvalidImageFormat(String),
    ImageTooSmall(usize, usize),
    ImageTooLarge(usize, usize),
    CorruptedImage(String),
    MissingCaption(String),
    InvalidCaption(String),
    DuplicateImage(String, String),
}

#[derive(Debug)]
pub enum ValidationWarning {
    UnusualAspectRatio(f32),
    LowQualityImage(String),
    ShortCaption(usize),
    LongCaption(usize),
}

/// Data validator
pub struct DataValidator {
    config: ValidationConfig,
    image_hashes: std::collections::HashMap<u64, String>,
}

impl DataValidator {
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            image_hashes: std::collections::HashMap::new(),
        }
    }
    
    /// Validate image file
    pub async fn validate_image(&mut self, path: &Path) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Check file format
        if self.config.check_image_validity {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if !self.config.allowed_formats.contains(&ext_str) {
                    errors.push(ValidationError::InvalidImageFormat(
                        path.to_string_lossy().to_string()
                    ));
                }
            } else {
                errors.push(ValidationError::InvalidImageFormat(
                    path.to_string_lossy().to_string()
                ));
            }
        }
        
        // Load and check image
        if let Ok(bytes) = tokio::fs::read(path).await {
            // Check for corruption (simplified)
            if self.config.check_corruption && bytes.len() < 100 {
                errors.push(ValidationError::CorruptedImage(
                    path.to_string_lossy().to_string()
                ));
            }
            
            // Check for duplicates
            if self.config.check_duplicates {
                let hash = Self::hash_bytes(&bytes);
                if let Some(existing) = self.image_hashes.get(&hash) {
                    errors.push(ValidationError::DuplicateImage(
                        path.to_string_lossy().to_string(),
                        existing.clone(),
                    ));
                } else {
                    self.image_hashes.insert(hash, path.to_string_lossy().to_string());
                }
            }
            
            // Check dimensions (would use actual image decoding)
            let (width, height) = (512, 512); // Placeholder
            
            if width < self.config.min_image_size.0 || height < self.config.min_image_size.1 {
                errors.push(ValidationError::ImageTooSmall(width, height));
            }
            
            if width > self.config.max_image_size.0 || height > self.config.max_image_size.1 {
                errors.push(ValidationError::ImageTooLarge(width, height));
            }
            
            // Check aspect ratio
            let aspect_ratio = width as f32 / height as f32;
            if aspect_ratio < 0.2 || aspect_ratio > 5.0 {
                warnings.push(ValidationWarning::UnusualAspectRatio(aspect_ratio));
            }
        }
        
        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        }
    }
    
    /// Validate caption
    pub fn validate_caption(&self, caption: &str, image_path: &Path) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        if self.config.check_caption_validity {
            // Check if caption is empty
            if caption.is_empty() {
                errors.push(ValidationError::MissingCaption(
                    image_path.to_string_lossy().to_string()
                ));
            }
            
            // Check caption length
            let word_count = caption.split_whitespace().count();
            
            if word_count < 3 {
                warnings.push(ValidationWarning::ShortCaption(word_count));
            }
            
            if word_count > 100 {
                warnings.push(ValidationWarning::LongCaption(word_count));
            }
            
            // Check for invalid characters
            if caption.chars().any(|c| c.is_control() && c != '\n') {
                errors.push(ValidationError::InvalidCaption(
                    "Contains control characters".to_string()
                ));
            }
        }
        
        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        }
    }
    
    /// Validate dataset item
    pub async fn validate_item(
        &mut self,
        image_path: &Path,
        caption: &str,
    ) -> ValidationResult {
        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();
        
        // Validate image
        let image_result = self.validate_image(image_path).await;
        all_errors.extend(image_result.errors);
        all_warnings.extend(image_result.warnings);
        
        // Validate caption
        let caption_result = self.validate_caption(caption, image_path);
        all_errors.extend(caption_result.errors);
        all_warnings.extend(caption_result.warnings);
        
        ValidationResult {
            valid: all_errors.is_empty(),
            errors: all_errors,
            warnings: all_warnings,
        }
    }
    
    /// Hash bytes for duplicate detection
    fn hash_bytes(bytes: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Generate validation report
    pub fn generate_report(&self, results: &[ValidationResult]) -> ValidationReport {
        let total = results.len();
        let valid = results.iter().filter(|r| r.valid).count();
        let invalid = total - valid;
        
        let mut error_counts = std::collections::HashMap::new();
        let mut warning_counts = std::collections::HashMap::new();
        
        for result in results {
            for error in &result.errors {
                let key = match error {
                    ValidationError::InvalidImageFormat(_) => "Invalid format",
                    ValidationError::ImageTooSmall(_, _) => "Image too small",
                    ValidationError::ImageTooLarge(_, _) => "Image too large",
                    ValidationError::CorruptedImage(_) => "Corrupted image",
                    ValidationError::MissingCaption(_) => "Missing caption",
                    ValidationError::InvalidCaption(_) => "Invalid caption",
                    ValidationError::DuplicateImage(_, _) => "Duplicate image",
                };
                *error_counts.entry(key).or_insert(0) += 1;
            }
            
            for warning in &result.warnings {
                let key = match warning {
                    ValidationWarning::UnusualAspectRatio(_) => "Unusual aspect ratio",
                    ValidationWarning::LowQualityImage(_) => "Low quality",
                    ValidationWarning::ShortCaption(_) => "Short caption",
                    ValidationWarning::LongCaption(_) => "Long caption",
                };
                *warning_counts.entry(key).or_insert(0) += 1;
            }
        }
        
        ValidationReport {
            total_items: total,
            valid_items: valid,
            invalid_items: invalid,
            error_counts,
            warning_counts,
        }
    }
}

/// Validation report
#[derive(Debug)]
pub struct ValidationReport {
    pub total_items: usize,
    pub valid_items: usize,
    pub invalid_items: usize,
    pub error_counts: std::collections::HashMap<&'static str, usize>,
    pub warning_counts: std::collections::HashMap<&'static str, usize>,
}

impl ValidationReport {
    pub fn print_summary(&self) {
        println!("Validation Report:");
        println!("  Total items: {}", self.total_items);
        println!("  Valid items: {} ({:.1}%)", 
            self.valid_items,
            self.valid_items as f32 / self.total_items as f32 * 100.0
        );
        println!("  Invalid items: {} ({:.1}%)",
            self.invalid_items,
            self.invalid_items as f32 / self.total_items as f32 * 100.0
        );
        
        if !self.error_counts.is_empty() {
            println!("\nErrors:");
            for (error_type, count) in &self.error_counts {
                println!("  {}: {}", error_type, count);
            }
        }
        
        if !self.warning_counts.is_empty() {
            println!("\nWarnings:");
            for (warning_type, count) in &self.warning_counts {
                println!("  {}: {}", warning_type, count);
            }
        }
    }
}

/// Quality assessment
pub struct QualityAssessment;

impl QualityAssessment {
    /// Assess image quality
    pub fn assess_image_quality(tensor: &Tensor) -> Result<QualityScore> {
        // Calculate various quality metrics
        let sharpness = Self::calculate_sharpness(tensor)?;
        let contrast = Self::calculate_contrast(tensor)?;
        let brightness = Self::calculate_brightness(tensor)?;
        let noise_level = Self::calculate_noise_level(tensor)?;
        
        // Overall quality score
        let overall = (sharpness + contrast + (1.0 - noise_level.abs()) + 
                      (1.0 - (brightness - 0.5).abs() * 2.0)) / 4.0;
        
        Ok(QualityScore {
            overall,
            sharpness,
            contrast,
            brightness,
            noise_level,
        })
    }
    
    fn calculate_sharpness(tensor: &Tensor) -> Result<f32> {
        // Simplified Laplacian variance
        Ok(0.8) // Placeholder
    }
    
    fn calculate_contrast(tensor: &Tensor) -> Result<f32> {
        // Standard deviation of pixel values
        let flattened = tensor.flatten_all()?;
        let mean = flattened.mean_all()?;
        let variance = ((flattened - mean)?.sqr()?.mean_all()?);
        let std = variance.sqrt()?.to_scalar::<f32>()?;
        Ok(std.min(1.0))
    }
    
    fn calculate_brightness(tensor: &Tensor) -> Result<f32> {
        // Mean pixel value
        let mean = tensor.flatten_all()?.mean_all()?;
        let mean_scalar = mean.to_scalar::<f32>()?;
        Ok(mean_scalar)
    }
    
    fn calculate_noise_level(tensor: &Tensor) -> Result<f32> {
        // Simplified noise estimation
        Ok(0.1) // Placeholder
    }
}

#[derive(Debug)]
pub struct QualityScore {
    pub overall: f32,
    pub sharpness: f32,
    pub contrast: f32,
    pub brightness: f32,
    pub noise_level: f32,
}