use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for bucket-aware training with CUDA alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketAlignmentConfig {
    /// Enable automatic CUDA alignment
    pub enable_alignment: bool,

    /// Target dtype for alignment (BF16 recommended for memory efficiency)
    pub target_dtype: String, // "BF16", "F32", etc.

    /// CUDA alignment boundary in bytes (default: 4MB)
    pub alignment_boundary_mb: usize,

    /// Maximum memory overhead allowed for alignment (as percentage)
    pub max_overhead_percent: f64,

    /// Predefined bucket configurations
    pub bucket_configs: Vec<BucketConfig>,

    /// Whether to precompute alignments for common buckets
    pub precompute_alignments: bool,

    /// Cache alignment strategies to disk
    pub cache_alignment_strategies: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketConfig {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub aspect_ratio: String, // e.g., "1:1", "4:3", "16:9"

    /// Optional: Override alignment for this specific bucket
    pub force_aligned_width: Option<usize>,
    pub force_aligned_height: Option<usize>,

    /// Whether this bucket is commonly used (affects precomputation)
    pub is_common: bool,
}

impl Default for BucketAlignmentConfig {
    fn default() -> Self {
        Self {
            enable_alignment: true,
            target_dtype: "BF16".to_string(),
            alignment_boundary_mb: 4,
            max_overhead_percent: 25.0, // Allow up to 25% memory overhead
            precompute_alignments: true,
            cache_alignment_strategies: true,
            bucket_configs: Self::default_bucket_configs(),
        }
    }
}

impl BucketAlignmentConfig {
    /// Generate common bucket configurations for training
    fn default_bucket_configs() -> Vec<BucketConfig> {
        vec![
            // Square buckets
            BucketConfig {
                name: "square_512".to_string(),
                width: 512,
                height: 512,
                aspect_ratio: "1:1".to_string(),
                force_aligned_width: None,
                force_aligned_height: None,
                is_common: true,
            },
            BucketConfig {
                name: "square_640".to_string(),
                width: 640,
                height: 640,
                aspect_ratio: "1:1".to_string(),
                force_aligned_width: None,
                force_aligned_height: None,
                is_common: true,
            },
            BucketConfig {
                name: "square_768".to_string(),
                width: 768,
                height: 768,
                aspect_ratio: "1:1".to_string(),
                force_aligned_width: None,
                force_aligned_height: None,
                is_common: true,
            },
            BucketConfig {
                name: "square_896".to_string(),
                width: 896,
                height: 896,
                aspect_ratio: "1:1".to_string(),
                force_aligned_width: None,
                force_aligned_height: None,
                is_common: true,
            },
            BucketConfig {
                name: "square_1024".to_string(),
                width: 1024,
                height: 1024,
                aspect_ratio: "1:1".to_string(),
                force_aligned_width: Some(1088), // Force better alignment
                force_aligned_height: Some(1088),
                is_common: true,
            },
            // Landscape 4:3
            BucketConfig {
                name: "landscape_4_3_small".to_string(),
                width: 683,
                height: 512,
                aspect_ratio: "4:3".to_string(),
                force_aligned_width: None,
                force_aligned_height: None,
                is_common: true,
            },
            BucketConfig {
                name: "landscape_4_3_medium".to_string(),
                width: 768,
                height: 576,
                aspect_ratio: "4:3".to_string(),
                force_aligned_width: None,
                force_aligned_height: None,
                is_common: true,
            },
            BucketConfig {
                name: "landscape_4_3_large".to_string(),
                width: 1024,
                height: 768,
                aspect_ratio: "4:3".to_string(),
                force_aligned_width: Some(1088),
                force_aligned_height: Some(768),
                is_common: true,
            },
            // Portrait 3:4
            BucketConfig {
                name: "portrait_3_4_small".to_string(),
                width: 512,
                height: 683,
                aspect_ratio: "3:4".to_string(),
                force_aligned_width: None,
                force_aligned_height: None,
                is_common: true,
            },
            BucketConfig {
                name: "portrait_3_4_medium".to_string(),
                width: 576,
                height: 768,
                aspect_ratio: "3:4".to_string(),
                force_aligned_width: None,
                force_aligned_height: None,
                is_common: true,
            },
            BucketConfig {
                name: "portrait_3_4_large".to_string(),
                width: 768,
                height: 1024,
                aspect_ratio: "3:4".to_string(),
                force_aligned_width: Some(768),
                force_aligned_height: Some(1088),
                is_common: true,
            },
            // Wide landscape 16:9
            BucketConfig {
                name: "wide_16_9_small".to_string(),
                width: 910,
                height: 512,
                aspect_ratio: "16:9".to_string(),
                force_aligned_width: Some(960),
                force_aligned_height: None,
                is_common: false,
            },
            BucketConfig {
                name: "wide_16_9_medium".to_string(),
                width: 1024,
                height: 576,
                aspect_ratio: "16:9".to_string(),
                force_aligned_width: Some(1088),
                force_aligned_height: None,
                is_common: false,
            },
            // Tall portrait 9:16
            BucketConfig {
                name: "tall_9_16_small".to_string(),
                width: 512,
                height: 910,
                aspect_ratio: "9:16".to_string(),
                force_aligned_width: None,
                force_aligned_height: Some(960),
                is_common: false,
            },
            BucketConfig {
                name: "tall_9_16_medium".to_string(),
                width: 576,
                height: 1024,
                aspect_ratio: "9:16".to_string(),
                force_aligned_width: None,
                force_aligned_height: Some(1088),
                is_common: false,
            },
        ]
    }

    /// Load configuration from YAML file
    pub fn from_yaml_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: BucketAlignmentConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to YAML file
    pub fn to_yaml_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }

    /// Get bucket config by name
    pub fn get_bucket_config(&self, name: &str) -> Option<&BucketConfig> {
        self.bucket_configs.iter().find(|b| b.name == name)
    }

    /// Get all common buckets (for precomputation)
    pub fn get_common_buckets(&self) -> Vec<&BucketConfig> {
        self.bucket_configs.iter().filter(|b| b.is_common).collect()
    }

    /// Generate alignment report for all configured buckets
    pub fn generate_alignment_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Bucket Alignment Report ===\n");

        let dtype_size = match self.target_dtype.as_str() {
            "BF16" => 2,
            "F32" => 4,
            _ => 4,
        };

        let alignment_bytes = self.alignment_boundary_mb * 1024 * 1024;

        for bucket in &self.bucket_configs {
            let (aligned_w, aligned_h) = if let (Some(fw), Some(fh)) =
                (bucket.force_aligned_width, bucket.force_aligned_height)
            {
                (fw, fh)
            } else {
                (bucket.width, bucket.height)
            };

            let original_bytes = bucket.width * bucket.height * 3 * dtype_size;
            let aligned_bytes = aligned_w * aligned_h * 3 * dtype_size;
            let overhead = ((aligned_bytes as f64 / original_bytes as f64) - 1.0) * 100.0;
            let is_aligned = aligned_bytes % alignment_bytes == 0;

            report.push_str(&format!(
                "{}: {}x{} -> {}x{} ({:.1}% overhead, aligned: {})\n",
                bucket.name,
                bucket.width,
                bucket.height,
                aligned_w,
                aligned_h,
                overhead,
                is_aligned
            ));
        }

        report
    }
}

/// Example YAML configuration file content
pub const EXAMPLE_CONFIG_YAML: &str = r#"
enable_alignment: true
target_dtype: "BF16"
alignment_boundary_mb: 4
max_overhead_percent: 25.0
precompute_alignments: true
cache_alignment_strategies: true

bucket_configs:
  - name: "square_1024"
    width: 1024
    height: 1024
    aspect_ratio: "1:1"
    force_aligned_width: 1088
    force_aligned_height: 1088
    is_common: true
    
  - name: "landscape_1024_768"
    width: 1024
    height: 768
    aspect_ratio: "4:3"
    force_aligned_width: 1088
    force_aligned_height: 768
    is_common: true
    
  - name: "portrait_768_1024"
    width: 768
    height: 1024
    aspect_ratio: "3:4"
    force_aligned_width: 768
    force_aligned_height: 1088
    is_common: true
"#;

/// Command-line interface for bucket management
pub struct BucketCLI;

impl BucketCLI {
    /// Generate a default configuration file
    pub fn generate_config(
        output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = BucketAlignmentConfig::default();
        config.to_yaml_file(output_path)?;
        println!("Generated default bucket configuration: {}", output_path.display());
        Ok(())
    }

    /// Analyze alignment for a specific bucket size
    pub fn analyze_bucket(width: usize, height: usize) {
        println!("=== Bucket Analysis: {}x{} ===", width, height);

        for (dtype_name, dtype_size) in [("F32", 4), ("BF16", 2)] {
            let elements = width * height * 3;
            let bytes = elements * dtype_size;
            let mb = bytes as f64 / (1024.0 * 1024.0);
            let aligned_4mb = bytes % (4 * 1024 * 1024) == 0;
            let aligned_2mb = bytes % (2 * 1024 * 1024) == 0;

            println!("  {}: {:.2}MB ({} bytes)", dtype_name, mb, bytes);
            println!("    4MB aligned: {}", aligned_4mb);
            println!("    2MB aligned: {}", aligned_2mb);
        }

        // Suggest aligned dimensions
        let mut manager =
            crate::models::bucket_alignment::BucketAlignmentManager::new(flame_core::DType::BF16);
        let strategy = manager.get_alignment_strategy(height, width);

        println!("  Suggested alignment: {}x{}", strategy.aligned_height, strategy.aligned_width);
        println!(
            "  Memory overhead: {:.1}%",
            ((strategy.memory_bytes as f64 / (width * height * 3 * 2) as f64) - 1.0) * 100.0
        );
    }

    /// Test all configured buckets
    pub fn test_config(config_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let config = BucketAlignmentConfig::from_yaml_file(config_path)?;
        println!("{}", config.generate_alignment_report());
        Ok(())
    }
}

/// Integration example for your training config
pub fn integrate_with_training_config(
    training_config_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Your existing training config loading
    // Add bucket alignment section:

    let bucket_config = BucketAlignmentConfig::default();

    // Save alongside training config
    let bucket_config_path = training_config_path.with_file_name("bucket_alignment.yaml");
    bucket_config.to_yaml_file(&bucket_config_path)?;

    println!("Created bucket alignment config: {}", bucket_config_path.display());
    println!("Add this to your training YAML:");
    println!("bucket_alignment_config: \"{}\"", bucket_config_path.display());

    Ok(())
}
