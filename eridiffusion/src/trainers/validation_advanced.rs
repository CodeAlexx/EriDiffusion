//! Comprehensive validation system for diffusion model training
//!
//! Provides validation dataset management, metrics tracking,
//! and model evaluation during training.

use chrono::{DateTime, Utc};
use flame_core::{CudaDevice, DType, Error, Parameter, Result, Shape, Tensor};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Path to validation dataset
    pub dataset_path: PathBuf,

    /// Batch size for validation
    pub batch_size: usize,

    /// Run validation every N steps
    pub every_n_steps: usize,

    /// Maximum number of validation samples (None = use all)
    pub num_samples: Option<usize>,

    /// Whether to cache VAE latents
    pub cache_latents: bool,

    /// Save validation samples
    pub save_samples: bool,

    /// Number of samples to generate
    pub num_generation_samples: usize,

    /// Timesteps to evaluate (None = random)
    pub eval_timesteps: Option<Vec<usize>>,

    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,

    /// Enable advanced metrics (FID, IS, etc.)
    pub compute_advanced_metrics: bool,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Metric to monitor (e.g., "val_loss", "fid")
    pub monitor: String,

    /// Minimum improvement required
    pub min_delta: f32,

    /// Number of checks with no improvement before stopping
    pub patience: usize,

    /// Whether lower is better
    pub mode: EarlyStoppingMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EarlyStoppingMode {
    Min,
    Max,
}

/// Validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// Step number
    pub step: usize,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Average validation loss
    pub loss: f32,

    /// Loss per timestep
    pub loss_per_timestep: Vec<(usize, f32)>,

    /// Number of samples evaluated
    pub num_samples: usize,

    /// FID score (if computed)
    pub fid_score: Option<f32>,

    /// Inception Score (if computed)
    pub inception_score: Option<(f32, f32)>, // (mean, std)

    /// CLIP score (if computed)
    pub clip_score: Option<f32>,

    /// Custom metrics
    pub custom_metrics: std::collections::HashMap<String, f32>,

    /// Sample generation time
    pub generation_time_ms: Option<u128>,
}

/// Validation dataset
pub struct ValidationDataset {
    items: Vec<ValidationItem>,
    config: ValidationConfig,
    device: Arc<CudaDevice>,
    latent_cache: std::collections::HashMap<PathBuf, Tensor>,
}

/// Single validation item
#[derive(Debug, Clone)]
pub struct ValidationItem {
    pub image_path: PathBuf,
    pub caption: String,
    pub metadata: Option<serde_json::Value>,
}

impl ValidationDataset {
    /// Create new validation dataset
    pub fn new(config: ValidationConfig, device: Arc<CudaDevice>) -> flame_core::Result<Self> {
        let mut items = Vec::new();

        // Load validation items
        let entries = fs::read_dir(&config.dataset_path).map_err(|e| {
            Error::InvalidOperation(format!("Failed to read directory: {}", e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                Error::InvalidOperation(format!("Failed to read entry: {}", e))
            })?;
            let path = entry.path();

            if path
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| matches!(ext, "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
            {
                // Look for corresponding caption file
                let caption_path = path.with_extension("txt");
                let caption = if caption_path.exists() {
                    fs::read_to_string(&caption_path).map_err(|e| {
                        Error::InvalidOperation(format!("Failed to read caption: {}", e))
                    })?
                } else {
                    path.file_stem().and_then(|s| s.to_str()).unwrap_or("a photo").to_string()
                };

                // Look for metadata file
                let metadata_path = path.with_extension("json");
                let metadata = if metadata_path.exists() {
                    let data = fs::read_to_string(&metadata_path).map_err(|e| {
                        Error::InvalidOperation(format!("Failed to read metadata: {}", e))
                    })?;
                    Some(serde_json::from_str(&data).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "Failed to parse JSON: {}",
                            e
                        ))
                    })?)
                } else {
                    None
                };

                items.push(ValidationItem {
                    image_path: path,
                    caption: caption.trim().to_string(),
                    metadata,
                });
            }
        }

        // Apply sample limit if specified
        if let Some(limit) = config.num_samples {
            items.truncate(limit);
        }

        println!("Loaded {} validation samples", items.len());

        Ok(Self { items, config, device, latent_cache: std::collections::HashMap::new() })
    }

    /// Get number of items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get validation items
    pub fn items(&self) -> &[ValidationItem] {
        &self.items
    }

    /// Get or compute latent for an image
    pub fn get_latent(
        &mut self,
        image_path: &Path,
        vae: &impl VAEEncoder,
    ) -> flame_core::Result<Tensor> {
        if !self.config.cache_latents {
            // Always compute fresh
            return self.encode_image(image_path, vae);
        }

        // Check cache
        if let Some(latent) = self.latent_cache.get(image_path) {
            return Ok(latent.clone());
        }

        // Compute and cache
        let latent = self.encode_image(image_path, vae)?;
        self.latent_cache.insert(image_path.to_path_buf(), latent.clone());
        Ok(latent)
    }

    /// Encode image to latent
    fn encode_image(&self, image_path: &Path, vae: &impl VAEEncoder) -> flame_core::Result<Tensor> {
        // Load and preprocess image
        let image = load_image(image_path, self.device.clone())?;

        // Encode with VAE
        let dist = vae.encode(&image)?;
        dist.sample(None)
    }
}

/// VAE encoder trait
pub trait VAEEncoder {
    fn encode(&self, image: &Tensor) -> flame_core::Result<Box<dyn Distribution>>;
}

/// Distribution trait
pub trait Distribution {
    fn sample(&self, generator: Option<&mut dyn rand::RngCore>) -> flame_core::Result<Tensor>;
}

/// Validation runner
pub struct ValidationRunner {
    dataset: ValidationDataset,
    metrics_history: Vec<ValidationMetrics>,
    early_stopping_state: Option<EarlyStoppingState>,
    output_dir: PathBuf,
}

/// Early stopping state
struct EarlyStoppingState {
    config: EarlyStoppingConfig,
    best_value: f32,
    patience_count: usize,
    best_step: usize,
}

impl ValidationRunner {
    /// Create new validation runner
    pub fn new(dataset: ValidationDataset, output_dir: PathBuf) -> flame_core::Result<Self> {
        // Create output directories
        fs::create_dir_all(&output_dir).map_err(|e| {
            Error::InvalidOperation(format!("Failed to create output dir: {}", e))
        })?;
        fs::create_dir_all(output_dir.join("samples")).map_err(|e| {
            Error::InvalidOperation(format!("Failed to create samples dir: {}", e))
        })?;
        fs::create_dir_all(output_dir.join("metrics")).map_err(|e| {
            Error::InvalidOperation(format!("Failed to create metrics dir: {}", e))
        })?;

        // Initialize early stopping if configured
        let early_stopping_state =
            dataset.config.early_stopping.as_ref().map(|config| EarlyStoppingState {
                config: config.clone(),
                best_value: match config.mode {
                    EarlyStoppingMode::Min => f32::INFINITY,
                    EarlyStoppingMode::Max => f32::NEG_INFINITY,
                },
                patience_count: 0,
                best_step: 0,
            });

        Ok(Self { dataset, metrics_history: Vec::new(), early_stopping_state, output_dir })
    }

    /// Run validation
    pub fn validate<M: DiffusionModel>(
        &mut self,
        model: &M,
        vae: &impl VAEEncoder,
        step: usize,
    ) -> flame_core::Result<ValidationMetrics> {
        let start_time = std::time::Instant::now();

        println!("\nRunning validation at step {}...", step);

        // Compute losses
        let (avg_loss, loss_per_timestep) = self.compute_validation_loss(model, vae)?;

        // Generate samples if requested
        let generation_time_ms = if self.dataset.config.save_samples {
            let gen_start = std::time::Instant::now();
            self.generate_samples(model, vae, step)?;
            Some(gen_start.elapsed().as_millis())
        } else {
            None
        };

        // Compute advanced metrics if requested
        let (fid_score, inception_score, clip_score) =
            if self.dataset.config.compute_advanced_metrics {
                self.compute_advanced_metrics(model, vae)?
            } else {
                (None, None, None)
            };

        // Create metrics
        let metrics = ValidationMetrics {
            step,
            timestamp: Utc::now(),
            loss: avg_loss,
            loss_per_timestep,
            num_samples: self.dataset.len(),
            fid_score,
            inception_score,
            clip_score,
            custom_metrics: std::collections::HashMap::new(),
            generation_time_ms,
        };

        // Save metrics
        self.save_metrics(&metrics)?;
        self.metrics_history.push(metrics.clone());

        // Check early stopping
        if let Some(should_stop) = self.check_early_stopping(&metrics)? {
            if should_stop {
                println!("Early stopping triggered at step {}", step);
            }
        }

        let elapsed = start_time.elapsed();
        println!("Validation completed in {:.2}s", elapsed.as_secs_f32());
        println!("  Average loss: {:.4}", avg_loss);
        if let Some(fid) = fid_score {
            println!("  FID score: {:.2}", fid);
        }

        Ok(metrics)
    }

    /// Compute validation loss
    fn compute_validation_loss<M: DiffusionModel>(
        &mut self,
        model: &M,
        vae: &impl VAEEncoder,
    ) -> flame_core::Result<(f32, Vec<(usize, f32)>)> {
        let mut total_loss = 0.0;
        let mut timestep_losses: std::collections::HashMap<usize, (f32, usize)> =
            std::collections::HashMap::new();
        let mut num_samples = 0;

        // Process in batches
        for batch_start in (0..self.dataset.len()).step_by(self.dataset.config.batch_size) {
            let batch_end = (batch_start + self.dataset.config.batch_size).min(self.dataset.len());
            let batch_size = batch_end - batch_start;

            // Prepare batch
            let mut latents = Vec::new();
            let mut captions = Vec::new();

            for i in batch_start..batch_end {
                let item = self.dataset.items()[i].clone();
                let latent = self.dataset.get_latent(&item.image_path, vae)?;
                latents.push(latent);
                captions.push(item.caption);
            }

            // Stack latents
            let latents = Tensor::cat(&latents.iter().collect::<Vec<_>>(), 0)?;

            // Sample timesteps
            let timesteps = if let Some(eval_timesteps) = &self.dataset.config.eval_timesteps {
                // Use specified timesteps
                let mut ts = Vec::new();
                for _ in 0..batch_size {
                    let t = eval_timesteps[num_samples % eval_timesteps.len()];
                    ts.push(t);
                }
                ts
            } else {
                // Random timesteps
                (0..batch_size).map(|_| rand::random::<usize>() % 1000).collect()
            };

            // Compute loss
            let loss = model.compute_loss(&latents, &captions, &timesteps)?;
            let loss_value = loss.mean()?.to_scalar::<f32>()?;

            total_loss += loss_value * batch_size as f32;
            num_samples += batch_size;

            // Track per-timestep losses
            for (i, &t) in timesteps.iter().enumerate() {
                let item_loss = loss.slice(&[(i, i + 1)])?.mean()?.to_scalar::<f32>()?;
                let (sum, count) = timestep_losses.entry(t).or_insert((0.0, 0));
                *sum += item_loss;
                *count += 1;
            }
        }

        // Compute averages
        let avg_loss = total_loss / num_samples as f32;
        let mut loss_per_timestep: Vec<_> =
            timestep_losses.into_iter().map(|(t, (sum, count))| (t, sum / count as f32)).collect();
        loss_per_timestep.sort_by_key(|&(t, _)| t);

        Ok((avg_loss, loss_per_timestep))
    }

    /// Generate validation samples
    fn generate_samples<M: DiffusionModel>(
        &self,
        model: &M,
        vae: &impl VAEEncoder,
        step: usize,
    ) -> flame_core::Result<()> {
        let num_samples = self.dataset.config.num_generation_samples.min(self.dataset.len());

        for i in 0..num_samples {
            let item = &self.dataset.items()[i];
            let sample = model.generate_sample(&item.caption, None)?;

            // Save sample
            let filename = format!("step_{:06}_sample_{:03}.png", step, i);
            let path = self.output_dir.join("samples").join(filename);
            save_tensor_as_image(&sample, &path)?;
        }

        Ok(())
    }

    /// Compute advanced metrics (placeholder - would use Python scripts)
    fn compute_advanced_metrics<M: DiffusionModel>(
        &self,
        _model: &M,
        _vae: &impl VAEEncoder,
    ) -> flame_core::Result<(Option<f32>, Option<(f32, f32)>, Option<f32>)> {
        // In practice, these would call out to Python evaluation scripts
        // For now, return None
        Ok((None, None, None))
    }

    /// Save metrics to file
    fn save_metrics(&self, metrics: &ValidationMetrics) -> flame_core::Result<()> {
        let path = self.output_dir.join("metrics").join(format!("step_{:06}.json", metrics.step));
        let json = serde_json::to_string_pretty(metrics)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to serialize JSON: {}", e))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        fs::write(path, json)
            .map_err(|e| Error::InvalidOperation(format!("Failed to write file: {}", e)))?;

        // Also update latest metrics
        let latest_path = self.output_dir.join("metrics").join("latest.json");
        let latest_json = serde_json::to_string_pretty(metrics)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to serialize JSON: {}", e))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        fs::write(latest_path, latest_json)
            .map_err(|e| Error::InvalidOperation(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Check early stopping condition
    fn check_early_stopping(
        &mut self,
        metrics: &ValidationMetrics,
    ) -> flame_core::Result<Option<bool>> {
        let state = match &mut self.early_stopping_state {
            Some(s) => s,
            None => return Ok(None),
        };

        // Get monitored value
        let value = match state.config.monitor.as_str() {
            "val_loss" => metrics.loss,
            "fid" => metrics.fid_score.unwrap_or(f32::INFINITY),
            metric => metrics.custom_metrics.get(metric).copied().unwrap_or(f32::NAN),
        };

        if value.is_nan() {
            return Ok(None);
        }

        // Check if improved
        let improved = match state.config.mode {
            EarlyStoppingMode::Min => value < state.best_value - state.config.min_delta,
            EarlyStoppingMode::Max => value > state.best_value + state.config.min_delta,
        };

        if improved {
            state.best_value = value;
            state.best_step = metrics.step;
            state.patience_count = 0;
            Ok(Some(false))
        } else {
            state.patience_count += 1;
            if state.patience_count >= state.config.patience {
                Ok(Some(true)) // Should stop
            } else {
                Ok(Some(false))
            }
        }
    }

    /// Get metrics history
    pub fn metrics_history(&self) -> &[ValidationMetrics] {
        &self.metrics_history
    }

    /// Get best step according to early stopping
    pub fn best_step(&self) -> Option<usize> {
        self.early_stopping_state.as_ref().map(|s| s.best_step)
    }

    /// Export metrics to CSV
    pub fn export_metrics_csv(&self, path: &Path) -> flame_core::Result<()> {
        let mut file = fs::File::create(path)
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        // Write header
        writeln!(file, "step,timestamp,loss,num_samples,fid_score,inception_score_mean,inception_score_std,clip_score,generation_time_ms").map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        // Write data
        for m in &self.metrics_history {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{}",
                m.step,
                m.timestamp.to_rfc3339(),
                m.loss,
                m.num_samples,
                m.fid_score.unwrap_or(0.0),
                m.inception_score.map(|(mean, _)| mean).unwrap_or(0.0),
                m.inception_score.map(|(_, std)| std).unwrap_or(0.0),
                m.clip_score.unwrap_or(0.0),
                m.generation_time_ms.unwrap_or(0),
            )
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to write metrics: {}", e))
            })?;
        }

        Ok(())
    }
}

/// Diffusion model trait for validation
pub trait DiffusionModel {
    fn compute_loss(
        &self,
        latents: &Tensor,
        captions: &[String],
        timesteps: &[usize],
    ) -> flame_core::Result<Tensor>;

    fn generate_sample(&self, caption: &str, seed: Option<u64>) -> flame_core::Result<Tensor>;
}

/// Load image from file
fn load_image(path: &Path, device: Arc<CudaDevice>) -> flame_core::Result<Tensor> {
    // Now actually tries to load real image
    use image::{io::Reader, DynamicImage};

    let img = Reader::open(path)
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to open image: {}", e))
        })?
        .decode()
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to decode image: {}", e))
        })?;

    // Convert to RGB if needed
    let rgb_img = img.to_rgb8();
    let (width, height) = (rgb_img.width() as usize, rgb_img.height() as usize);

    // Convert to tensor [C, H, W] format
    let mut data = vec![0.0f32; 3 * height * width];
    for (x, y, pixel) in rgb_img.enumerate_pixels() {
        let idx = y as usize * width + x as usize;
        data[idx] = pixel[0] as f32 / 255.0; // R
        data[height * width + idx] = pixel[1] as f32 / 255.0; // G
        data[2 * height * width + idx] = pixel[2] as f32 / 255.0; // B
    }

    Tensor::from_vec(data, Shape::new(vec![1, 3, height, width]), device)
}

/// Save tensor as image
fn save_tensor_as_image(tensor: &Tensor, path: &Path) -> flame_core::Result<()> {
    // In practice, would convert tensor to image and save
    // For now, just create the file
    fs::File::create(path).map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
    Ok(())
}

/// Create validation runner with default settings
pub fn create_validation_runner(
    config: ValidationConfig,
    output_dir: PathBuf,
    device: Arc<CudaDevice>,
) -> flame_core::Result<ValidationRunner> {
    let dataset = ValidationDataset::new(config, device)?;
    ValidationRunner::new(dataset, output_dir)
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            dataset_path: PathBuf::from("validation"),
            batch_size: 4,
            every_n_steps: 500,
            num_samples: None,
            cache_latents: true,
            save_samples: true,
            num_generation_samples: 8,
            eval_timesteps: None,
            early_stopping: None,
            compute_advanced_metrics: false,
        }
    }
}
