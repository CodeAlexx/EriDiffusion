use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Debug, Default, Clone)]
pub struct ValidationMetrics {
    pub loss: f32,
    pub num_samples: usize,
    pub losses_per_timestep: Vec<(usize, f32)>, // (timestep, loss)
}
pub struct ValidationRunner {
    dataset: ValidationDataset,
    metrics_history: Vec<(usize, ValidationMetrics)>, // (step, metrics)
}

// Validation dataset support for diffusion training

// FLAME uses flame_core::device::Device instead of Device

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub dataset_path: PathBuf,
    pub batch_size: usize,
    pub every_n_steps: usize,
    pub num_samples: Option<usize>, // Limit number of validation samples
}

/// Validation dataset item
pub struct ValidationItem {
    pub image_path: PathBuf,
    pub caption: String,
    pub latent: Option<Tensor>, // Cached latent
}

/// Validation dataset manager
pub struct ValidationDataset {
    items: Vec<ValidationItem>,
    config: ValidationConfig,
    device: Device,
}

impl ValidationDataset {
    /// Create new validation dataset
    pub fn new(config: ValidationConfig, device: Device) -> flame_core::Result<Self> {
        let mut items = Vec::new();

        // Load validation items
        for entry in fs::read_dir(&config.dataset_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to read directory: {}", e))
        })? {
            let entry = entry.map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to read entry: {}", e))
            })?;
            let path = entry.path();

            if path
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ["jpg", "jpeg", "png"].contains(&ext))
                .unwrap_or(false)
            {
                let caption_path = path.with_extension("txt");
                if caption_path.exists() {
                    let caption = fs::read_to_string(&caption_path).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "Failed to read caption: {}",
                            e
                        ))
                    })?;
                    items.push(ValidationItem {
                        image_path: path,
                        caption: caption.trim().to_string(),
                        latent: None,
                    });
                }
            }
        }

        // Limit number of samples if specified
        if let Some(num_samples) = config.num_samples {
            items.truncate(num_samples);
        }

        println!(
            "Loaded {} validation samples from {}",
            items.len(),
            config.dataset_path.display()
        );

        Ok(Self { items, config, device })
    }

    /// Get number of validation samples
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if validation should run at this step
    pub fn should_validate(&self, step: usize) -> bool {
        step > 0 && step % self.config.every_n_steps == 0
    }

    /// Get validation items
    pub fn items(&self) -> &[ValidationItem] {
        &self.items
    }

    /// Get batched validation items
    pub fn get_batches(&self) -> Vec<&[ValidationItem]> {
        self.items.chunks(self.config.batch_size).collect()
    }

    /// Cache latents for validation set
    pub fn cache_latents<F>(&mut self, encode_fn: F) -> flame_core::Result<()>
    where
        F: Fn(&Path) -> flame_core::Result<Tensor>,
    {
        println!("Caching validation latents...");

        let total_items = self.items.len();
        for (i, item) in self.items.iter_mut().enumerate() {
            if item.latent.is_none() {
                print!("\rCaching validation latent {}/{}", i + 1, total_items);
                std::io::stdout().flush().map_err(|e| {
                    flame_core::Error::InvalidOperation(format!("Failed to flush stdout: {}", e))
                })?;

                let latent = encode_fn(&item.image_path)?;
                item.latent = Some(latent);
            }
        }
        println!("\nValidation latents cached");

        Ok(())
    }
}

// Default implementation removed - already derived

impl ValidationMetrics {
    /// Add loss for a batch
    pub fn add_batch_loss(&mut self, loss: f32, batch_size: usize) {
        self.loss += loss * batch_size as f32;
        self.num_samples += batch_size;
    }

    /// Add loss for specific timestep
    pub fn add_timestep_loss(&mut self, timestep: usize, loss: f32) {
        self.losses_per_timestep.push((timestep, loss));
    }

    /// Get average loss
    pub fn avg_loss(&self) -> f32 {
        if self.num_samples > 0 {
            self.loss / self.num_samples as f32
        } else {
            0.0
        }
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!("Validation Loss: {:.6} (n={})", self.avg_loss(), self.num_samples)
    }
}

impl ValidationRunner {
    pub fn new(dataset: ValidationDataset) -> Self {
        Self { dataset, metrics_history: Vec::new() }
    }

    /// Run validation
    pub fn validate<F>(
        &mut self,
        step: usize,
        compute_loss_fn: F,
    ) -> flame_core::Result<ValidationMetrics>
    where
        F: Fn(&[ValidationItem]) -> flame_core::Result<(f32, usize)>, // Returns (total_loss, num_samples)
    {
        if !self.dataset.should_validate(step) {
            return Ok(ValidationMetrics::default());
        }

        println!("\n=== Running validation at step {} ===", step);

        let mut metrics = ValidationMetrics::default();

        for (i, batch) in self.dataset.get_batches().iter().enumerate() {
            print!("\rValidating batch {}/{}", i + 1, self.dataset.get_batches().len());
            std::io::stdout().flush().map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to flush stdout: {}", e))
            })?;

            let (batch_loss, batch_size) = compute_loss_fn(batch)?;
            metrics.add_batch_loss(batch_loss, batch_size);
        }

        println!("\n{}", metrics.summary());

        // Store metrics
        self.metrics_history.push((step, metrics.clone()));

        // Log improvement/degradation
        if self.metrics_history.len() > 1 {
            let prev_metrics = &self.metrics_history[self.metrics_history.len() - 2].1;
            let improvement = prev_metrics.avg_loss() - metrics.avg_loss();

            if improvement > 0.0 {
                println!("✅ Validation loss improved by {:.6}", improvement);
            } else {
                println!("⚠️  Validation loss increased by {:.6}", -improvement);
            }
        }

        Ok(metrics)
    }

    /// Get metrics history
    pub fn metrics_history(&self) -> &[(usize, ValidationMetrics)] {
        &self.metrics_history
    }

    /// Get best checkpoint step based on validation loss
    pub fn best_checkpoint_step(&self) -> Option<usize> {
        self.metrics_history
            .iter()
            .min_by(|a, b| a.1.avg_loss().partial_cmp(&b.1.avg_loss()).unwrap())
            .map(|(step, _)| *step)
    }

    /// Save metrics to file
    pub fn save_metrics(&self, output_path: &Path) -> flame_core::Result<()> {
        let mut content = String::from("step,validation_loss,num_samples\n");

        for (step, metrics) in &self.metrics_history {
            content.push_str(&format!(
                "{},{:.6},{}\n",
                step,
                metrics.avg_loss(),
                metrics.num_samples
            ));
        }

        fs::write(output_path, content).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to write metrics: {}", e))
        })?;
        println!("Saved validation metrics to {}", output_path.display());

        Ok(())
    }
}

/// Compute MSE loss
pub fn compute_loss(prediction: &Tensor, target: &Tensor) -> flame_core::Result<f32> {
    // MSE loss
    let diff = prediction.sub(target)?;
    let squared = diff.square()?;
    squared.mean()?.to_scalar::<f32>()
}

/// Create sample directory
pub fn create_sample_directory(name: &str) -> flame_core::Result<PathBuf> {
    let dir = PathBuf::from(format!("samples/{}", name));
    std::fs::create_dir_all(&dir).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to create sample directory: {}", e))
    })?;
    Ok(dir)
}
