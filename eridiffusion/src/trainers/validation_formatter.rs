//! Pretty formatted validation output for training logs

use crate::trainers::validation_advanced::ValidationMetrics;
use flame_core::Result;
use std::io::{self, Write};
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

/// Validation output formatter for beautiful console output
pub struct ValidationFormatter {
    color_output: bool,
    show_progress: bool,
}

impl ValidationFormatter {
    pub fn new(color_output: bool, show_progress: bool) -> Self {
        Self { color_output, show_progress }
    }

    /// Print validation header
    pub fn print_header(&self, step: usize) -> flame_core::Result<()> {
        let mut stdout = StandardStream::stdout(ColorChoice::Always);

        // Print separator
        println!("================================================================");

        // Print header with step
        if self.color_output {
            stdout.set_color(ColorSpec::new().set_fg(Some(Color::Cyan)).set_bold(true)).map_err(
                |e| flame_core::Error::InvalidOperation(format!("Failed to set color: {}", e)),
            )?;
        }
        println!("Running validation at step {}", step);
        stdout.reset().map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to reset color: {}", e))
        })?;

        println!("================================================================");

        Ok(())
    }

    /// Print batch progress
    pub fn print_batch_progress(&self, current: usize, total: usize) -> flame_core::Result<()> {
        if self.show_progress {
            print!("\rValidating batch {}/{}", current, total);
            io::stdout().flush().map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to flush stdout: {}", e))
            })?;
            if current == total {
                println!(); // New line after completion
            }
        }
        Ok(())
    }

    /// Print sample generation progress
    pub fn print_sample_progress(&self, current: usize, total: usize) -> flame_core::Result<()> {
        if self.show_progress {
            if current == 1 {
                println!("Generating validation samples...");
            }
            print!("\rGenerating sample {}/{}", current, total);
            io::stdout().flush().map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to flush stdout: {}", e))
            })?;
            if current == total {
                println!(); // New line after completion
            }
        }
        Ok(())
    }

    /// Print validation results
    pub fn print_results(
        &self,
        metrics: &ValidationMetrics,
        output_dir: &str,
        previous_best: Option<f32>,
    ) -> flame_core::Result<()> {
        let mut stdout = StandardStream::stdout(ColorChoice::Always);

        // Print sample save location
        println!("Saved samples to {}/samples_step_{:08}", output_dir, metrics.step);

        // Print main metrics
        let perplexity = (metrics.loss).exp();
        print!("Validation Loss: {:.6} (n={}), ", metrics.loss, metrics.num_samples);
        print!("Perplexity: {:.2}", perplexity);

        // Print FID if available
        if let Some(fid) = metrics.fid_score {
            print!(", FID: {:.2}", fid);
        }
        println!();

        // Print loss by timestep range if available
        if !metrics.loss_per_timestep.is_empty() {
            println!("\nLoss by timestep range:");

            // Group into ranges
            let ranges = vec![(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)];

            for (start, end) in ranges {
                let range_losses: Vec<f32> = metrics
                    .loss_per_timestep
                    .iter()
                    .filter(|(t, _)| *t >= start && *t < end)
                    .map(|(_, loss)| *loss)
                    .collect();

                if !range_losses.is_empty() {
                    let avg_loss = range_losses.iter().sum::<f32>() / range_losses.len() as f32;
                    println!("  {}-{}: {:.6}", start, end, avg_loss);
                }
            }
        }

        // Print improvement status
        println!();
        if let Some(prev) = previous_best {
            let improved = metrics.loss < prev;
            let delta = prev - metrics.loss;

            if improved {
                if self.color_output {
                    stdout.set_color(ColorSpec::new().set_fg(Some(Color::Green))).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!("Failed to set color: {}", e))
                    })?;
                }
                println!(
                    "✅ Validation improved: {:.6} -> {:.6} (Δ={:.6})",
                    prev,
                    metrics.loss,
                    delta.abs()
                );
            } else {
                if self.color_output {
                    stdout.set_color(ColorSpec::new().set_fg(Some(Color::Yellow))).map_err(
                        |e| {
                            flame_core::Error::InvalidOperation(format!(
                                "Failed to set color: {}",
                                e
                            ))
                        },
                    )?;
                }
                println!(
                    "⚠️  No improvement: {:.6} -> {:.6} (Δ=+{:.6})",
                    prev,
                    metrics.loss,
                    delta.abs()
                );
            }
            stdout.reset().map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to reset color: {}", e))
            })?;
        }

        // Print save location
        if self.color_output {
            stdout.set_color(ColorSpec::new().set_fg(Some(Color::Blue))).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to set color: {}", e))
            })?;
        }
        println!("📊 Saved validation plots to {}", output_dir);
        stdout.reset().map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to reset color: {}", e))
        })?;

        Ok(())
    }

    /// Print early stopping warning
    pub fn print_early_stopping_warning(
        &self,
        patience_count: usize,
        max_patience: usize,
    ) -> flame_core::Result<()> {
        let mut stdout = StandardStream::stdout(ColorChoice::Always);

        if self.color_output {
            stdout.set_color(ColorSpec::new().set_fg(Some(Color::Yellow))).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to set color: {}", e))
            })?;
        }

        println!("\n⚠️  Early stopping patience: {}/{}", patience_count, max_patience);

        stdout.reset().map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to reset color: {}", e))
        })?;
        Ok(())
    }

    /// Print validation complete message
    pub fn print_complete(&self, elapsed_secs: f32) -> flame_core::Result<()> {
        let mut stdout = StandardStream::stdout(ColorChoice::Always);

        if self.color_output {
            stdout.set_color(ColorSpec::new().set_fg(Some(Color::Green))).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to set color: {}", e))
            })?;
        }

        println!("\n✓ Validation completed in {:.1}s", elapsed_secs);

        stdout.reset().map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to reset color: {}", e))
        })?;
        Ok(())
    }

    /// Print error message
    pub fn print_error(&self, error: &str) -> flame_core::Result<()> {
        let mut stderr = StandardStream::stderr(ColorChoice::Always);

        if self.color_output {
            stderr.set_color(ColorSpec::new().set_fg(Some(Color::Red)).set_bold(true)).map_err(
                |e| flame_core::Error::InvalidOperation(format!("Failed to set color: {}", e)),
            )?;
        }

        eprintln!("\n❌ Validation error: {}", error);

        stderr.reset().map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to reset color: {}", e))
        })?;
        Ok(())
    }
}

/// Helper function to create a default formatter
pub fn create_validation_formatter() -> ValidationFormatter {
    ValidationFormatter::new(
        atty::is(atty::Stream::Stdout), // Auto-detect color support
        true,                           // Show progress by default
    )
}

/// Format validation metrics as a summary string
pub fn format_validation_summary(metrics: &ValidationMetrics) -> String {
    let mut summary = format!(
        "Step {} | Loss: {:.6} | Samples: {}",
        metrics.step, metrics.loss, metrics.num_samples
    );

    if let Some(fid) = metrics.fid_score {
        summary.push_str(&format!(" | FID: {:.2}", fid));
    }

    if let Some((is_mean, is_std)) = metrics.inception_score {
        summary.push_str(&format!(" | IS: {:.2}±{:.2}", is_mean, is_std));
    }

    if let Some(clip) = metrics.clip_score {
        summary.push_str(&format!(" | CLIP: {:.3}", clip));
    }

    summary
}

/// Format loss by timestep for logging
pub fn format_timestep_losses(loss_per_timestep: &[(usize, f32)]) -> String {
    if loss_per_timestep.is_empty() {
        return String::new();
    }

    let mut output = String::from("Timestep losses: ");

    // Sample some key timesteps
    let key_timesteps = vec![0, 250, 500, 750, 999];

    for t in key_timesteps {
        if let Some((_, loss)) = loss_per_timestep.iter().find(|(ts, _)| *ts == t) {
            output.push_str(&format!("t{}={:.4} ", t, loss));
        }
    }

    output
}

/// Progress bar for validation batches
pub struct ValidationProgressBar {
    total: usize,
    current: usize,
    width: usize,
}

impl ValidationProgressBar {
    pub fn new(total: usize) -> Self {
        Self { total, current: 0, width: 50 }
    }

    pub fn update(&mut self, current: usize) {
        self.current = current;
        self.draw();
    }

    pub fn finish(&self) {
        println!(); // New line after progress bar
    }

    fn draw(&self) {
        let progress = self.current as f32 / self.total as f32;
        let filled = (progress * self.width as f32) as usize;
        let empty = self.width - filled;

        print!("\r[");
        print!("{}", "=".repeat(filled));
        if filled < self.width {
            print!(">");
            print!("{}", " ".repeat(empty.saturating_sub(1)));
        }
        print!("] {}/{} ({:.0}%)", self.current, self.total, progress * 100.0);

        io::stdout().flush().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    #[test]
    fn test_format_validation_summary() {
        let metrics = ValidationMetrics {
            step: 5000,
            timestamp: Utc::now(),
            loss: 0.045623,
            loss_per_timestep: vec![],
            num_samples: 40,
            fid_score: Some(25.40),
            inception_score: Some((8.5, 0.3)),
            clip_score: Some(0.285),
            custom_metrics: HashMap::new(),
            generation_time_ms: Some(1500),
        };

        let summary = format_validation_summary(&metrics);
        assert!(summary.contains("Step 5000"));
        assert!(summary.contains("Loss: 0.045623"));
        assert!(summary.contains("FID: 25.40"));
        assert!(summary.contains("IS: 8.50±0.30"));
        assert!(summary.contains("CLIP: 0.285"));
    }

    #[test]
    fn test_format_timestep_losses() {
        let losses = vec![(0, 0.01), (250, 0.03), (500, 0.05), (750, 0.07), (999, 0.09)];

        let formatted = format_timestep_losses(&losses);
        assert!(formatted.contains("t0=0.0100"));
        assert!(formatted.contains("t500=0.0500"));
        assert!(formatted.contains("t999=0.0900"));
    }
}
