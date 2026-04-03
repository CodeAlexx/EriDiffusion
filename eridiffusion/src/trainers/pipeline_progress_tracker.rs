//! Progress tracking utilities for training pipelines

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Progress tracker for training operations
#[derive(Clone)]
pub struct ProgressTracker {
    inner: Arc<Mutex<ProgressTrackerInner>>,
}

struct ProgressTrackerInner {
    // Phase tracking
    current_phase: String,
    phase_start: Instant,

    // Overall progress
    total_steps: usize,
    current_step: usize,
    start_time: Instant,

    // Per-phase stats
    phase_stats: Vec<PhaseStats>,

    // Training metrics
    recent_losses: Vec<f32>,
    recent_grad_norms: Vec<f32>,
    samples_processed: usize,

    // Memory tracking
    peak_memory_mb: f32,
    current_memory_mb: f32,
}

#[derive(Clone)]
struct PhaseStats {
    name: String,
    duration: Duration,
    items_processed: usize,
    errors: usize,
}

impl ProgressTracker {
    pub fn new(total_steps: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ProgressTrackerInner {
                current_phase: "Initialization".to_string(),
                phase_start: Instant::now(),
                total_steps,
                current_step: 0,
                start_time: Instant::now(),
                phase_stats: Vec::new(),
                recent_losses: Vec::new(),
                recent_grad_norms: Vec::new(),
                samples_processed: 0,
                peak_memory_mb: 0.0,
                current_memory_mb: 0.0,
            })),
        }
    }

    /// Start a new phase
    pub fn start_phase(&self, phase_name: &str) {
        let mut inner = self.inner.lock().unwrap();

        // Complete previous phase
        if !inner.current_phase.is_empty() {
            let duration = inner.phase_start.elapsed();
            let phase_name_clone = inner.current_phase.clone();
            inner.phase_stats.push(PhaseStats {
                name: phase_name_clone,
                duration,
                items_processed: 0,
                errors: 0,
            });
        }

        inner.current_phase = phase_name.to_string();
        inner.phase_start = Instant::now();

        println!("\n{}", "=".repeat(60));
        println!("Starting Phase: {}", phase_name);
        println!("{}", "=".repeat(60));
    }

    /// Update progress within current phase
    pub fn update(&self, items_done: usize, total_items: usize, message: &str) {
        let inner = self.inner.lock().unwrap();
        let phase_elapsed = inner.phase_start.elapsed();
        let progress_pct = (items_done as f32 / total_items as f32) * 100.0;

        // Calculate ETA
        let items_per_sec = if phase_elapsed.as_secs() > 0 {
            items_done as f32 / phase_elapsed.as_secs_f32()
        } else {
            0.0
        };

        let remaining_items = total_items.saturating_sub(items_done);
        let eta_seconds =
            if items_per_sec > 0.0 { (remaining_items as f32 / items_per_sec) as u64 } else { 0 };

        let eta_str = format_duration(Duration::from_secs(eta_seconds));

        println!(
            "[{} {}/{}] {:.1}% - {} | {:.1} items/s | ETA: {}",
            inner.current_phase,
            items_done,
            total_items,
            progress_pct,
            message,
            items_per_sec,
            eta_str
        );
    }

    /// Update training step
    pub fn update_step(&self, step: usize, loss: f32, grad_norm: f32, lr: f32) {
        let mut inner = self.inner.lock().unwrap();
        inner.current_step = step;
        inner.samples_processed += 1;

        // Track recent metrics
        inner.recent_losses.push(loss);
        if inner.recent_losses.len() > 100 {
            inner.recent_losses.remove(0);
        }

        inner.recent_grad_norms.push(grad_norm);
        if inner.recent_grad_norms.len() > 100 {
            inner.recent_grad_norms.remove(0);
        }

        // Calculate averages
        let avg_loss = inner.recent_losses.iter().sum::<f32>() / inner.recent_losses.len() as f32;
        let avg_grad =
            inner.recent_grad_norms.iter().sum::<f32>() / inner.recent_grad_norms.len() as f32;

        // Progress
        let progress_pct = (step as f32 / inner.total_steps as f32) * 100.0;
        let elapsed = inner.start_time.elapsed();
        let steps_per_sec = step as f32 / elapsed.as_secs_f32();
        let samples_per_sec = inner.samples_processed as f32 / elapsed.as_secs_f32();

        // ETA
        let remaining_steps = inner.total_steps.saturating_sub(step);
        let eta_seconds = (remaining_steps as f32 / steps_per_sec) as u64;
        let eta_str = format_duration(Duration::from_secs(eta_seconds));

        // Memory usage
        let memory_str = if inner.current_memory_mb > 0.0 {
            format!(
                " | Mem: {:.1}GB (peak: {:.1}GB)",
                inner.current_memory_mb / 1024.0,
                inner.peak_memory_mb / 1024.0
            )
        } else {
            String::new()
        };

        println!("Step {:5}/{} ({:5.1}%) | Loss: {:.6} (avg: {:.6}) | Grad: {:.4} (avg: {:.4}) | LR: {:.2e} | {:.1} steps/s | {:.1} samples/s | ETA: {}{}",
                 step,
                 inner.total_steps,
                 progress_pct,
                 loss,
                 avg_loss,
                 grad_norm,
                 avg_grad,
                 lr,
                 steps_per_sec,
                 samples_per_sec,
                 eta_str,
                 memory_str);
    }

    /// Update memory usage
    pub fn update_memory(&self, current_mb: f32) {
        let mut inner = self.inner.lock().unwrap();
        inner.current_memory_mb = current_mb;
        if current_mb > inner.peak_memory_mb {
            inner.peak_memory_mb = current_mb;
        }
    }

    /// Log an error
    pub fn log_error(&self, error: &str) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(last_phase) = inner.phase_stats.last_mut() {
            last_phase.errors += 1;
        }
        eprintln!("ERROR in {}: {}", inner.current_phase, error);
    }

    /// Print final summary
    pub fn print_summary(&self) {
        let inner = self.inner.lock().unwrap();
        let total_duration = inner.start_time.elapsed();

        println!("\n{}", "=".repeat(60));
        println!("Training Summary");
        println!("{}", "=".repeat(60));
        println!("Total time: {}", format_duration(total_duration));
        println!("Steps completed: {}/{}", inner.current_step, inner.total_steps);
        println!("Samples processed: {}", inner.samples_processed);

        if !inner.recent_losses.is_empty() {
            let final_loss = inner.recent_losses.last().unwrap();
            let avg_loss =
                inner.recent_losses.iter().sum::<f32>() / inner.recent_losses.len() as f32;
            println!("Final loss: {:.6} (avg: {:.6})", final_loss, avg_loss);
        }

        if inner.peak_memory_mb > 0.0 {
            println!("Peak memory usage: {:.1} GB", inner.peak_memory_mb / 1024.0);
        }

        // Phase breakdown
        println!("\nPhase Breakdown:");
        for phase in &inner.phase_stats {
            println!(
                "  {} - {} ({} items, {} errors)",
                phase.name,
                format_duration(phase.duration),
                phase.items_processed,
                phase.errors
            );
        }
    }
}

/// Format duration in human-readable format
fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    } else {
        format!("{:02}:{:02}", minutes, seconds)
    }
}

/// GPU memory tracker
pub struct MemoryTracker {
    device_id: usize,
}

impl MemoryTracker {
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    /// Get current GPU memory usage in MB
    pub fn get_memory_usage_mb(&self) -> Result<f32, String> {
        // Try to get memory info using nvidia-ml
        match std::process::Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                &self.device_id.to_string(),
            ])
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("Failed to parse memory usage: {}", e))
            }
            Err(e) => Err(format!("Failed to run nvidia-smi: {}", e)),
        }
    }
}

/// Error recovery helper
pub struct ErrorRecovery {
    max_retries: usize,
    backoff_ms: u64,
}

impl ErrorRecovery {
    pub fn new() -> Self {
        Self { max_retries: 3, backoff_ms: 1000 }
    }

    /// Retry an operation with exponential backoff
    pub fn retry<F, T, E>(&self, mut op: F) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
        E: std::fmt::Display,
    {
        let mut last_error = None;

        for attempt in 0..self.max_retries {
            match op() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    eprintln!("Attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);

                    if attempt < self.max_retries - 1 {
                        let sleep_ms = self.backoff_ms * (1 << attempt);
                        std::thread::sleep(Duration::from_millis(sleep_ms));
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }
}
