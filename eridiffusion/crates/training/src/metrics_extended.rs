//! Extended metrics and logging functionality

use eridiffusion_core::{Result, Error};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Extended metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtendedMetricValue {
    Scalar(f32),
    Histogram(Vec<f32>),
    Image(Vec<u8>),
    Text(String),
    Audio(Vec<f32>),
    Embedding(Vec<f32>),
}

/// Logger trait for various backends
#[async_trait::async_trait]
pub trait MetricLogger: Send + Sync {
    async fn log_scalar(&self, name: &str, value: f32, step: usize) -> Result<()>;
    async fn log_histogram(&self, name: &str, values: &[f32], step: usize) -> Result<()>;
    async fn log_image(&self, name: &str, image: &[u8], step: usize) -> Result<()>;
    async fn log_text(&self, name: &str, text: &str, step: usize) -> Result<()>;
    async fn flush(&self) -> Result<()>;
    async fn close(&self) -> Result<()>;
}

/// Console logger implementation
pub struct ConsoleLogger {
    interval: usize,
    last_log: Arc<RwLock<usize>>,
    format: ConsoleFormat,
}

#[derive(Debug, Clone)]
pub enum ConsoleFormat {
    Simple,
    Detailed,
    Json,
}

impl ConsoleLogger {
    pub fn new(interval: usize, format: ConsoleFormat) -> Self {
        Self {
            interval,
            last_log: Arc::new(RwLock::new(0)),
            format,
        }
    }
}

#[async_trait::async_trait]
impl MetricLogger for ConsoleLogger {
    async fn log_scalar(&self, name: &str, value: f32, step: usize) -> Result<()> {
        let mut last = self.last_log.write().await;
        
        if step - *last >= self.interval {
            match self.format {
                ConsoleFormat::Simple => {
                    println!("[Step {}] {}: {:.6}", step, name, value);
                }
                ConsoleFormat::Detailed => {
                    println!("[Step {} @ {}] {}: {:.6}", 
                        step, 
                        chrono::Local::now().format("%H:%M:%S"),
                        name, 
                        value
                    );
                }
                ConsoleFormat::Json => {
                    println!("{}", serde_json::json!({
                        "step": step,
                        "metric": name,
                        "value": value,
                        "timestamp": chrono::Local::now().to_rfc3339()
                    }));
                }
            }
            *last = step;
        }
        
        Ok(())
    }
    
    async fn log_histogram(&self, name: &str, values: &[f32], step: usize) -> Result<()> {
        let stats = calculate_statistics(values);
        println!("[Step {}] {} histogram - mean: {:.4}, std: {:.4}, min: {:.4}, max: {:.4}",
            step, name, stats.mean, stats.std, stats.min, stats.max);
        Ok(())
    }
    
    async fn log_image(&self, name: &str, _image: &[u8], step: usize) -> Result<()> {
        println!("[Step {}] Image logged: {}", step, name);
        Ok(())
    }
    
    async fn log_text(&self, name: &str, text: &str, step: usize) -> Result<()> {
        println!("[Step {}] {}: {}", step, name, text);
        Ok(())
    }
    
    async fn flush(&self) -> Result<()> {
        Ok(())
    }
    
    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

/// TensorBoard logger
pub struct TensorBoardLogger {
    log_dir: String,
    writer: Arc<RwLock<Option<()>>>, // Would be actual TensorBoard writer
}

impl TensorBoardLogger {
    pub async fn new(log_dir: &str) -> Result<Self> {
        tokio::fs::create_dir_all(log_dir).await?;
        
        Ok(Self {
            log_dir: log_dir.to_string(),
            writer: Arc::new(RwLock::new(None)),
        })
    }
}

#[async_trait::async_trait]
impl MetricLogger for TensorBoardLogger {
    async fn log_scalar(&self, name: &str, value: f32, step: usize) -> Result<()> {
        // Would write to TensorBoard event file
        Ok(())
    }
    
    async fn log_histogram(&self, name: &str, values: &[f32], step: usize) -> Result<()> {
        // Would write histogram to TensorBoard
        Ok(())
    }
    
    async fn log_image(&self, name: &str, image: &[u8], step: usize) -> Result<()> {
        // Would write image to TensorBoard
        Ok(())
    }
    
    async fn log_text(&self, name: &str, text: &str, step: usize) -> Result<()> {
        // Would write text to TensorBoard
        Ok(())
    }
    
    async fn flush(&self) -> Result<()> {
        // Would flush TensorBoard writer
        Ok(())
    }
    
    async fn close(&self) -> Result<()> {
        // Would close TensorBoard writer
        Ok(())
    }
}

/// CSV logger
pub struct CSVLogger {
    path: String,
    headers_written: Arc<RwLock<bool>>,
    buffer: Arc<RwLock<Vec<HashMap<String, String>>>>,
    buffer_size: usize,
}

impl CSVLogger {
    pub fn new(path: &str, buffer_size: usize) -> Self {
        Self {
            path: path.to_string(),
            headers_written: Arc::new(RwLock::new(false)),
            buffer: Arc::new(RwLock::new(Vec::new())),
            buffer_size,
        }
    }
    
    async fn write_buffer(&self) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        
        let mut buffer = self.buffer.write().await;
        if buffer.is_empty() {
            return Ok(());
        }
        
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;
        
        let mut headers_written = self.headers_written.write().await;
        
        // Write headers if needed
        if !*headers_written && !buffer.is_empty() {
            let headers: Vec<String> = buffer[0].keys().cloned().collect();
            file.write_all(headers.join(",").as_bytes()).await?;
            file.write_all(b"\n").await?;
            *headers_written = true;
        }
        
        // Write rows
        for row in buffer.iter() {
            let values: Vec<String> = row.values().cloned().collect();
            file.write_all(values.join(",").as_bytes()).await?;
            file.write_all(b"\n").await?;
        }
        
        buffer.clear();
        Ok(())
    }
}

#[async_trait::async_trait]
impl MetricLogger for CSVLogger {
    async fn log_scalar(&self, name: &str, value: f32, step: usize) -> Result<()> {
        let mut buffer = self.buffer.write().await;
        
        let mut row = HashMap::new();
        row.insert("step".to_string(), step.to_string());
        row.insert(name.to_string(), format!("{:.6}", value));
        row.insert("timestamp".to_string(), chrono::Local::now().to_rfc3339());
        
        buffer.push(row);
        
        if buffer.len() >= self.buffer_size {
            drop(buffer);
            self.write_buffer().await?;
        }
        
        Ok(())
    }
    
    async fn log_histogram(&self, _name: &str, _values: &[f32], _step: usize) -> Result<()> {
        // CSV doesn't support histograms directly
        Ok(())
    }
    
    async fn log_image(&self, _name: &str, _image: &[u8], _step: usize) -> Result<()> {
        // CSV doesn't support images
        Ok(())
    }
    
    async fn log_text(&self, name: &str, text: &str, step: usize) -> Result<()> {
        self.log_scalar(&format!("{}_text", name), 0.0, step).await
    }
    
    async fn flush(&self) -> Result<()> {
        self.write_buffer().await
    }
    
    async fn close(&self) -> Result<()> {
        self.flush().await
    }
}

/// Multi-logger that writes to multiple backends
pub struct MultiLogger {
    loggers: Vec<Box<dyn MetricLogger>>,
}

impl MultiLogger {
    pub fn new(loggers: Vec<Box<dyn MetricLogger>>) -> Self {
        Self { loggers }
    }
}

#[async_trait::async_trait]
impl MetricLogger for MultiLogger {
    async fn log_scalar(&self, name: &str, value: f32, step: usize) -> Result<()> {
        for logger in &self.loggers {
            logger.log_scalar(name, value, step).await?;
        }
        Ok(())
    }
    
    async fn log_histogram(&self, name: &str, values: &[f32], step: usize) -> Result<()> {
        for logger in &self.loggers {
            logger.log_histogram(name, values, step).await?;
        }
        Ok(())
    }
    
    async fn log_image(&self, name: &str, image: &[u8], step: usize) -> Result<()> {
        for logger in &self.loggers {
            logger.log_image(name, image, step).await?;
        }
        Ok(())
    }
    
    async fn log_text(&self, name: &str, text: &str, step: usize) -> Result<()> {
        for logger in &self.loggers {
            logger.log_text(name, text, step).await?;
        }
        Ok(())
    }
    
    async fn flush(&self) -> Result<()> {
        for logger in &self.loggers {
            logger.flush().await?;
        }
        Ok(())
    }
    
    async fn close(&self) -> Result<()> {
        for logger in &self.loggers {
            logger.close().await?;
        }
        Ok(())
    }
}

/// Statistics for histogram values
#[derive(Debug, Clone)]
struct Statistics {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
    median: f32,
}

fn calculate_statistics(values: &[f32]) -> Statistics {
    if values.is_empty() {
        return Statistics {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
        };
    }
    
    let sum: f32 = values.iter().sum();
    let mean = sum / values.len() as f32;
    
    let variance: f32 = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    let std = variance.sqrt();
    
    let min = values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0);
    let max = values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0);
    
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    
    Statistics { mean, std, min, max, median }
}

/// Progress tracker
pub struct ProgressTracker {
    total_steps: usize,
    current_step: Arc<RwLock<usize>>,
    start_time: std::time::Instant,
    last_update: Arc<RwLock<std::time::Instant>>,
    update_interval: std::time::Duration,
}

impl ProgressTracker {
    pub fn new(total_steps: usize) -> Self {
        let now = std::time::Instant::now();
        Self {
            total_steps,
            current_step: Arc::new(RwLock::new(0)),
            start_time: now,
            last_update: Arc::new(RwLock::new(now)),
            update_interval: std::time::Duration::from_millis(100),
        }
    }
    
    pub async fn update(&self, step: usize) {
        let mut current = self.current_step.write().await;
        *current = step;
        
        let mut last_update = self.last_update.write().await;
        let now = std::time::Instant::now();
        
        if now.duration_since(*last_update) >= self.update_interval {
            let progress = step as f32 / self.total_steps as f32;
            let elapsed = now.duration_since(self.start_time);
            let eta = if step > 0 {
                elapsed.mul_f32((self.total_steps - step) as f32 / step as f32)
            } else {
                std::time::Duration::from_secs(0)
            };
            
            let bar_width = 40;
            let filled = (progress * bar_width as f32) as usize;
            
            print!("\r[");
            print!("{}", "=".repeat(filled));
            print!("{}", " ".repeat(bar_width - filled));
            print!("] {}/{} ({:.1}%) - ETA: {}",
                step,
                self.total_steps,
                progress * 100.0,
                format_duration(eta)
            );
            
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
            
            *last_update = now;
        }
    }
    
    pub fn finish(&self) {
        println!();
    }
}

fn format_duration(duration: std::time::Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    
    if hours > 0 {
        format!("{}h {}m {}s", hours, mins, secs)
    } else if mins > 0 {
        format!("{}m {}s", mins, secs)
    } else {
        format!("{}s", secs)
    }
}