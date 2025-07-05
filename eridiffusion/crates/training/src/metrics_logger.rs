//! Metrics logger for training

use std::path::Path;
use std::fs::File;
use std::io::Write;
use eridiffusion_core::{Result, Error};

/// Simple metrics logger that writes to CSV
pub struct MetricsLogger {
    file: File,
    first_write: bool,
}

impl MetricsLogger {
    /// Create new metrics logger
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Create parent directory if needed
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let file = File::create(path)?;
        Ok(Self {
            file,
            first_write: true,
        })
    }
    
    /// Log a scalar value
    pub fn log_scalar(&mut self, name: &str, value: f32, step: usize) -> Result<()> {
        // Write header on first write
        if self.first_write {
            writeln!(self.file, "step,metric,value")?;
            self.first_write = false;
        }
        
        writeln!(self.file, "{},{},{}", step, name, value)?;
        self.file.flush()?;
        Ok(())
    }
}