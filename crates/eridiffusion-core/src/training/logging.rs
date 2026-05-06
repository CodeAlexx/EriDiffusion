//! JSONL metrics writer. One line per training step.
//!
//! Ported verbatim from flame-diffusion/src/logging.rs (2026-05-05).
//! Compatible with downstream parsing by jq, pandas, or SerenityBoard's JSON
//! ingest. NOT tensorboard — project rule forbids tensorboard.

use flame_core::{Error, Result};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};

pub struct MetricsWriter {
    writer: BufWriter<File>,
}

impl MetricsWriter {
    pub fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Io(format!("metrics dir {}: {e}", parent.display())))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| Error::Io(format!("open metrics {}: {e}", path.display())))?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn log_step(&mut self, record: &StepRecord) -> Result<()> {
        let line = format!(
            "{{\"step\":{},\"epoch\":{},\"loss\":{:.6},\"grad_norm\":{:.6},\"lr\":{:.6e},\"timestep\":{:.6},\"bs\":{},\"grad_accum\":{},\"bucket_h\":{},\"bucket_w\":{},\"bucket_c\":{},\"text_seq\":{}}}\n",
            record.step,
            record.epoch,
            record.loss,
            record.grad_norm,
            record.lr,
            record.timestep,
            record.batch_size,
            record.grad_accum,
            record.bucket_h,
            record.bucket_w,
            record.bucket_c,
            record.text_seq,
        );
        self.writer
            .write_all(line.as_bytes())
            .map_err(|e| Error::Io(format!("metrics write: {e}")))?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| Error::Io(format!("metrics flush: {e}")))
    }
}

#[derive(Debug, Clone)]
pub struct StepRecord {
    pub step: usize,
    pub epoch: u64,
    pub loss: f32,
    pub grad_norm: f32,
    pub lr: f32,
    pub timestep: f32,
    pub batch_size: usize,
    pub grad_accum: usize,
    pub bucket_h: usize,
    pub bucket_w: usize,
    pub bucket_c: usize,
    pub text_seq: usize,
}
