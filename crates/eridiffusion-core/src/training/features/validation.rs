//! Validation loop — held-out dataset with periodic eval-loss tracking, logged
//! to the SerenityBoard.
//!
//! Phase 2 — wired (not skeleton). Each trainer drives the actual eval forward
//! pass because forward signatures are model-specific. This module owns:
//!
//!   - discovery of the held-out cache directory at startup
//!   - the cadence check (`should_run`)
//!   - the cache file list (caller iterates it under `AutogradContext::no_grad()`)
//!
//! Phase target: 2
//! Config flags:
//!   - `validation_dataset_dir: Option<PathBuf>` (default None)
//!   - `validation_every_steps: u64` (default 0 = disabled)
//! Reference: SimpleTuner `helpers/training/validate.py`.

use crate::Result;
use std::path::{Path, PathBuf};

/// Validation harness. Holds the cached eval cache file list and the cadence.
///
/// Construction discovers `*.safetensors` files under `dir` (sorted).
/// `every_steps == 0` is a sentinel meaning "feature off" — callers should
/// not construct a `ValidationLoop` when the feature is disabled.
pub struct ValidationLoop {
    pub cache_files: Vec<PathBuf>,
    pub every_steps: u64,
}

impl ValidationLoop {
    /// Discover the held-out cache directory. Errors when `dir` is missing or
    /// contains zero `.safetensors` files (we'd rather fail-fast than silently
    /// skip eval).
    pub fn new(dir: &Path, every_steps: u64) -> Result<Self> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            crate::EriDiffusionError::Data(format!(
                "validation dir {}: {e}",
                dir.display()
            ))
        })?;
        let mut files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("safetensors"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(crate::EriDiffusionError::Data(format!(
                "validation dir {} has no .safetensors files",
                dir.display()
            )));
        }
        Ok(Self {
            cache_files: files,
            every_steps,
        })
    }

    /// Whether validation should run at the given step.
    ///
    /// Convention: `step` is the 1-based step number that just COMPLETED.
    /// Callers pass `step + 1` from their 0-based loop.
    pub fn should_run(&self, step: usize) -> bool {
        self.every_steps > 0 && step > 0 && (step as u64) % self.every_steps == 0
    }

    pub fn len(&self) -> usize {
        self.cache_files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache_files.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cadence_zero_is_disabled() {
        // Build a fake ValidationLoop without touching disk.
        let v = ValidationLoop {
            cache_files: vec![PathBuf::from("fake.safetensors")],
            every_steps: 0,
        };
        assert!(!v.should_run(1));
        assert!(!v.should_run(100));
        assert!(!v.should_run(0));
    }

    #[test]
    fn cadence_fires_on_multiples() {
        let v = ValidationLoop {
            cache_files: vec![PathBuf::from("a.safetensors")],
            every_steps: 50,
        };
        assert!(!v.should_run(0));
        assert!(!v.should_run(1));
        assert!(!v.should_run(49));
        assert!(v.should_run(50));
        assert!(!v.should_run(51));
        assert!(v.should_run(100));
        assert!(v.should_run(150));
    }

    #[test]
    fn missing_dir_errors() {
        let result = ValidationLoop::new(Path::new("/no/such/dir/zzz"), 50);
        assert!(result.is_err());
    }

    #[test]
    fn empty_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        let result = ValidationLoop::new(dir.path(), 50);
        assert!(result.is_err());
    }
}
