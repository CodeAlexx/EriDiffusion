//! Multi-backend dataset — sample weighted across N concept directories so
//! mixing e.g. character LoRA + style LoRA doesn't collapse to whichever
//! directory is larger.
//!
//! Phase 2 — wired (not skeleton). Trainer holds a `MultiBackend` (when CLI
//! flags request it) and calls `pick(rng)` at every-step file selection time.
//! When unset, single-backend code path is unchanged → byte-identical.
//!
//! Phase target: 2
//! Config flags:
//!   - `multi_backend_weights: Vec<f32>` (default empty = single backend)
//!   - `--multi-backend-cache-dirs <DIR1> <DIR2> ...` (CLI; matches weights len)
//!
//! Reference: SimpleTuner `helpers/data_backend/factory.py` weighted sampling.

use crate::Result;
use rand::Rng;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A weighted blend of cached `.safetensors` cache directories.
///
/// `weights` is normalized to sum to 1.0 internally; callers pass raw weights
/// (e.g. `[0.7, 0.3]` or `[7.0, 3.0]` — equivalent).
pub struct MultiBackend {
    /// One sorted list of `*.safetensors` paths per backend.
    pub backends: Vec<Vec<PathBuf>>,
    /// Normalized categorical weights, one per backend (sum to 1.0).
    pub weights: Vec<f32>,
}

impl MultiBackend {
    /// Construct from N cache directories with N corresponding weights.
    /// Errors if:
    ///   - `dirs.len() != weights.len()`
    ///   - any weight is non-positive
    ///   - any directory is missing or has zero `.safetensors` files
    pub fn new(dirs: &[PathBuf], weights: &[f32]) -> Result<Self> {
        if dirs.len() != weights.len() {
            return Err(crate::EriDiffusionError::Data(format!(
                "multi-backend: {} dirs vs {} weights — must match",
                dirs.len(),
                weights.len()
            )));
        }
        if dirs.is_empty() {
            return Err(crate::EriDiffusionError::Data(
                "multi-backend: zero backends — set --multi-backend-cache-dirs and --multi-backend-weights".into(),
            ));
        }
        for (i, w) in weights.iter().enumerate() {
            if !(*w > 0.0) || !w.is_finite() {
                return Err(crate::EriDiffusionError::Data(format!(
                    "multi-backend: weight[{i}]={w} must be > 0 and finite"
                )));
            }
        }

        let mut backends: Vec<Vec<PathBuf>> = Vec::with_capacity(dirs.len());
        for d in dirs {
            let entries = std::fs::read_dir(d).map_err(|e| {
                crate::EriDiffusionError::Data(format!(
                    "multi-backend dir {}: {e}",
                    d.display()
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
                    "multi-backend dir {} has no .safetensors files",
                    d.display()
                )));
            }
            backends.push(files);
        }

        let sum: f32 = weights.iter().sum();
        let normalized: Vec<f32> = weights.iter().map(|w| w / sum).collect();
        Ok(Self {
            backends,
            weights: normalized,
        })
    }

    /// Pick one cached sample path. Two-stage draw:
    ///   1. Categorical over backends with `weights`
    ///   2. Uniform within the chosen backend's file list
    pub fn pick<R: Rng>(&self, rng: &mut R) -> &PathBuf {
        let r: f32 = rng.gen();
        let mut acc = 0.0_f32;
        let mut chosen = self.backends.len() - 1;
        for (i, w) in self.weights.iter().enumerate() {
            acc += *w;
            if r < acc {
                chosen = i;
                break;
            }
        }
        let backend = &self.backends[chosen];
        let idx = rng.gen_range(0..backend.len());
        &backend[idx]
    }

    /// Phase 6: construct from N dirs + N weights + N repeats. The effective
    /// categorical weight per backend is `weights[i] * repeats[i]`. A backend
    /// with `repeats=3` is sampled 3× more often than one with `repeats=1`
    /// at the same `weights[i]`. Equivalent to passing pre-multiplied weights
    /// into `new`; this entry point is sugar for the `--multi-backend-repeats`
    /// CLI surface.
    ///
    /// Errors if `dirs.len() != weights.len() != repeats.len()` or any
    /// `repeats[i] == 0` (a zero-repeat backend would have zero effective
    /// weight and could never be sampled — surface as an error rather than
    /// silently dropping the backend).
    pub fn new_with_repeats(
        dirs: &[PathBuf],
        weights: &[f32],
        repeats: &[u32],
    ) -> Result<Self> {
        if dirs.len() != weights.len() || weights.len() != repeats.len() {
            return Err(crate::EriDiffusionError::Data(format!(
                "multi-backend with repeats: {} dirs vs {} weights vs {} repeats — must match",
                dirs.len(),
                weights.len(),
                repeats.len()
            )));
        }
        if repeats.iter().any(|&r| r == 0) {
            return Err(crate::EriDiffusionError::Data(
                "multi-backend with repeats: every repeats[i] must be ≥ 1".into(),
            ));
        }
        let combined: Vec<f32> = weights
            .iter()
            .zip(repeats.iter())
            .map(|(w, &r)| *w * r as f32)
            .collect();
        Self::new(dirs, &combined)
    }

    /// Total file count across all backends. Used for startup diagnostics.
    pub fn total_files(&self) -> usize {
        self.backends.iter().map(|b| b.len()).sum()
    }

    /// Per-backend file count, indexed by backend.
    pub fn per_backend_counts(&self) -> Vec<usize> {
        self.backends.iter().map(|b| b.len()).collect()
    }

    /// Count latent (h, w) tuples across ALL backends. Returns one map per
    /// backend so callers can render a per-backend report.
    ///
    /// Best-effort: any file whose latent header can't be parsed is skipped.
    pub fn bucket_distribution(&self) -> Vec<HashMap<(usize, usize), usize>> {
        let mut out = Vec::with_capacity(self.backends.len());
        for backend in &self.backends {
            let mut sizes: HashMap<(usize, usize), usize> = HashMap::new();
            for f in backend {
                if let Some((h, w)) = read_latent_hw(f) {
                    *sizes.entry((h, w)).or_default() += 1;
                }
            }
            out.push(sizes);
        }
        out
    }
}

/// Parse the safetensors header for a sample to recover the latent (H, W).
/// Mirrors `bucket_dataset::read_safetensors_header` but only reads the
/// `latent` shape (skipping rank checks). Returns None on any error.
pub fn read_latent_hw(path: &Path) -> Option<(usize, usize)> {
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, SeekFrom};

    let f = File::open(path).ok()?;
    let mut reader = BufReader::new(f);
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes).ok()?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len > 16 * 1024 * 1024 {
        return None;
    }
    let mut buf = vec![0u8; header_len];
    reader.seek(SeekFrom::Start(8)).ok()?;
    reader.read_exact(&mut buf).ok()?;
    let header = String::from_utf8(buf).ok()?;
    let json: serde_json::Value = serde_json::from_str(&header).ok()?;
    let obj = json.as_object()?;
    let latent = obj.get("latent")?.as_object()?;
    let arr = latent.get("shape")?.as_array()?;
    let dims: Vec<usize> = arr
        .iter()
        .map(|v| v.as_u64().map(|x| x as usize))
        .collect::<Option<Vec<_>>>()?;
    if dims.len() < 2 {
        return None;
    }
    let h = dims[dims.len() - 2];
    let w = dims[dims.len() - 1];
    Some((h, w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_dir_with_files(n: usize) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for i in 0..n {
            let p = d.path().join(format!("sample_{i:03}.safetensors"));
            // Minimal placeholder; bucket_distribution / dataset loaders will
            // ignore files they can't parse; pick() doesn't read content.
            std::fs::write(p, b"placeholder").unwrap();
        }
        d
    }

    #[test]
    fn mismatched_lengths_err() {
        let d = make_dir_with_files(3);
        let dirs = vec![d.path().to_path_buf()];
        let weights = vec![1.0, 2.0];
        assert!(MultiBackend::new(&dirs, &weights).is_err());
    }

    #[test]
    fn nonpositive_weight_errs() {
        let d = make_dir_with_files(3);
        let dirs = vec![d.path().to_path_buf()];
        let weights = vec![0.0];
        assert!(MultiBackend::new(&dirs, &weights).is_err());
        let weights = vec![-1.0];
        assert!(MultiBackend::new(&dirs, &weights).is_err());
    }

    #[test]
    fn weights_normalize() {
        let d1 = make_dir_with_files(2);
        let d2 = make_dir_with_files(2);
        let dirs = vec![d1.path().to_path_buf(), d2.path().to_path_buf()];
        let mb = MultiBackend::new(&dirs, &[7.0, 3.0]).unwrap();
        assert!((mb.weights[0] - 0.7).abs() < 1e-6);
        assert!((mb.weights[1] - 0.3).abs() < 1e-6);
        assert_eq!(mb.total_files(), 4);
    }

    #[test]
    fn extreme_weights_pick_dominates() {
        let d1 = make_dir_with_files(2);
        let d2 = make_dir_with_files(2);
        let dirs = vec![d1.path().to_path_buf(), d2.path().to_path_buf()];
        let mb = MultiBackend::new(&dirs, &[1000.0, 1.0]).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let mut from_first = 0;
        for _ in 0..1000 {
            let p = mb.pick(&mut rng);
            if p.starts_with(d1.path()) {
                from_first += 1;
            }
        }
        // Should be ~999/1000 on average — assert strictly > 950.
        assert!(from_first > 950, "got {from_first}/1000 from first backend");
    }

    #[test]
    fn repeats_multiply_into_weight() {
        let d1 = make_dir_with_files(2);
        let d2 = make_dir_with_files(2);
        let dirs = vec![d1.path().to_path_buf(), d2.path().to_path_buf()];
        // Weights 1.0/1.0 with repeats 3/1 → effective 3.0/1.0 → 0.75/0.25.
        let mb =
            MultiBackend::new_with_repeats(&dirs, &[1.0, 1.0], &[3, 1]).unwrap();
        assert!((mb.weights[0] - 0.75).abs() < 1e-6);
        assert!((mb.weights[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn zero_repeat_errs() {
        let d1 = make_dir_with_files(2);
        let dirs = vec![d1.path().to_path_buf()];
        let r = MultiBackend::new_with_repeats(&dirs, &[1.0], &[0]);
        assert!(r.is_err());
    }

    #[test]
    fn repeats_length_mismatch_errs() {
        let d1 = make_dir_with_files(2);
        let dirs = vec![d1.path().to_path_buf()];
        let r = MultiBackend::new_with_repeats(&dirs, &[1.0], &[1, 2]);
        assert!(r.is_err());
    }

    #[test]
    fn empty_backend_dir_errs() {
        let d = tempfile::tempdir().unwrap();
        let dirs = vec![d.path().to_path_buf()];
        let weights = vec![1.0];
        assert!(MultiBackend::new(&dirs, &weights).is_err());
    }
}
