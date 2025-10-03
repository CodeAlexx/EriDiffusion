use crate::{Result, Error};
use flame_core::{Tensor, DType};

pub trait TensorDebugExt {
    fn debug_check(&self, tag: &str) -> Result<()>;
}

impl TensorDebugExt for Tensor {
    fn debug_check(&self, tag: &str) -> Result<()> {
        // Env-gated to avoid overhead on hot paths
        let enabled = std::env::var("TRIPWIRES").ok().map(|v| v != "0").unwrap_or(false);
        if !enabled { return Ok(()); }

        // Conservative: host read to check finiteness (debug-only)
        // If desired, replace with a GPU-side reduction once available.
        let v = self.to_dtype(DType::F32).map_err(Error::from)?.to_vec().map_err(Error::from)?;
        if v.iter().any(|x| !x.is_finite()) {
            // Best-effort stats
            let mut minv = f32::INFINITY;
            let mut maxv = f32::NEG_INFINITY;
            let mut sum = 0.0f32;
            let mut n = 0usize;
            for &x in &v {
                if x.is_finite() {
                    if x < minv { minv = x; }
                    if x > maxv { maxv = x; }
                    sum += x;
                    n += 1;
                }
            }
            let mean = if n > 0 { sum / n as f32 } else { f32::NAN };
            eprintln!(
                "[tensor-check] {} non-finite | min={:.3e} max={:.3e} mean={:.3e}",
                tag, minv, maxv, mean
            );
            return Err(Error::Training(format!("non-finite {tag}")));
        }
        Ok(())
    }
}
