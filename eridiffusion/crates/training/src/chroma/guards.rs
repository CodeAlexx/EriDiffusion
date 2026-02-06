use eridiffusion_core::{Error, Result};
use flame_core::{DType, Tensor};

#[inline]
pub fn guard(tag: &str, t: &Tensor) -> Result<()> {
    if let Ok(v) = t.to_dtype(DType::F32).and_then(|u| u.to_vec()) {
        if v.iter().any(|x| !x.is_finite()) {
            if std::env::var("TRIPWIRES").ok().map(|v| v != "0").unwrap_or(false) {
                let mut minv = f32::INFINITY;
                let mut maxv = f32::NEG_INFINITY;
                let mut sum = 0.0f32;
                let mut n = 0usize;
                for &x in &v {
                    if x.is_finite() {
                        minv = minv.min(x);
                        maxv = maxv.max(x);
                        sum += x;
                        n += 1;
                    }
                }
                let mean = if n > 0 { sum / n as f32 } else { f32::NAN };
                println!(
                    "[nan] {} non-finite | min={:.3e} max={:.3e} mean={:.3e}",
                    tag, minv, maxv, mean
                );
            }
            return Err(Error::Training(format!("non-finite {}", tag)));
        }
    }
    Ok(())
}

#[inline]
pub fn log_minmax(tag: &str, t: &Tensor) -> Result<()> {
    if std::env::var("TRIPWIRES").ok().map(|v| v != "0").unwrap_or(false) {
        if let Ok(v) = t.to_dtype(DType::F32).and_then(|u| u.to_vec()) {
            if let (Some(minv), Some(maxv)) = (
                v.iter()
                    .filter(|x| x.is_finite())
                    .cloned()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
                v.iter()
                    .filter(|x| x.is_finite())
                    .cloned()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
            ) {
                println!("[dbg] {} min={:.2e} max={:.2e}", tag, minv, maxv);
            }
        }
    }
    Ok(())
}
