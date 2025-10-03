use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

use eridiffusion_core::Result;

pub struct TelemetryCsv {
    f: File,
    num_layers: usize,
}

impl TelemetryCsv {
    pub fn new<P: AsRef<Path>>(path: P, num_layers: usize) -> Result<Self> {
        let mut f = OpenOptions::new().create(true).append(true).open(path)?;
        let meta = f.metadata()?;
        if meta.len() == 0 {
            // Base columns (perf + loss)
            write!(&mut f, "step,fw_sec,bw_sec,h2d_mb_s,d2h_mb_s,gpu_mem_peak,tokens_per_s,grad_norm,loss,fused")?;
            // Routing columns (optional, filled with zeros if unused)
            write!(&mut f, ",route_lambda,route_loss,kept_avg,shelves_mb,route_miss")?;
            for i in 0..num_layers {
                write!(&mut f, ",kept_frac_L{}", i)?;
            }
            writeln!(&mut f)?;
        }
        Ok(Self { f, num_layers })
    }

    /// Write a consolidated perf + routing row. For non-routed runs, pass zeros for routing fields.
    pub fn write(
        &mut self,
        step: u64,
        fw_sec: f32,
        bw_sec: f32,
        h2d_mb_s: f32,
        d2h_mb_s: f32,
        gpu_mem_peak: f32,
        tokens_per_s: f32,
        grad_norm: f32,
        loss: f32,
        fused: bool,
        _route_lambda: f32,
        _route_loss: f32,
        _kept_avg: f32,
        kept_frac: &[f32],
        _shelves_mb: f32,
        _route_miss: u32,
    ) -> Result<()> {
        // EXPECT_STEP_SEC guard
        let total_sec = fw_sec + bw_sec;
        if let Ok(expect) = std::env::var("EXPECT_STEP_SEC") {
            if let Ok(max_sec) = expect.parse::<f32>() {
                if total_sec > max_sec {
                    return Err(eridiffusion_core::Error::InvalidInput(format!(
                        "step {} exceeded EXPECT_STEP_SEC: {:.3}s > {:.3}s",
                        step, total_sec, max_sec
                    )));
                }
            }
        }
        write!(
            &mut self.f,
            "{},{:.6},{:.6},{:.3},{:.3},{:.1},{:.2},{:.5},{:.6},{}",
            step,
            fw_sec,
            bw_sec,
            h2d_mb_s,
            d2h_mb_s,
            gpu_mem_peak,
            tokens_per_s,
            grad_norm,
            loss,
            if fused { 1 } else { 0 }
        )?;
        for i in 0..self.num_layers {
            let v = *kept_frac.get(i).unwrap_or(&0.0);
            write!(&mut self.f, ",{:.6}", v)?;
        }
        writeln!(&mut self.f)?;
        Ok(())
    }
}

/// Helpers
pub fn fused_runtime_enabled() -> bool {
    match std::env::var("ERID_FUSE").ok().as_deref() {
        Some("1") | Some("on") | Some("true") | Some("ON") | Some("True") => true,
        _ => false,
    }
}

pub fn anomaly_loss(loss: f32) -> bool {
    !loss.is_finite()
}
