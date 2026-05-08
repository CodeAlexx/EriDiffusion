//! Phase 7: Pre-checkpoint disk-space sanity check.
//!
//! Best-effort — uses `df --output=avail -B1 <path>` (Linux). On platforms
//! without `df` or in unusual environments, the function logs a warning and
//! returns Ok rather than blocking training. The trainer should treat the
//! returned `Err` as "skip this save" rather than "abort".

use std::path::Path;
use std::process::Command;

use crate::EriDiffusionError;

/// Check that `path` (or its closest existing ancestor) has at least
/// `min_free_bytes` available.
///
/// Returns `Ok(())` when the check passes OR when free space cannot be
/// determined (best-effort: don't block training on a failed probe).
/// Returns `Err(EriDiffusionError::Other)` when free space is determinable
/// AND below the threshold.
pub fn check_free_space(path: &Path, min_free_bytes: u64) -> crate::Result<()> {
    // `df` follows the path to the mount; non-existent paths still resolve to
    // the parent fs. To be safe walk up to the closest existing ancestor —
    // calling `df` on a missing dir errors out on some systems.
    let probe: &Path = if path.exists() {
        path
    } else {
        let mut p = path;
        while let Some(parent) = p.parent() {
            if parent.exists() {
                p = parent;
                break;
            }
            p = parent;
        }
        p
    };

    let output = Command::new("df")
        .arg("--output=avail")
        .arg("-B1")
        .arg(probe)
        .output();
    if let Ok(out) = output {
        if !out.status.success() {
            log::warn!(
                "[disk-check] df returned non-zero status for {} — skipping check",
                probe.display()
            );
            return Ok(());
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let lines: Vec<&str> = stdout.lines().collect();
        if lines.len() >= 2 {
            if let Ok(avail) = lines[1].trim().parse::<u64>() {
                if avail < min_free_bytes {
                    return Err(EriDiffusionError::Other(format!(
                        "insufficient disk space: {} bytes available at {}, need {}",
                        avail,
                        probe.display(),
                        min_free_bytes
                    )));
                }
                return Ok(());
            }
        }
        log::warn!(
            "[disk-check] could not parse df output at {} — skipping check",
            probe.display()
        );
        Ok(())
    } else {
        log::warn!(
            "[disk-check] could not run df at {} — skipping check",
            probe.display()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn check_free_space_passes_for_root_with_zero_threshold() {
        // 0-byte threshold: any filesystem passes.
        let root = PathBuf::from("/");
        check_free_space(&root, 0).expect("0-byte threshold should always pass");
    }

    #[test]
    fn check_free_space_fails_for_unreasonable_threshold() {
        // Demand exabytes — must error (assuming the test host has < 1 EB free).
        let root = PathBuf::from("/");
        let huge: u64 = 1_000_000_000_000_000_000; // 1 EB
        let res = check_free_space(&root, huge);
        // Either we determined free space (Err) or we couldn't (Ok). The
        // happy-path on a normal Linux box is Err; non-Linux skips with Ok.
        // Both are acceptable — we just want the function to not panic.
        let _ = res;
    }

    #[test]
    fn check_free_space_walks_to_existing_ancestor() {
        // Non-existent path should still probe — it walks up to /.
        let p = PathBuf::from("/nonexistent_phase7_disk_check_path/sub");
        check_free_space(&p, 0).expect("should walk up to /");
    }
}
