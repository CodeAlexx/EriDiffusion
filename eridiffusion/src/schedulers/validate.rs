use anyhow::{ensure, Result};

/// Validate Karras/Euler schedule parameters and the generated sigma schedule.
/// This is self-contained (generates the schedule locally) to avoid cross-module deps.
pub fn validate_euler_karras_cfg(
    steps: usize,
    sigma_min: f32,
    sigma_max: f32,
    rho: f32,
) -> Result<()> {
    ensure!(steps >= 2, "steps must be >= 2");
    ensure!(sigma_min > 0.0, "sigma_min must be > 0");
    ensure!(sigma_max > sigma_min, "sigma_max must be > sigma_min");
    ensure!(rho > 0.0, "rho must be > 0");

    // Generate Karras sigmas (monotonically descending from sigma_max -> sigma_min)
    let sigmas = karras_sigmas(steps, sigma_min, sigma_max, rho);

    // Monotonic descending check
    ensure!(sigmas.windows(2).all(|w| w[0] >= w[1]), "sigmas must be monotonically descending");

    // Bounds check
    let lo = sigmas.iter().copied().fold(f32::INFINITY, |a, x| a.min(x));
    let hi = sigmas.iter().copied().fold(f32::NEG_INFINITY, |a, x| a.max(x));
    ensure!(lo >= sigma_min - 1e-6 && hi <= sigma_max + 1e-6, "sigmas out of bounds");

    Ok(())
}

fn karras_sigmas(n: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Vec<f32> {
    assert!(n >= 2);
    let inv_rho = 1.0 / rho;
    (0..n)
        .map(|i| i as f32 / (n as f32 - 1.0))
        .map(|t| {
            let v =
                sigma_max.powf(inv_rho) + t * (sigma_min.powf(inv_rho) - sigma_max.powf(inv_rho));
            v.powf(rho)
        })
        .collect()
}
