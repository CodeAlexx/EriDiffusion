use anyhow::Result;
use flame_core::Tensor;

/// Karras sigma schedule for EDM-style sampling (monotonically decreasing)
pub struct Karras {
    sigmas: Vec<f32>,
    timesteps: Vec<i32>,
    sigma_max: f32,
}

impl Karras {
    /// Create Karras schedule with typical SDXL ranges
    pub fn new(steps: usize) -> Self {
        let sigma_min = 0.03f32;
        let sigma_max = 14.6f32;
        let rho = 7.0f32;
        let n = steps as i32;
        let mut sigmas = Vec::with_capacity(steps);
        for i in 0..steps {
            let t = i as f32 / ((steps - 1).max(1) as f32);
            let sigma = (sigma_max.powf(1.0/rho) * (1.0 - t) + sigma_min.powf(1.0/rho) * t).powf(rho);
            sigmas.push(sigma);
        }
        // Ensure strictly decreasing
        sigmas.sort_by(|a,b| b.partial_cmp(a).unwrap());
        let timesteps = (0..steps as i32).rev().collect();
        Self { sigma_max, sigmas, timesteps }
    }
    pub fn sigma_max(&self) -> f32 { self.sigma_max }
    pub fn iter(&self) -> impl Iterator<Item=(f32, i32)> + '_ { self.sigmas.iter().cloned().zip(self.timesteps.iter().cloned()) }
    /// Find the next sigma below the given current value; returns 0.0 if none.
    pub fn next_sigma(&self, sigma: f32) -> f32 {
        for (i, &s) in self.sigmas.iter().enumerate() {
            if (s - sigma).abs() < 1e-6 {
                if i + 1 < self.sigmas.len() { return self.sigmas[i+1]; } else { return 0.0; }
            }
            if s < sigma { return s; }
        }
        0.0
    }
}

/// Euler-A update for epsilon prediction in NHWC layout
/// x_{next} = x + (sigma_next - sigma) * eps
pub fn euler_a_step(x: Tensor, eps: Tensor, sigma: f32, sched: &Karras) -> Result<Tensor> {
    let sigma_next = sched.next_sigma(sigma);
    let ds = sigma_next - sigma; // negative
    let step = eps.mul_scalar(ds)?;
    Ok(x.add(&step)?)
}
