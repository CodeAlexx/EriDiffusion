//! RAdam Schedule-Free optimizer implementation (FLAME-only)
//! Based on the Schedule-Free Learning framework

use std::collections::HashMap;

use anyhow::Result;
use flame_core::{Parameter, Tensor};
pub struct RAdamScheduleFree {
    vars: Vec<Parameter>,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    warmup_steps: usize,
    r: f64,
    k: f64,
    step_count: usize,
    first_moments: HashMap<String, Tensor>,
    second_moments: HashMap<String, Tensor>,
    z_vars: HashMap<String, Tensor>, // Schedule-free auxiliary variables stored as tensors
}

impl RAdamScheduleFree {
    /// Create a new RAdam Schedule-Free optimizer
    pub fn new(
        vars: Vec<Parameter>,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        warmup_steps: usize,
        r: f64,
    ) -> Result<Self> {
        let mut first_moments = HashMap::new();
        let mut second_moments = HashMap::new();
        let mut z_vars = HashMap::new();

        // Initialize moments and auxiliary variables
        for (idx, var) in vars.iter().enumerate() {
            let name = format!("var_{}", idx);
            let vt = var.tensor()?;
            let zeros = Tensor::zeros(vt.shape().clone(), vt.device().clone())?;
            first_moments.insert(name.clone(), zeros.clone());
            second_moments.insert(name.clone(), zeros);

            // Initialize z = x (auxiliary variable)
            let z = vt;
            z_vars.insert(name, z);
        }

        Ok(Self {
            vars,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            warmup_steps,
            r,
            k: 0.0,
            step_count: 0,
            first_moments,
            second_moments,
            z_vars,
        })
    }

    /// Switch to evaluation mode (use z variables)
    pub fn eval(&mut self) -> Result<()> {
        for (idx, var) in self.vars.iter().enumerate() {
            let name = format!("var_{}", idx);
            if let Some(z_var) = self.z_vars.get(&name) {
                var.set_data(z_var.clone())?;
            }
        }
        Ok(())
    }

    /// Switch to training mode (use x variables)
    pub fn train(&mut self) -> Result<()> {
        // In training mode, we use the regular variables
        // The z variables are updated during the step
        Ok(())
    }
}

impl RAdamScheduleFree {
    /// Perform optimization step
    pub fn step(&mut self, _params: &[&Tensor], grads: &[Tensor]) -> Result<()> {
        self.step_count += 1;
        let step = self.step_count;

        // Update k for schedule-free
        self.k = self.k * self.beta2 + (1.0 - self.beta2);

        // Bias correction terms
        let bias_correction1 = 1.0 - self.beta1.powi(step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(step as i32);

        // RAdam specific - compute rectification term
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;
        let rho_t = rho_inf - 2.0 * (step as f64) * self.beta2.powi(step as i32) / bias_correction2;

        // Compute adaptive learning rate
        let lr = if rho_t > 5.0 {
            // Variance is tractable
            let l_t =
                ((1.0 - self.beta2.powi(step as i32)) * (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                    / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                    .sqrt();
            self.learning_rate * l_t / bias_correction1
        } else {
            // Variance is not tractable, use simple lr
            self.learning_rate / bias_correction1
        };

        // Schedule-free learning rate
        let c_t = self.r / (self.k.powf(1.0 / 3.0));
        let y_lr = c_t.min(lr);

        for (idx, (var, grad)) in self.vars.iter().zip(grads.iter()).enumerate() {
            let name = format!("var_{}", idx);
            // Get moments
            let m = self.first_moments.get_mut(&name).unwrap();
            let v = self.second_moments.get_mut(&name).unwrap();
            let z_var = self.z_vars.get_mut(&name).unwrap();

            // Apply weight decay to gradient
            let grad = if self.weight_decay > 0.0 {
                grad.add(&var.tensor()?.affine(self.weight_decay as f32, 0.0f32)?)?
            } else {
                grad.clone()
            };

            // Update biased first moment estimate
            *m = m
                .affine(self.beta1 as f32, 0.0f32)?
                .add(&grad.affine((1.0 - self.beta1) as f32, 0.0f32)?)?;

            // Update biased second raw moment estimate
            let grad_sq = grad.square()?;
            *v = v
                .affine(self.beta2 as f32, 0.0f32)?
                .add(&grad_sq.affine((1.0 - self.beta2) as f32, 0.0f32)?)?;

            // Compute update
            let update = if rho_t > 5.0 {
                // Use adaptive momentum
                let v_sqrt = v.sqrt()?.affine((1.0 / bias_correction2.sqrt()) as f32, 0.0f32)?;
                m.affine((1.0 / bias_correction1) as f32, 0.0f32)?
                    .div(&v_sqrt.affine(1.0f32, self.eps as f32)?)?
            } else {
                // Simple SGD-like update
                m.affine((1.0 / bias_correction1) as f32, 0.0f32)?
            };

            // Schedule-free update
            // z = z - y_lr * update
            let z_update = update.affine(-(y_lr as f32), 0.0f32)?;
            *z_var = z_var.add(&z_update)?;

            // Interpolate between x and z for training
            // x = (1 - c_t) * x + c_t * z
            let var_t = var.tensor()?;
            let x_new = var_t
                .affine((1.0 - c_t) as f32, 0.0f32)?
                .add(&z_var.affine(c_t as f32, 0.0f32)?)?;
            var.set_data(x_new)?;
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[allow(dead_code)]
    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

/// Configuration for RAdam Schedule-Free optimizer
#[derive(Debug, Clone)]
pub struct RAdamScheduleFreeConfig {
    /// Learning rate
    pub lr: f64,
    /// Beta1 for first moment
    pub beta1: f64,
    /// Beta2 for second moment
    pub beta2: f64,
    /// Small constant for numerical stability
    pub eps: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Warm-up steps
    pub warmup_steps: usize,
    /// Schedule-free parameter r
    pub r: f64,
}

impl Default for RAdamScheduleFreeConfig {
    fn default() -> Self {
        Self {
            lr: 0.0025,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            warmup_steps: 0,
            r: 1.0,
        }
    }
}
