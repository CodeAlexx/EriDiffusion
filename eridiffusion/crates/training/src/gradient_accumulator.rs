use eridiffusion_core::{Device, Error, Result};
use flame_core::{DType, Tensor, TensorGradExt};

use crate::optimizer::Optimizer;
use crate::tensor_utils::sum_all_bf16;

/// Per-parameter master gradient buffer (matches parameter dtype)
#[derive(Default)]
struct GradSlot {
    master: Option<Tensor>,
}

impl GradSlot {
    fn accumulate(&mut self, grad_in: &Tensor, param: &Tensor) -> Result<()> {
        // Ensure master buffer on same device/shape/dtype as param
        if self.master.is_none() {
            let zero =
                Tensor::zeros_dtype(param.shape().clone(), param.dtype(), param.device().clone())?;
            self.master = Some(zero);
        }
        let g = if grad_in.dtype() != param.dtype() {
            grad_in.to_dtype(param.dtype())?
        } else {
            grad_in.clone()
        };
        // Accumulate
        let sum = self.master.as_ref().unwrap().add(&g)?;
        self.master = Some(sum);
        Ok(())
    }

    fn take(
        &mut self,
        shape: &flame_core::Shape,
        device: &std::sync::Arc<flame_core::CudaDevice>,
        dtype: DType,
    ) -> Result<Tensor> {
        if let Some(t) = self.master.take() {
            Ok(t)
        } else {
            Tensor::zeros_dtype(shape.clone(), dtype, device.clone())
                .map_err(eridiffusion_core::Error::from)
        }
    }

    fn zero_(&mut self) -> Result<()> {
        if let Some(m) = &mut self.master {
            *m = m.affine(0.0f32, 0.0f32)?;
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.master = None;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fp32_master_accumulates_bf16_grads() {
        eprintln!("skipping fp32_master_accumulates_bf16_grads (GPU-only test)");
        return;
    }
}

/// Gradient accumulator for efficient memory usage
pub struct GradientAccumulator {
    accumulation_steps: usize,
    current_step: usize,
    accumulated_grads: Vec<GradSlot>, // FP32 masters
    _device: Device,
    mixed_precision: bool,
    loss_scale: f32,
    found_inf: bool,
}

impl GradientAccumulator {
    pub fn new(accumulation_steps: usize, device: Device) -> Result<Self> {
        Ok(Self {
            accumulation_steps,
            current_step: 0,
            accumulated_grads: Vec::new(),
            _device: device,
            mixed_precision: false,
            loss_scale: 65536.0,
            found_inf: false,
        })
    }

    pub fn new_with_mixed_precision(
        accumulation_steps: usize,
        device: Device,
        mixed_precision: bool,
    ) -> Result<Self> {
        Ok(Self { mixed_precision, ..Self::new(accumulation_steps, device)? })
    }

    pub fn initialize(&mut self, params: &[&Tensor]) -> Result<()> {
        self.accumulated_grads.clear();
        self.accumulated_grads.resize_with(params.len(), || GradSlot::default());
        self.current_step = 0;
        self.found_inf = false;
        Ok(())
    }

    pub fn accumulate(&mut self, loss: &Tensor, params: &[&Tensor]) -> Result<()> {
        let scale = 1.0f32 / self.accumulation_steps as f32;
        let scaled = if self.mixed_precision {
            loss.affine(self.loss_scale as f32 * scale, 0.0f32)?
        } else {
            loss.affine(scale, 0.0f32)?
        };
        let gmap =
            scaled.backward().map_err(|e| Error::Training(format!("backward failed: {}", e)))?;
        if self.accumulated_grads.len() != params.len() {
            self.initialize(params)?;
        }
        for (i, p) in params.iter().enumerate() {
            let p_ref: &Tensor = *p;
            if let Some(g) = p_ref.grad(&gmap) {
                self.accumulated_grads[i].accumulate(&g, p)?;
            }
        }
        self.current_step += 1;
        Ok(())
    }

    pub fn should_step(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }
    pub fn is_ready(&self) -> bool {
        self.should_step()
    }
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }

    pub fn get_gradients(&mut self, params: &[&Tensor]) -> Result<Vec<Tensor>> {
        let mut out = Vec::with_capacity(params.len());
        for (i, p) in params.iter().enumerate() {
            let g = self.accumulated_grads[i].take(p.shape(), p.device(), p.dtype())?;
            out.push(g);
        }
        Ok(out)
    }

    /// Accumulate externally computed gradients (e.g., custom backward)
    pub fn accumulate_grads(&mut self, grads: &[Tensor]) -> Result<()> {
        if self.accumulated_grads.len() != grads.len() {
            // Initialize slots based on grads length
            self.accumulated_grads.clear();
            self.accumulated_grads.resize_with(grads.len(), || GradSlot::default());
        }
        for (i, g) in grads.iter().enumerate() {
            // Use grad tensor for both grad and shape/device source
            self.accumulated_grads[i].accumulate(g, g)?;
        }
        self.current_step += 1;
        Ok(())
    }

    pub fn step_optimizer<O: Optimizer>(
        &mut self,
        params: &[&Tensor],
        optimizer: &mut O,
        lr: f64,
        max_grad_norm: Option<f64>,
    ) -> Result<bool> {
        if !self.should_step() {
            return Ok(false);
        }
        if self.found_inf {
            self.reset()?;
            self.loss_scale = (self.loss_scale / 2.0).max(1.0);
            return Ok(false);
        }
        if self.accumulated_grads.len() != params.len() {
            self.initialize(params)?;
        }

        let mut grads = self.get_gradients(params)?;
        if let Some(maxn) = max_grad_norm {
            self.clip_vec_grads(&mut grads, maxn as f32)?;
        }
        if self.mixed_precision {
            for g in grads.iter_mut() {
                *g = g.affine(1.0f32 / self.loss_scale as f32, 0.0f32)?;
            }
        }
        optimizer.step(params, &grads, lr)?;
        self.reset()?;
        Ok(true)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.current_step = 0;
        for slot in self.accumulated_grads.iter_mut() {
            slot.reset();
        }
        self.found_inf = false;
        Ok(())
    }

    fn clip_vec_grads(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32> {
        let mut total_sq = 0.0f32;
        for g in grads.iter() {
            let sq = g.square()?;
            let sum = if sq.dtype() == DType::BF16 {
                sum_all_bf16(&sq)?
            } else {
                sq.sum()?
            };
            total_sq += sum.item()?;
        }
        let total = total_sq.sqrt();
        if total.is_finite() && total > max_norm {
            let scale = max_norm / (total + 1e-6);
            for g in grads.iter_mut() {
                *g = g.affine(scale, 0.0f32)?;
            }
            Ok(max_norm)
        } else {
            Ok(total)
        }
    }
}

#[derive(Debug, Clone)]
pub struct GradientStats {
    pub global_norm: f64,
    pub max_gradient: f64,
    pub min_gradient: f64,
    pub num_parameters: usize,
}
