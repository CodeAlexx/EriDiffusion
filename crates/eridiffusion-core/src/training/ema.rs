//! Exponential moving average for trainable parameters.
//!
//! Moved verbatim from `klein-trainer/src/ema.rs`, and renamed from
//! `LoRAEma` to `ParameterEma` because the struct itself doesn't care
//! whether the params are LoRA tensors or base weights — it just keeps
//! an F32 shadow for each input [`Parameter`] and applies
//!
//!   `shadow = decay * shadow + (1 - decay) * param`
//!
//! Decay = 0 disables EMA. Typical values: 0.999, 0.9995, 0.9999.
//!
//! Save format is whatever the caller wants — `shadow()` returns the
//! current F32 shadow tensors and it's up to the trainer to map them
//! to save-time safetensors keys. (Klein's trainer does this in
//! `KleinLoRAModel::save_lora_weights_from_ema`.)

use flame_core::{parameter::Parameter, DType, Result, Tensor};

pub struct ParameterEma {
    decay: f32,
    /// Same length as the parameter list passed at construction time;
    /// shadow values stored as F32 tensors on the same device as their param.
    shadow: Vec<Tensor>,
}

impl ParameterEma {
    pub fn new(params: &[Parameter], decay: f32) -> Result<Self> {
        let _no_grad = flame_core::autograd::AutogradContext::no_grad();
        let mut shadow = Vec::with_capacity(params.len());
        for p in params {
            let t = p.tensor()?.to_dtype(DType::F32)?.detach()?;
            shadow.push(t);
        }
        Ok(Self { decay, shadow })
    }

    pub fn decay(&self) -> f32 {
        self.decay
    }

    /// Apply one EMA update step. Call after `optimizer.step`.
    pub fn update(&mut self, params: &[Parameter]) -> Result<()> {
        if self.decay <= 0.0 {
            return Ok(());
        }
        if self.shadow.len() != params.len() {
            return Err(flame_core::Error::InvalidInput(format!(
                "EMA shadow count {} != params count {}",
                self.shadow.len(),
                params.len()
            )));
        }
        let _no_grad = flame_core::autograd::AutogradContext::no_grad();
        let d = self.decay;
        let one_minus_d = 1.0 - d;
        for (slot, p) in self.shadow.iter_mut().zip(params.iter()) {
            let cur = p.tensor()?.to_dtype(DType::F32)?.detach()?;
            let new = slot.mul_scalar(d)?.add(&cur.mul_scalar(one_minus_d)?)?;
            *slot = new.detach()?;
        }
        Ok(())
    }

    /// Number of shadow slots (= number of tracked parameters).
    pub fn len(&self) -> usize {
        self.shadow.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shadow.is_empty()
    }

    /// Indexed read of one shadow tensor. Preserved from the original
    /// `klein-trainer::ema::LoRAEma` API so the Klein trainer's save
    /// path doesn't need to change.
    pub fn shadow(&self, index: usize) -> &Tensor {
        &self.shadow[index]
    }

    /// All shadow tensors as a slice.
    pub fn shadow_all(&self) -> &[Tensor] {
        &self.shadow
    }
}
