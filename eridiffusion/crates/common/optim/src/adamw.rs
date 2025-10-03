use anyhow::Result;
use hashbrown::HashMap;
use flame_core::{Tensor, DType};
use eridiffusion_common_weights::{ParamId, ParamRegistry};
use crate::grad_store::GradStore;

#[derive(Clone)]
pub struct AdamWCfg { pub lr: f32, pub beta1: f32, pub beta2: f32, pub eps: f32, pub weight_decay: f32 }

pub struct AdamW { pub cfg: AdamWCfg, m: HashMap<ParamId, Tensor>, v: HashMap<ParamId, Tensor>, step: usize }

impl AdamW {
    pub fn new(_params: &[ParamId], lr: f32, betas: (f32,f32), eps: f32, weight_decay: f32) -> Self {
        Self { cfg: AdamWCfg { lr, beta1: betas.0, beta2: betas.1, eps, weight_decay }, m: HashMap::new(), v: HashMap::new(), step: 0 }
    }

    pub fn zero_grad(&mut self, _reg: &ParamRegistry) {}

    /// Step over provided ids using grads from GradStore. Computes in FP32, casts back to BF16.
    pub fn step_with_grads(&mut self, reg: &mut ParamRegistry, grads: &GradStore, ids: &[ParamId], lr: f32) -> Result<()> {
        for &id in ids {
            let g = grads.get(&id).ok_or_else(|| anyhow::anyhow!("missing grad for {:?}", id))?;
            let p = reg.get_mut_by_id(id).ok_or_else(|| anyhow::anyhow!("missing param for {:?}", id))?;

            // Cast to F32 for math
            let mut p32 = p.to_dtype(DType::F32)?;
            let g32 = g.to_dtype(DType::F32)?;

            // Init moments as zeros like p32
            let m = self.m.entry(id).or_insert_with(|| p32.zeros_like().unwrap().to_dtype(DType::F32).unwrap());
            let v = self.v.entry(id).or_insert_with(|| p32.zeros_like().unwrap().to_dtype(DType::F32).unwrap());

            // m = b1*m + (1-b1)*g
            let m_scaled = m.mul_scalar(self.cfg.beta1)?;
            let g_scaled = g32.mul_scalar(1.0 - self.cfg.beta1)?;
            *m = m_scaled.add(&g_scaled)?;

            // v = b2*v + (1-b2)*g^2
            let v_scaled = v.mul_scalar(self.cfg.beta2)?;
            let g2 = g32.mul(&g32)?;
            let g2_scaled = g2.mul_scalar(1.0 - self.cfg.beta2)?;
            *v = v_scaled.add(&g2_scaled)?;

            // denom = sqrt(v) + eps
            let denom = v.sqrt()?.add_scalar(self.cfg.eps)?;
            let update = m.div(&denom)?;

            // Decoupled weight decay
            let wd = if self.cfg.weight_decay != 0.0 { p32.mul_scalar(self.cfg.weight_decay)? } else { p32.zeros_like()? };
            let total = update.add(&wd)?;
            p32 = p32.add(&total.mul_scalar(-lr)?)?;

            // Back to BF16
            *p = p32.to_dtype(DType::BF16)?;
        }
        self.step += 1;
        Ok(())
    }
}
