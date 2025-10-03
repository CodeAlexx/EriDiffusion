
#![allow(dead_code)]
use super::{ema::EmaState, losses::{LossKind,l2_loss,srpo_loss}};
use crate::sd35::host_tensor::HostTensor;

#[derive(Clone, Debug)]
pub struct TrainCfg {
    pub steps: u64,
    pub batch: usize,
    pub loss: LossKind,
    pub use_ema: bool,
}
#[derive(Clone, Debug)]
pub struct SD35Trainer {
    pub cfg: TrainCfg,
    pub ema: Option<EmaState>,
}
impl SD35Trainer {
    pub fn new(cfg: TrainCfg) -> Self {
        Self { ema: if cfg.use_ema { Some(EmaState::new()) } else { None }, cfg }
    }
    pub fn step_once(&mut self) -> f32 {
        let pred = [0.0_f32; 4]; let target = [0.0_f32; 4];
        let loss = match self.cfg.loss { LossKind::L2 => l2_loss(&pred,&target), LossKind::SRPO => srpo_loss(&pred,&target) };
        if let Some(ema) = &mut self.ema { ema.update(); }
        loss
    }
    pub fn run(&mut self) {
        for _ in 0..self.cfg.steps { let _ = self.step_once(); }
    }
    pub fn make_dummy_batch(&self) -> HostTensor {
        HostTensor::zeros(&[self.cfg.batch, 4, 32, 32], 2, Some("bf16"))
    }
}
