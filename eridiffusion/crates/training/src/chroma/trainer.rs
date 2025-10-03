
#![allow(dead_code)]
use super::{registry::ChromaRegistry, scheduler::SchedulerCfg};
use crate::chroma::host_tensor::HostTensor;

#[derive(Clone, Debug)]
pub struct ChromaCfg {
    pub steps: u64,
    pub batch: usize,
    pub scheduler: SchedulerCfg,
}
pub struct ChromaTrainer {
    pub cfg: ChromaCfg,
    pub registry: ChromaRegistry,
}
impl ChromaTrainer {
    pub fn new(cfg: ChromaCfg) -> Self {
        Self { cfg, registry: ChromaRegistry::new() }
    }
    pub fn step_once(&mut self) { let _ = &self.registry; }
    pub fn run(&mut self) { for _ in 0..self.cfg.steps { self.step_once(); } }
    pub fn make_dummy_batch(&self) -> HostTensor { HostTensor::zeros(&[self.cfg.batch, 16, 32, 32], 2, Some("bf16")) }
}
