
#![allow(dead_code)]
use super::{lora_dual::{DualLoraMap}, parity::{ParityCfg, run_parity}};
use crate::wan2::host_tensor::HostTensor;

#[derive(Clone, Debug)]
pub struct WanCfg { pub steps: u64, pub batch: usize }
#[derive(Clone, Debug)]
pub struct WanTrainer { pub cfg: WanCfg, pub dual: DualLoraMap }
impl WanTrainer {
    pub fn new(cfg: WanCfg) -> Self { Self { cfg, dual: DualLoraMap::new() } }
    pub fn step_once(&mut self) {}
    pub fn run(&mut self) { for _ in 0..self.cfg.steps { self.step_once(); } }
    pub fn parity_smoke(&self) -> bool { run_parity(&ParityCfg{ mode: "smoke".into() }) }
    pub fn make_dummy_batch(&self) -> HostTensor { HostTensor::zeros(&[self.cfg.batch, 4, 32, 32], 2, Some("bf16")) }
}
