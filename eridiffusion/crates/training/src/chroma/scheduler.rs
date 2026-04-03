#![allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SchedulerCfg {
    pub use_beta_sigmas: bool,
}
impl SchedulerCfg {
    pub fn default() -> Self {
        Self { use_beta_sigmas: true }
    }
}
