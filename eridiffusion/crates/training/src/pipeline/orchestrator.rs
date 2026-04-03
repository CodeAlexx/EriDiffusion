use anyhow::Result;

use super::{
    adapter::ModelAdapter,
    stages::{Ctx, StageName},
};
use crate::model_registry::TrainCfg;

pub struct Orchestrator<'a> {
    pub adapter: Box<dyn ModelAdapter + 'a>,
    pub recipe: Vec<StageName>,
}

impl<'a> Orchestrator<'a> {
    pub fn new(adapter: Box<dyn ModelAdapter>, cfg: &TrainCfg) -> Self {
        let _ = cfg; // reserved for future per-model overrides
        let recipe = adapter.default_recipe().to_vec();
        Self { adapter, recipe }
    }
    pub fn run_step(&mut self, cfg: &TrainCfg, ctx: &mut Ctx) -> Result<()> {
        for &stage in &self.recipe {
            self.adapter.run(stage, cfg, ctx)?;
        }
        Ok(())
    }
}
