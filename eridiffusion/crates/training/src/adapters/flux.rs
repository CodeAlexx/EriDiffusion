use anyhow::Result;

use crate::{
    model_registry::TrainCfg,
    pipeline::{
        adapter::ModelAdapter,
        recipes::FLUX_RECIPE,
        stages::{Ctx, StageName},
    },
};

pub struct FluxAdapter;

impl FluxAdapter {
    pub fn new(_cfg: &TrainCfg) -> Result<Self> {
        Ok(Self)
    }
}

impl ModelAdapter for FluxAdapter {
    fn default_recipe(&self) -> &'static [StageName] {
        FLUX_RECIPE
    }
    fn run(&mut self, _stage: StageName, _cfg: &TrainCfg, _ctx: &mut Ctx) -> Result<()> {
        Ok(())
    }
}
