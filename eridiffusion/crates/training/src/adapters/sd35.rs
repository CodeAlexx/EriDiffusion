use anyhow::Result;

use crate::{
    model_registry::TrainCfg,
    pipeline::{
        adapter::ModelAdapter,
        recipes::SD35_RECIPE,
        stages::{Ctx, StageName},
    },
};

pub struct Sd35Adapter;

impl Sd35Adapter {
    pub fn new(_cfg: &TrainCfg) -> Result<Self> {
        Ok(Self)
    }
}

impl ModelAdapter for Sd35Adapter {
    fn default_recipe(&self) -> &'static [StageName] {
        SD35_RECIPE
    }
    fn run(&mut self, _stage: StageName, _cfg: &TrainCfg, _ctx: &mut Ctx) -> Result<()> {
        Ok(())
    }
}
