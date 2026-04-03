use anyhow::Result;

use crate::{
    model_registry::TrainCfg,
    pipeline::{
        adapter::ModelAdapter,
        recipes::SDXL_RECIPE,
        stages::{Ctx, StageName},
    },
};

pub struct SdxlAdapter;

impl SdxlAdapter {
    pub fn new(_cfg: &TrainCfg) -> Result<Self> {
        Ok(Self)
    }
}

impl ModelAdapter for SdxlAdapter {
    fn default_recipe(&self) -> &'static [StageName] {
        SDXL_RECIPE
    }
    fn run(&mut self, _stage: StageName, _cfg: &TrainCfg, _ctx: &mut Ctx) -> Result<()> {
        Ok(())
    }
}
