use anyhow::Result;
use crate::pipeline::{adapter::ModelAdapter, stages::{Ctx, StageName}, recipes::SDXL_RECIPE};
use crate::model_registry::TrainCfg;

pub struct SdxlAdapter;

impl SdxlAdapter { pub fn new(_cfg: &TrainCfg) -> Result<Self> { Ok(Self) } }

impl ModelAdapter for SdxlAdapter {
    fn default_recipe(&self) -> &'static [StageName] { SDXL_RECIPE }
    fn run(&mut self, _stage: StageName, _cfg: &TrainCfg, _ctx: &mut Ctx) -> Result<()> {
        // Route to existing trainer components; minimal no-op scaffold for compile
        Ok(())
    }
}

