use anyhow::Result;
use crate::pipeline::{adapter::ModelAdapter, stages::{Ctx, StageName}, recipes::SD35_RECIPE};
use crate::model_registry::TrainCfg;

pub struct Sd35Adapter;

impl Sd35Adapter { pub fn new(_cfg: &TrainCfg) -> Result<Self> { Ok(Self) } }

impl ModelAdapter for Sd35Adapter {
    fn default_recipe(&self) -> &'static [StageName] { SD35_RECIPE }
    fn run(&mut self, _stage: StageName, _cfg: &TrainCfg, _ctx: &mut Ctx) -> Result<()> {
        Ok(())
    }
}

