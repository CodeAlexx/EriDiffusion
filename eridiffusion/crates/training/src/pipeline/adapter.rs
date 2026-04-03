use anyhow::Result;

use super::stages::{Ctx, StageName};
use crate::model_registry::TrainCfg;

pub trait ModelAdapter: Send {
    fn default_recipe(&self) -> &'static [StageName];
    fn run(&mut self, stage: StageName, cfg: &TrainCfg, ctx: &mut Ctx) -> Result<()>;
}
