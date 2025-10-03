use anyhow::Result;
use tracing::info;
use eridiffusion_common_weights::{SafeLoader, ParamRegistry};

pub mod manager;
pub mod prefetch;

pub use manager::{BlockSwapCfg, BlockSwapManager};

