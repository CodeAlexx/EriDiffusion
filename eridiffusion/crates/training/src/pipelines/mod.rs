//! Model-specific training pipelines

#[cfg(any(feature = "flux", feature = "sdxl", feature = "sd35"))]
pub mod base_pipeline;
#[cfg(feature = "flux")]
pub mod flux_pipeline;
pub mod sampling;
#[cfg(feature = "sd35")]
pub mod sd3_pipeline;
#[cfg(feature = "sdxl")]
pub mod sdxl_pipeline;

pub use base_pipeline::{
    PipelineConfig, PipelineUtils, PreparedBatch, PromptEmbeds, TrainingPipeline,
};
use eridiffusion_core::ModelArchitecture;
#[cfg(feature = "flux")]
pub use flux_pipeline::FluxPipeline;
pub use sampling::{SamplingConfig, TrainingSampler};
#[cfg(feature = "sd35")]
pub use sd3_pipeline::SD3Pipeline;
#[cfg(feature = "sdxl")]
pub use sdxl_pipeline::{DDPMScheduler, SDXLPipeline};

// Re-export common helpers to avoid duplicating logic across pipelines
pub use crate::loss::{masked_eps_loss, masked_l1_loss, masked_v_loss, scale_by_sigma, to_fp32};
pub use crate::tensor_utils::{scalar_f32, scalar_i32};

/// Factory for creating model-specific pipelines
pub struct PipelineFactory;

impl PipelineFactory {
    pub fn create(
        architecture: ModelArchitecture,
        config: PipelineConfig,
    ) -> anyhow::Result<Box<dyn TrainingPipeline>> {
        match architecture {
            ModelArchitecture::SD15 => {
                Err(anyhow::anyhow!("SD15 pipeline not supported in Phase 4"))
            }
            #[cfg(feature = "sdxl")]
            ModelArchitecture::SDXL => Ok(Box::new(SDXLPipeline::new(config)?)),
            #[cfg(not(feature = "sdxl"))]
            ModelArchitecture::SDXL => Err(anyhow::anyhow!("SDXL pipeline not enabled")),
            #[cfg(feature = "sd35")]
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
                Ok(Box::new(SD3Pipeline::new(config)?))
            }
            #[cfg(not(feature = "sd35"))]
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
                Err(anyhow::anyhow!("SD35 pipeline not enabled"))
            }
            #[cfg(feature = "flux")]
            ModelArchitecture::Flux
            | ModelArchitecture::FluxSchnell
            | ModelArchitecture::FluxDev => Ok(Box::new(FluxPipeline::new(config)?)),
            #[cfg(not(feature = "flux"))]
            ModelArchitecture::Flux
            | ModelArchitecture::FluxSchnell
            | ModelArchitecture::FluxDev => Err(anyhow::anyhow!("Flux pipeline not enabled")),
            _ => Err(anyhow::anyhow!(format!("No pipeline for architecture: {:?}", architecture))),
        }
    }
}
