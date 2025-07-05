//! Model-specific training pipelines

pub mod sd15_pipeline;
pub mod sdxl_pipeline;
pub mod sd3_pipeline;
pub mod flux_pipeline;
pub mod base_pipeline;
pub mod sampling;

pub use base_pipeline::{TrainingPipeline, PipelineConfig, PreparedBatch, PromptEmbeds, PipelineUtils};
pub use sd15_pipeline::SD15Pipeline;
pub use sdxl_pipeline::{SDXLPipeline, DDPMScheduler};
pub use sd3_pipeline::SD3Pipeline;
pub use flux_pipeline::FluxPipeline;
pub use sampling::{SamplingConfig, TrainingSampler};

use eridiffusion_core::{Result, Error, ModelArchitecture};

/// Factory for creating model-specific pipelines
pub struct PipelineFactory;

impl PipelineFactory {
    pub fn create(architecture: ModelArchitecture, config: PipelineConfig) -> Result<Box<dyn TrainingPipeline>> {
        match architecture {
            ModelArchitecture::SD15 => Ok(Box::new(SD15Pipeline::new(config)?)),
            ModelArchitecture::SDXL => Ok(Box::new(SDXLPipeline::new(config)?)),
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => Ok(Box::new(SD3Pipeline::new(config)?)),
            ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => Ok(Box::new(FluxPipeline::new(config)?)),
            _ => Err(Error::Unsupported(format!("No pipeline for architecture: {:?}", architecture))),
        }
    }
}