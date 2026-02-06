//! Training infrastructure for AI-Toolkit

// Prelude with common extensions (e.g., DimExt)
pub mod prelude;

pub mod accumulation;
pub mod augmentation;
pub mod callbacks;
pub mod checkpoint;
pub mod checkpoint_safetensors;
pub mod conditioning;
pub mod dataloader;
pub mod distributed;
pub mod gradient_accumulator;
pub mod loss;
pub mod metrics;
pub mod metrics_extended;
pub mod metrics_logger;
pub mod mixed_precision;
pub mod model_registry;
pub mod optimizer;
pub mod optimizers;
pub mod scheduler;
pub mod schedulers;
pub mod trainer;
pub mod models {
    pub use super::model_registry::models::*;
}
pub mod checkpoint_manager;
pub mod chroma;
pub mod data;
pub mod error_helpers;
pub mod flux;
pub mod flux_trainer;
pub mod gradient_checkpointing;
pub mod lora_keys;
pub mod pipeline;
pub mod pipelines;
#[cfg(feature = "examples")]
pub mod run;
pub mod sd35;
pub mod sdxl;
// flux_* modules are disabled in this compile-only pass
// pub mod flux_model_loader;
// pub mod flux_forward_test;
// pub mod flux_simple_test;
pub mod flux_preprocessor;
// pub mod flux_lora_trainer_24gb;
pub mod eridiffusion_config;
// pub mod sd35_trainer;
pub mod adapters;
pub mod eval;
pub mod policy;
pub mod streaming;
pub mod tensor_utils;
// pub mod tread;
pub mod io;
pub mod lora_builders;
pub mod samplers;
pub mod telemetry;
pub mod telemetry_mem;
pub mod tread;
pub mod utils;
pub mod util {
    pub mod broadcast;
}
pub mod init;
pub mod mp_ops;
pub use accumulation::{AccumulationConfig, GradientAccumulator};
pub use callbacks::{Callback, CallbackManager, EarlyStopping, ModelCheckpoint};
pub use checkpoint::{Checkpoint, CheckpointManager as OldCheckpointManager};
pub use checkpoint_manager::{CheckpointManager, CheckpointMetadata};
pub use dataloader::{DataLoader, Dataset};
pub use distributed::{DistributedConfig, ProcessGroup};
pub use flux_trainer::{FluxTrainer, FluxTrainingConfig};
pub use loss::{compute_loss, Loss, LossType};
pub use metrics::{Metric, MetricsTracker};
pub use metrics_logger::MetricsLogger;
pub use mixed_precision::{GradScaler, MixedPrecisionConfig};
pub use optimizer::{create_optimizer, Optimizer, OptimizerConfig, OptimizerState};
pub use pipelines::{PipelineConfig, PipelineFactory, TrainingPipeline};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerState};
pub use schedulers::{create_scheduler as create_lr_scheduler, LRScheduler};
// pub use flux_lora_trainer_24gb::{FluxLoRATrainer24GB, FluxLoRATraining24GBConfig, DTypeConfig};
// pub use flux_model_loader::{FluxVAE, T5TextEncoder, CLIPTextEncoder};
// pub use sd35_trainer::{SD35Trainer, SD35TrainingConfig, SD35ModelVariant};
// pub use pipeline::{stages, adapter, orchestrator, recipes};

/// Initialize the training module
pub fn initialize() -> anyhow::Result<()> {
    // Register built-in optimizers and schedulers
    optimizer::register_builtin_optimizers()?;
    scheduler::register_builtin_schedulers()?;

    Ok(())
}
