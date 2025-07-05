//! Training infrastructure for AI-Toolkit

pub mod trainer;
pub mod loss;
pub mod optimizer;
pub mod scheduler;
pub mod optimizers;
pub mod schedulers;
pub mod metrics;
pub mod metrics_extended;
pub mod metrics_logger;
pub mod checkpoint;
pub mod dataloader;
pub mod augmentation;
pub mod distributed;
pub mod accumulation;
pub mod callbacks;
pub mod mixed_precision;
pub mod gradient_accumulator;
pub mod pipelines;
pub mod checkpoint_manager;
pub mod flux_trainer;
pub mod flux_model_loader;
pub mod flux_forward_test;
pub mod flux_simple_test;
pub mod flux_preprocessor;
pub mod flux_trainer_24gb;
pub mod flux_lora_trainer_24gb;
pub mod eridiffusion_config;
pub mod sd35_trainer;
pub mod candle_utils;

// Re-exports
pub use trainer::{Trainer, TrainerConfig, TrainingState, TrainingConfig};
pub use loss::{Loss, LossType, compute_loss};
pub use optimizer::{Optimizer, OptimizerConfig, OptimizerState, create_optimizer};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerState};
pub use metrics::{MetricsTracker, Metric};
pub use metrics_logger::MetricsLogger;
pub use checkpoint::{CheckpointManager as OldCheckpointManager, Checkpoint};
pub use checkpoint_manager::{CheckpointManager, CheckpointMetadata};
pub use dataloader::{DataLoader, Dataset};
pub use distributed::{ProcessGroup, DistributedConfig};
pub use accumulation::{GradientAccumulator, AccumulationConfig};
pub use schedulers::{LRScheduler, create_scheduler as create_lr_scheduler};
pub use callbacks::{Callback, CallbackManager, EarlyStopping, ModelCheckpoint};
pub use mixed_precision::{MixedPrecisionConfig, GradScaler};
pub use pipelines::{TrainingPipeline, PipelineConfig, PipelineFactory};
pub use flux_trainer::{FluxTrainer, FluxTrainingConfig};
pub use flux_lora_trainer_24gb::{FluxLoRATrainer24GB, FluxLoRATraining24GBConfig, DTypeConfig};
pub use flux_model_loader::{FluxVAE, T5TextEncoder, CLIPTextEncoder};
pub use sd35_trainer::{SD35Trainer, SD35TrainingConfig, SD35ModelVariant};

use eridiffusion_core::Result;

/// Initialize the training module
pub fn initialize() -> Result<()> {
    // Register built-in optimizers and schedulers
    optimizer::register_builtin_optimizers()?;
    scheduler::register_builtin_schedulers()?;
    
    Ok(())
}