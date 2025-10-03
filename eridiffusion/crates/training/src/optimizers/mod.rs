//! Optimizer implementations

pub mod adam8bit;
pub mod lion;
pub mod prodigy_schedulefree;
pub mod radam_schedulefree;
pub mod radam_schedulefree_wrapper;
pub mod wrapper;

// Re-exports
pub use adam8bit::{analyze_optimizer_memory, AdamW8bit, AdamW8bitConfig, OptimizerMemoryReport};
pub use lion::{Lion, LionConfig, LionMonitor, LionPreset, LionSchedule, LionTrainingGuide};
pub use radam_schedulefree::{RAdamScheduleFree, RAdamScheduleFreeConfig};
pub use radam_schedulefree_wrapper::RAdamScheduleFreeWrapper;
pub use wrapper::{build_optimizer, OptimizerConfig, OptimizerType, OptimizerWrapper};
// Prodigy core algorithms (analysis/types) are currently disabled;
// training-time Prodigy optimizer lives in optimizer.rs as ProdigyOptimizer.
