//! Optimizer implementations

pub mod radam_schedulefree;
pub mod radam_schedulefree_wrapper;
pub mod adam8bit;
pub mod lion;
pub mod prodigy;

// Re-exports
pub use radam_schedulefree::{RAdamScheduleFree, RAdamScheduleFreeConfig};
pub use radam_schedulefree_wrapper::RAdamScheduleFreeWrapper;
pub use adam8bit::{AdamW8bit, AdamW8bitConfig, analyze_optimizer_memory, OptimizerMemoryReport};
pub use lion::{Lion, LionConfig, LionPreset, LionSchedule, LionMonitor, LionTrainingGuide};
pub use prodigy::{Prodigy, ProdigyConfig, ProdigyMonitor, ProdigySummary};