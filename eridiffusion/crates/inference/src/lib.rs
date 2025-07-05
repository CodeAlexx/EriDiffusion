//! Inference module for AI-Toolkit

pub mod pipeline;
pub mod batch;
pub mod optimization;
pub mod cache;
pub mod server;
pub mod client;
pub mod monitoring;
pub mod sd3_pipeline;

pub use pipeline::{InferencePipeline, InferenceConfig, InferenceOutput};
pub use batch::{BatchInferenceEngine, BatchRequest, BatchResponse};
pub use optimization::{ModelOptimizer, OptimizationConfig};
pub use cache::{InferenceCache, CacheConfig, CacheKey};
pub use server::{InferenceServer, ServerConfig};
pub use client::{InferenceClient, ClientConfig};
pub use monitoring::{PerformanceMonitor, MonitoringConfig};
pub use sd3_pipeline::{SD3Pipeline, SD3PipelineConfig, Scheduler};

use eridiffusion_core::Result;

/// SD3.5 model variants
#[derive(Debug, Clone, Copy)]
pub enum SD35ModelVariant {
    Medium,
    Large,
    LargeTurbo,
}