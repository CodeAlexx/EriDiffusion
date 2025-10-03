//! Sampling modules for diffusion models

pub mod flame_sampler_complete;
pub mod flame_schedulers;

// Re-export commonly used items
pub use flame_sampler_complete::{sample_images, FlameSampler, SamplingConfig};
pub use flame_schedulers::{
    DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, Scheduler, SchedulerConfig,
};

// FLAME exports
