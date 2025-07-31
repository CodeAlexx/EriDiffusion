pub mod trainers;
pub mod loaders;
pub mod models;
pub mod memory;

// Re-export common types
pub use trainers::{Config, ProcessConfig, load_config};

// Create eridiffusion_core module for compatibility
pub mod eridiffusion_core {
    pub use candle_core::{Device, DType, Tensor, Module, Result, Var, D};
    
    // Model input structure
    pub struct ModelInputs {
        pub latents: Tensor,
        pub timestep: Tensor,
        pub encoder_hidden_states: Option<Tensor>,
        pub pooled_projections: Option<Tensor>,
        pub guidance_scale: Option<f32>,
        pub attention_mask: Option<Tensor>,
        pub additional: std::collections::HashMap<String, Tensor>,
    }
    
    // Model output structure
    pub struct ModelOutput {
        pub sample: Tensor,
    }
    
    // Diffusion model trait
    pub trait DiffusionModel {
        fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput>;
    }
}

pub mod logging {
    use log::LevelFilter;
    use env_logger::Builder;
    use std::io::Write;
    
    pub fn init_logger() {
        Builder::new()
            .format(|buf, record| {
                writeln!(
                    buf,
                    "{} [{}] - {}",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                    record.level(),
                    record.args()
                )
            })
            .filter(None, LevelFilter::Info)
            .init();
    }
}