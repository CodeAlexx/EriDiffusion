use std::sync::Arc;

use anyhow::{Context, Result};
use eridiffusion_common_vae::{VaeKind, VaeSpec};
use eridiffusion_common_weights::strict_loader::StrictMmapLoader;
use eridiffusion_core::Device as ModelDevice;
use eridiffusion_training::sdxl::infer::SdxlInferencePipeline;
use eridiffusion_training::sdxl::RuntimeMode;
use flame_core::{DType, Device as FlameDevice};

use crate::sdxl::{config::SdxlPaths, prompt::PromptEncoder};

/// Core resources required for the native SDXL inference path.
pub struct SdxlResources {
    pub prompt_encoder: PromptEncoder,
    pub unet_mmap: Arc<StrictMmapLoader>,
    pub vae_spec: VaeSpec,
    pub seq_len: usize,
}

impl SdxlResources {
    /// Convenience accessor for the underlying CUDA device.
    pub fn flame_device(&self) -> FlameDevice {
        self.prompt_encoder.device_clone()
    }
}

pub fn load_resources(
    paths: &SdxlPaths,
    seq_len: usize,
    device_index: usize,
) -> Result<SdxlResources> {
    let prompt_encoder = PromptEncoder::load(paths, seq_len, device_index)?;
    let unet_path = paths.unet.as_path();
    let mmap =
        Arc::new(StrictMmapLoader::open(unet_path).with_context(|| {
            format!("failed to mmap UNet weights from {}", paths.unet.display())
        })?);

    let vae_spec = VaeSpec {
        kind: VaeKind::Sdxl,
        path: paths.vae.to_string_lossy().to_string(),
        latent_div: 8,
        latent_channels: 4,
        latent_scale: 0.18215,
    };

    Ok(SdxlResources { prompt_encoder, unet_mmap: mmap, vae_spec, seq_len })
}

/// Temporary helper to keep the legacy runtime available while the native path is wired.
pub fn build_pipeline(
    paths: &SdxlPaths,
    steps: usize,
    device_index: usize,
    mode: RuntimeMode,
) -> Result<SdxlInferencePipeline> {
    let device = ModelDevice::Cuda(device_index);
    let cfg = eridiffusion_training::sdxl::infer::SdxlInferConfig {
        unet_path: paths.unet.to_string_lossy().to_string(),
        vae_path: Some(paths.vae.to_string_lossy().to_string()),
        clip_l_path: paths.clip_l.to_string_lossy().to_string(),
        clip_g_path: paths.clip_g.to_string_lossy().to_string(),
        tokenizer_path: paths.tokenizer.to_string_lossy().to_string(),
        seq_len: 77,
        device,
        dtype: DType::F32,
        schedule: eridiffusion_training::sdxl::scheduler::SchedulerCfg {
            steps,
            sigma_min: 0.029,
            sigma_max: 14.0,
            rho: 7.0,
            kind: eridiffusion_training::sdxl::scheduler::ScheduleKind::Karras,
        },
        attn_chunk: None,
        kv_chunk: None,
        kernel_telemetry: false,
        unet_tile: None,
        vae_tile: None,
        mode,
    };

    SdxlInferencePipeline::new(cfg)
}
