use anyhow::{ensure, Result};
use eridiffusion_training::sdxl::infer::{SamplerMode, SdxlInferencePipeline};
use flame_core::{DType, Tensor};
use image::{ImageBuffer, Rgb, RgbImage};
use std::path::Path;

use crate::sdxl::{config::SdxlRunConfig, SdxlNativePipeline};

pub enum SamplerBackend<'a> {
    Native(&'a SdxlNativePipeline),
    Legacy(&'a mut SdxlInferencePipeline),
}

pub struct SdxlSampler<'a> {
    backend: SamplerBackend<'a>,
}

impl<'a> SdxlSampler<'a> {
    pub fn new_native(pipeline: &'a SdxlNativePipeline) -> Self {
        Self { backend: SamplerBackend::Native(pipeline) }
    }

    pub fn new_legacy(pipeline: &'a mut SdxlInferencePipeline) -> Self {
        Self { backend: SamplerBackend::Legacy(pipeline) }
    }

    pub fn run(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
        run_cfg: &SdxlRunConfig,
    ) -> Result<Tensor> {
        match &mut self.backend {
            SamplerBackend::Native(pipeline) => pipeline.sample(
                prompt,
                negative_prompt,
                run_cfg.steps,
                run_cfg.guidance_scale,
                run_cfg.height,
                run_cfg.width,
                run_cfg.seed,
            ),
            SamplerBackend::Legacy(pipeline) => {
                let images = pipeline.sample_prompts_with_mode(
                    prompt,
                    if negative_prompt.is_empty() { None } else { Some(negative_prompt) },
                    run_cfg.steps,
                    run_cfg.guidance_scale,
                    run_cfg.height,
                    run_cfg.width,
                    run_cfg.seed,
                    SamplerMode::Euler,
                )?;
                Ok(images)
            }
        }
    }
}

pub fn tensor_to_image(tensor: &Tensor) -> Result<RgbImage> {
    let image = tensor.to_dtype(DType::F32)?;
    let dims = image.shape().dims().to_vec();
    ensure!(dims.len() == 4, "expected NHWC tensor, got {:?}", dims);
    ensure!(dims[0] == 1, "batch size must be 1, got {}", dims[0]);
    ensure!(dims[3] == 3, "expected 3 channels, got {}", dims[3]);

    let height = dims[1] as u32;
    let width = dims[2] as u32;
    let data = image.get(0)?.flatten_all()?.to_vec1::<f32>()?;

    let mut buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let r = (data[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (data[idx + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (data[idx + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            buffer.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    Ok(buffer)
}

pub fn save_image(tensor: &Tensor, path: &Path) -> Result<()> {
    let image = tensor_to_image(tensor)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    image.save(path)?;
    Ok(())
}
