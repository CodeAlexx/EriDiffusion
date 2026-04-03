use super::{flame_inference::SD35Pipeline, DiffusionInference, ModelConfig, SamplingConfig};
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Tensor};
use std::path::{Path, PathBuf};

/// Generate an SD3.5 image using the FLAME-backed pipeline.
pub fn generate_sd35_image(
    prompt: &str,
    negative: &str,
    variant: &str,
    _adapter: Option<&std::path::Path>,
    _adapter_scale: f32,
    output: &std::path::Path,
    steps: usize,
    cfg: f64,
    _shift: f64,
    device: Device,
    dtype: DType,
) -> Result<()> {
    let cfg_scale = cfg as f32;
    let variant_norm = variant.to_lowercase();
    let width =
        std::env::var("SD35_WIDTH").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024);
    let height =
        std::env::var("SD35_HEIGHT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024);

    let paths = resolve_sd35_paths(&variant_norm)?;

    println!("Loading SD3.5 ({variant_norm}) pipeline…");
    let mut pipeline = SD35Pipeline::from_safetensors(
        &paths.vae,
        &paths.mmdit,
        &paths.clip_l,
        &paths.clip_g,
        &paths.t5,
        device.clone(),
    )?;

    println!(
        "Generating image {width}x{height} (steps: {steps}, cfg: {cfg_scale:.2})",
        width = width,
        height = height,
        steps = steps
    );
    let image = pipeline.generate(prompt, negative, width, height, steps, cfg_scale, None)?;

    save_tensor_image(&image, output)?;
    println!("Image written to {}", output.display());
    Ok(())
}

fn save_tensor_image(tensor: &Tensor, path: &Path) -> Result<()> {
    let data = tensor.to_vec()?;
    let dims = tensor.shape().dims();
    if dims.len() != 4 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Expected [B,C,H,W] tensor, got {:?}",
            dims
        )));
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    if b != 1 || c < 3 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Expected single image with >=3 channels, got B={} C={}",
            b, c
        )));
    }

    let w_u32 = u32::try_from(w)
        .map_err(|_| Error::InvalidOperation(format!("width {w} does not fit into u32")))?;
    let h_u32 = u32::try_from(h)
        .map_err(|_| Error::InvalidOperation(format!("height {h} does not fit into u32")))?;

    let mut buffer = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * c;
            let dst = (y * w + x) * 3;
            buffer[dst] = data[base].clamp(0.0, 255.0) as u8;
            buffer[dst + 1] = data[base + 1].clamp(0.0, 255.0) as u8;
            buffer[dst + 2] = data[base + 2].clamp(0.0, 255.0) as u8;
        }
    }

    image::save_buffer_with_format(
        path,
        &buffer,
        w_u32,
        h_u32,
        image::ColorType::Rgb8,
        image::ImageFormat::Png,
    )
    .map_err(|e| Error::InvalidOperation(format!("failed to write image: {e}")))?;
    Ok(())
}

struct SD35Paths {
    vae: PathBuf,
    mmdit: PathBuf,
    clip_l: PathBuf,
    clip_g: PathBuf,
    t5: PathBuf,
}

fn resolve_sd35_paths(variant: &str) -> Result<SD35Paths> {
    let root = std::env::var("SD35_MODEL_ROOT")
        .unwrap_or_else(|_| "/home/alex/SwarmUI/Models/diffusion_models".to_string());
    let clip_root = std::env::var("SD35_CLIP_ROOT")
        .unwrap_or_else(|_| "/home/alex/SwarmUI/Models/clip".to_string());
    let t5_root = std::env::var("SD35_T5_ROOT")
        .unwrap_or_else(|_| "/home/alex/SwarmUI/Models/clip".to_string());

    let mmdit_name = match variant {
        "medium" => "sd35-medium-mmdit.safetensors",
        "large" => "sd35-large-mmdit.safetensors",
        other => {
            return Err(Error::InvalidInput(format!(
                "Unknown SD3.5 variant '{other}'. Expected 'medium' or 'large'."
            )))
        }
    };

    let base = Path::new(&root);
    let mmdit = base.join(mmdit_name);
    let vae = base.join("sd35-vae.safetensors");
    let clip_l = Path::new(&clip_root).join("clip_l.safetensors");
    let clip_g = Path::new(&clip_root).join("clip_g.safetensors");
    let t5 = Path::new(&t5_root).join("t5xxl_fp16.safetensors");

    for path in [&vae, &mmdit, &clip_l, &clip_g, &t5] {
        if !path.exists() {
            return Err(Error::InvalidInput(format!(
                "Required SD3.5 weight file not found: {} (override with SD35_* env vars)",
                path.display()
            )));
        }
    }

    Ok(SD35Paths { vae, mmdit, clip_l, clip_g, t5 })
}

#[derive(Clone, Debug)]
pub struct SD35Config {
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub linear_timesteps: bool,
    pub snr_gamma: Option<f32>,
}

impl Default for SD35Config {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            num_inference_steps: 28,
            guidance_scale: 7.0,
            seed: None,
            linear_timesteps: false,
            snr_gamma: None,
        }
    }
}

pub struct SD35Inference {
    device: Device,
    dtype: DType,
    pipeline: Option<SD35Pipeline>,
    variant: String,
}

impl SD35Inference {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            dtype: DType::BF16,
            pipeline: None,
            variant: std::env::var("SD35_VARIANT").unwrap_or_else(|_| "medium".to_string()),
        })
    }

    pub fn new_with_config(config: &ModelConfig, device: &Device) -> Result<Self> {
        let mut inference = Self::new(device)?;
        inference.variant =
            if config.unet_path.contains("large") { "large".into() } else { "medium".into() };
        inference.load_model(config)?;
        Ok(inference)
    }

    fn ensure_pipeline(&mut self) -> Result<&mut SD35Pipeline> {
        if self.pipeline.is_none() {
            let paths = resolve_sd35_paths(&self.variant)?;
            let pipeline = SD35Pipeline::from_safetensors(
                &paths.vae,
                &paths.mmdit,
                &paths.clip_l,
                &paths.clip_g,
                &paths.t5,
                self.device.clone(),
            )?;
            self.pipeline = Some(pipeline);
        }
        Ok(self.pipeline.as_mut().expect("pipeline just initialized"))
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
        config: &SD35Config,
        _device: &Device,
    ) -> Result<Tensor> {
        let pipeline = self.ensure_pipeline()?;
        pipeline.generate(
            prompt,
            negative_prompt,
            config.width,
            config.height,
            config.num_inference_steps,
            config.guidance_scale as f32,
            config.seed,
        )
    }

    pub fn apply_lora_weights(
        &mut self,
        _weights: &std::collections::HashMap<String, Tensor>,
        _scale: f32,
    ) -> Result<()> {
        Err(Error::InvalidOperation(
            "LoRA application for SD3.5 inference is not implemented yet".into(),
        ))
    }
}

impl DiffusionInference for SD35Inference {
    fn load_model(&mut self, config: &ModelConfig) -> Result<()> {
        let vae = Path::new(&config.vae_path);
        let mmdit = Path::new(&config.unet_path);
        let clip_l = Path::new(&config.clip_path);
        let clip_g = config
            .clip2_path
            .as_ref()
            .map(Path::new)
            .unwrap_or_else(|| Path::new("/home/alex/SwarmUI/Models/clip/clip_g.safetensors"));
        let t5_path =
            config.t5_path.as_ref().map(Path::new).unwrap_or_else(|| {
                Path::new("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors")
            });

        let mut pipeline = SD35Pipeline::from_safetensors(
            vae,
            mmdit,
            clip_l,
            clip_g,
            t5_path,
            self.device.clone(),
        )?;
        self.pipeline = Some(pipeline);
        Ok(())
    }

    fn encode_prompt(&mut self, prompt: &str) -> Result<Tensor> {
        let pipeline = self.pipeline.as_mut().ok_or_else(|| {
            Error::InvalidOperation("SD35Inference::encode_prompt called before load_model".into())
        })?;
        let text_encoders = pipeline.text_encoders.as_mut().ok_or_else(|| {
            Error::InvalidOperation("SD35Pipeline text encoders not initialised".into())
        })?;
        let (pos, _, _, _) = text_encoders.encode_sd35(prompt, "")?;
        Ok(pos)
    }

    fn denoise(
        &mut self,
        latents: &Tensor,
        text_embeds: &Tensor,
        steps: usize,
        cfg_scale: f64,
    ) -> Result<Tensor> {
        Err(Error::InvalidOperation(format!(
            "denoise not implemented (steps={}, cfg_scale={})",
            steps, cfg_scale
        )))
    }

    fn decode_vae(&self, latents: &Tensor) -> Result<Tensor> {
        Err(Error::InvalidOperation(format!(
            "decode_vae not implemented for tensor with shape {:?}",
            latents.shape().dims()
        )))
    }

    fn apply_lora(
        &mut self,
        _lora_weights: &std::collections::HashMap<String, Tensor>,
        _scale: f32,
    ) -> Result<()> {
        self.apply_lora_weights(_lora_weights, _scale)
    }
}
