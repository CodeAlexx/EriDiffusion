use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct SdxlPaths {
    pub unet: PathBuf,
    pub vae: PathBuf,
    pub clip_l: PathBuf,
    pub clip_g: PathBuf,
    pub tokenizer: PathBuf,
}

fn default_steps() -> usize {
    30
}
fn default_guidance() -> f32 {
    7.5
}
fn default_height() -> usize {
    1024
}
fn default_width() -> usize {
    1024
}

#[derive(Debug, Clone, Deserialize)]
pub struct SdxlRunConfig {
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default = "default_guidance")]
    pub guidance_scale: f32,
    #[serde(default = "default_height")]
    pub height: usize,
    #[serde(default = "default_width")]
    pub width: usize,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Default for SdxlRunConfig {
    fn default() -> Self {
        Self {
            steps: default_steps(),
            guidance_scale: default_guidance(),
            height: default_height(),
            width: default_width(),
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SdxlConfig {
    pub paths: SdxlPaths,
    #[serde(default)]
    pub run: SdxlRunConfig,
}
