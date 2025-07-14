use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae::{self, AutoEncoderKL};

pub struct SDXLVAE {
    vae: AutoEncoderKL,
    scale_factor: f64,
}

impl SDXLVAE {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let config = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
            use_quant_conv: true,  // SDXL uses quantization convolutions
            use_post_quant_conv: true,
        };
        
        let vae = AutoEncoderKL::new(vb, 3, 3, config)?;
        
        Ok(Self {
            vae,
            scale_factor: 0.18215,
        })
    }
    
    pub fn encode(&self, images: &Tensor) -> Result<Tensor> {
        // Get original dtype
        let orig_dtype = images.dtype();
        
        // Normalize images to [-1, 1] range in F32 for numerical stability
        let images_f32 = images.to_dtype(DType::F32)?;
        let images_normalized = ((images_f32 / 127.5)? - 1.0)?;
        
        // Convert back to original dtype if needed
        let images_for_vae = if orig_dtype == DType::F16 || orig_dtype == DType::BF16 {
            images_normalized.to_dtype(orig_dtype)?
        } else {
            images_normalized
        };
        
        // SDXL VAE encoding with quantization
        let dist = self.vae.encode(&images_for_vae)?;
        
        // Sample from the distribution (use mean for deterministic results)
        let latents = dist.sample()?;
        
        // Apply SDXL scaling factor
        Ok((latents * self.scale_factor)?)
    }
    
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = (latents / self.scale_factor)?;
        let images = self.vae.decode(&latents)?;
        let images = ((images + 1.0)? * 127.5)?;
        Ok(images.clamp(0.0, 255.0)?)
    }
}

pub struct SD3VAE {
    vae: AutoEncoderKL,
    scale_factor: f64,
}

impl SD3VAE {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let config = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 16,
            norm_num_groups: 32,
            use_quant_conv: false,
            use_post_quant_conv: false,
        };
        
        let vae = AutoEncoderKL::new(vb, 3, 3, config)?;
        
        Ok(Self {
            vae,
            scale_factor: 0.13025,
        })
    }
    
    pub fn encode(&self, images: &Tensor) -> Result<Tensor> {
        let images_normalized = ((images.to_dtype(DType::F32)? / 127.5)? - 1.0)?;
        let images_f16 = images_normalized.to_dtype(DType::F16)?;
        let dist = self.vae.encode(&images_f16)?;
        let latents = dist.sample()?;
        Ok((latents * self.scale_factor)?)
    }
    
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = (latents / self.scale_factor)?;
        let images = self.vae.decode(&latents)?;
        let images = ((images + 1.0)? * 127.5)?;
        Ok(images.clamp(0.0, 255.0)?)
    }
}

// SDXL VAE weight remapping function
pub fn sdxl_vae_vb_rename(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < parts.len() {
        match parts[i] {
            "encoder" => {
                result.push("encoder");
            }
            "decoder" => {
                result.push("decoder");
            }
            "quant_conv" => {
                result.push("quant_conv");
            }
            "post_quant_conv" => {
                result.push("post_quant_conv");
            }
            "down_blocks" => {
                result.push("down");
            }
            "mid_block" => {
                result.push("mid");
            }
            "up_blocks" => {
                result.push("up");
            }
            "resnets" => {
                if i > 0 && parts[i - 1] == "mid_block" {
                    match parts[i + 1] {
                        "0" => result.push("block_1"),
                        "1" => result.push("block_2"),
                        _ => {}
                    }
                    i += 1;
                } else {
                    result.push("block");
                }
            }
            "downsamplers" => {
                result.push("downsample");
                i += 1;
            }
            "upsamplers" => {
                result.push("upsample");
                i += 1;
            }
            "conv_shortcut" => {
                result.push("nin_shortcut");
            }
            "attentions" => {
                if i + 1 < parts.len() && parts[i + 1] == "0" {
                    result.push("attn_1")
                }
                i += 1;
            }
            "group_norm" => {
                result.push("norm");
            }
            "to_q" => {
                result.push("q");
            }
            "to_k" => {
                result.push("k");
            }
            "to_v" => {
                result.push("v");
            }
            "to_out" => {
                if i + 1 < parts.len() && parts[i + 1] == "0" {
                    result.push("proj_out");
                    i += 1;
                } else {
                    result.push("proj_out");
                }
            }
            "conv_norm_out" => {
                result.push("norm_out");
            }
            "conv_out" => {
                result.push("conv_out");
            }
            "conv_in" => {
                result.push("conv_in");
            }
            part => result.push(part),
        }
        i += 1;
    }
    result.join(".")
}

// SD3 VAE weight remapping function
pub fn sd3_vae_vb_rename(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < parts.len() {
        match parts[i] {
            "down_blocks" => {
                result.push("down");
            }
            "mid_block" => {
                result.push("mid");
            }
            "up_blocks" => {
                result.push("up");
                match parts[i + 1] {
                    "0" => result.push("3"),
                    "1" => result.push("2"),
                    "2" => result.push("1"),
                    "3" => result.push("0"),
                    _ => {}
                }
                i += 1;
            }
            "resnets" => {
                if i > 0 && parts[i - 1] == "mid_block" {
                    match parts[i + 1] {
                        "0" => result.push("block_1"),
                        "1" => result.push("block_2"),
                        _ => {}
                    }
                    i += 1;
                } else {
                    result.push("block");
                }
            }
            "downsamplers" => {
                result.push("downsample");
                i += 1;
            }
            "conv_shortcut" => {
                result.push("nin_shortcut");
            }
            "attentions" => {
                if parts[i + 1] == "0" {
                    result.push("attn_1")
                }
                i += 1;
            }
            "group_norm" => {
                result.push("norm");
            }
            "query" => {
                result.push("q");
            }
            "key" => {
                result.push("k");
            }
            "value" => {
                result.push("v");
            }
            "proj_attn" => {
                result.push("proj_out");
            }
            "conv_norm_out" => {
                result.push("norm_out");
            }
            "upsamplers" => {
                result.push("upsample");
                i += 1;
            }
            part => result.push(part),
        }
        i += 1;
    }
    result.join(".")
}

pub enum VAEModel {
    SDXL(SDXLVAE),
    SD3(SD3VAE),
}

impl VAEModel {
    pub fn new_sdxl(vb: VarBuilder) -> Result<Self> {
        Ok(VAEModel::SDXL(SDXLVAE::new(vb)?))
    }
    
    pub fn new_sd3(vb: VarBuilder) -> Result<Self> {
        Ok(VAEModel::SD3(SD3VAE::new(vb)?))
    }
    
    pub fn encode(&self, images: &Tensor) -> Result<Tensor> {
        match self {
            VAEModel::SDXL(vae) => vae.encode(images),
            VAEModel::SD3(vae) => vae.encode(images),
        }
    }
    
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        match self {
            VAEModel::SDXL(vae) => vae.decode(latents),
            VAEModel::SD3(vae) => vae.decode(latents),
        }
    }
    
    pub fn latent_channels(&self) -> usize {
        match self {
            VAEModel::SDXL(_) => 4,
            VAEModel::SD3(_) => 16,
        }
    }
    
    pub fn scale_factor(&self) -> f64 {
        match self {
            VAEModel::SDXL(_) => 0.18215,
            VAEModel::SD3(_) => 0.13025,
        }
    }
    
    pub fn vb_rename(&self, name: &str) -> String {
        match self {
            VAEModel::SDXL(_) => sdxl_vae_vb_rename(name),
            VAEModel::SD3(_) => sd3_vae_vb_rename(name),
        }
    }
}
