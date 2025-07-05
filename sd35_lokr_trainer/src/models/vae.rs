use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae::{self, AutoEncoderKL};

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
            scale_factor: 0.13025, // SD3.5 VAE scaling factor
        })
    }
    
    pub fn encode(&self, images: &Tensor) -> Result<Tensor> {
        // Ensure images are in the right format [-1, 1] and F16 for VAE
        let images_normalized = ((images.to_dtype(DType::F32)? / 127.5)? - 1.0)?;
        let images_f16 = images_normalized.to_dtype(DType::F16)?;
        
        // Encode to latent distribution
        let dist = self.vae.encode(&images_f16)?;
        
        // Sample from distribution (no randomness during training)
        let latents = dist.sample()?;
        
        // Apply scaling factor
        Ok((latents * self.scale_factor)?)
    }
    
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        // Remove scaling factor
        let latents = (latents / self.scale_factor)?;
        
        // Decode
        let images = self.vae.decode(&latents)?;
        
        // Convert back to [0, 255]
        let images = ((images + 1.0)? * 127.5)?;
        
        Ok(images.clamp(0.0, 255.0)?)
    }
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
                    // Reverse the order of up_blocks.
                    "0" => result.push("3"),
                    "1" => result.push("2"),
                    "2" => result.push("1"),
                    "3" => result.push("0"),
                    _ => {}
                }
                i += 1; // Skip the number after up_blocks.
            }
            "resnets" => {
                if i > 0 && parts[i - 1] == "mid_block" {
                    match parts[i + 1] {
                        "0" => result.push("block_1"),
                        "1" => result.push("block_2"),
                        _ => {}
                    }
                    i += 1; // Skip the number after resnets.
                } else {
                    result.push("block");
                }
            }
            "downsamplers" => {
                result.push("downsample");
                i += 1; // Skip the 0 after downsamplers.
            }
            "conv_shortcut" => {
                result.push("nin_shortcut");
            }
            "attentions" => {
                if parts[i + 1] == "0" {
                    result.push("attn_1")
                }
                i += 1; // Skip the number after attentions.
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
                i += 1; // Skip the 0 after upsamplers.
            }
            part => result.push(part),
        }
        i += 1;
    }
    result.join(".")
}