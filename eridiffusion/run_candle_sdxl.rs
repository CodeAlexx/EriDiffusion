
use candle_transformers::models::stable_diffusion;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Candle Stable Diffusion...");
    
    let device = Device::cuda_if_available(0)?;
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
    
    // Model paths
    let model_file = "/home/alex/SwarmUI/Models/diffusion_models/epicrealismXL_v9unflux.safetensors";
    let vae_file = "/home/alex/SwarmUI/Models/VAE/OfficialStableDiffusion/sdxl_vae.safetensors";
    
    println!("Loading models...");
    
    // Initialize models
    let vb_unet = VarBuilder::from_safetensors(&[model_file], dtype, &device)?;
    let vb_vae = VarBuilder::from_safetensors(&[vae_file], dtype, &device)?;
    
    // Create UNet
    let unet_config = stable_diffusion::unet_2d::UNet2DConditionModelConfig {
        blocks: vec![
            stable_diffusion::unet_2d::BlockConfig {
                out_channels: 320,
                use_cross_attn: true,
                attention_head_dim: 5,
            },
            stable_diffusion::unet_2d::BlockConfig {
                out_channels: 640,
                use_cross_attn: true,
                attention_head_dim: 10,
            },
            stable_diffusion::unet_2d::BlockConfig {
                out_channels: 1280,
                use_cross_attn: true,
                attention_head_dim: 20,
            },
        ],
        center_input_sample: false,
        cross_attention_dim: 2048,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        num_class_embeds: None,
        use_linear_projection: true,
        upcast_attention: false,
        use_flash_attn: false,
    };
    
    let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
        vb_unet,
        4, // in_channels
        320, // out_channels  
        unet_config
    );
    
    println!("Model loaded!");
    
    // Generate latents
    let latents = Tensor::randn(0f32, 1.0, (1, 4, 128, 128), &device)?;
    
    // Simple denoising (without full pipeline for now)
    let timesteps = Tensor::new(&[999i64], &device)?;
    let encoder_hidden_states = Tensor::randn(0f32, 1.0, (1, 77, 2048), &device)?;
    
    let noise_pred = unet.forward(&latents, &timesteps, &encoder_hidden_states)?;
    
    println!("Generated noise prediction: {:?}", noise_pred.shape());
    
    // Save result
    std::fs::write("candle_sdxl_result.txt", format!("Success! Shape: {:?}", noise_pred.shape()))?;
    
    Ok(())
}
