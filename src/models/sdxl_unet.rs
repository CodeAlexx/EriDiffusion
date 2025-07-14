use anyhow::Result;
use candle_core::{Device, DType, Tensor, Module};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::unet_2d;
use std::collections::HashMap;

/// SDXL UNet2DConditionModel wrapper that integrates with our LoRA system
pub struct SDXLUNet2DConditionModel {
    inner: unet_2d::UNet2DConditionModel,
    device: Device,
    dtype: DType,
}

impl SDXLUNet2DConditionModel {
    pub fn new(vb: VarBuilder, config: SDXLConfig) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        
        // Create the candle UNet config for SDXL
        let unet_config = get_sdxl_unet_config();
        
        // Create the inner UNet model
        let inner = unet_2d::UNet2DConditionModel::new(
            vb,
            config.in_channels,
            config.out_channels,
            false, // use_default_resblocks
            unet_config,
        )?;
        
        Ok(Self {
            inner,
            device,
            dtype,
        })
    }
    
    /// Create from an existing candle UNet model
    pub fn from_inner(
        inner: unet_2d::UNet2DConditionModel,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            inner,
            device,
            dtype,
        }
    }
    
    /// Forward pass for SDXL with additional conditioning
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        class_labels: Option<&Tensor>,
        return_dict: bool,
    ) -> Result<Tensor> {
        // Convert timestep to tensor if needed
        let timestep_tensor = Tensor::new(&[timestep], &self.device)?;
        
        // For now, use the standard forward pass
        // TODO: Add support for SDXL's additional conditioning (text_embeds, time_ids)
        Ok(self.inner.forward(sample, timestep, encoder_hidden_states)?)
    }
    
    /// Forward pass for training (expects timestep as tensor)
    pub fn forward_train(
        &self,
        sample: &Tensor,
        timesteps: &Tensor,
        encoder_hidden_states: &Tensor,
        added_cond_kwargs: Option<&HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        // Extract single timestep value for candle's UNet
        let timestep = if timesteps.dims().len() == 0 {
            timesteps.to_scalar::<f64>()?
        } else {
            timesteps.get(0)?.to_scalar::<f64>()?
        };
        
        // Handle SDXL additional conditioning
        let encoder_hidden_states = encoder_hidden_states.clone();
        
        if let Some(_kwargs) = added_cond_kwargs {
            // SDXL additional conditioning (time_ids and text_embeds) would be handled here
            // For now, we're using the standard SD forward pass
            // TODO: Implement proper SDXL conditioning when Candle supports it
        }
        
        // The UNet is loaded in F32, so convert inputs to F32
        let sample_f32 = sample.to_dtype(DType::F32)?;
        let encoder_hidden_states_f32 = encoder_hidden_states.to_dtype(DType::F32)?;
        
        // Debug: Check dtypes before forward pass (commented out for cleaner output)
        // println!("UNet forward - input sample dtype: {:?}, converted to: {:?}", sample.dtype(), sample_f32.dtype());
        // println!("UNet forward - input encoder_hidden_states dtype: {:?}, converted to: {:?}", encoder_hidden_states.dtype(), encoder_hidden_states_f32.dtype());
        // println!("UNet forward - UNet internal dtype: {:?}", self.dtype);
        
        // Forward pass through the UNet (which expects F32)
        let result = self.inner.forward(&sample_f32, timestep, &encoder_hidden_states_f32)?;
        
        // Convert back to original dtype if needed
        if result.dtype() != sample.dtype() {
            Ok(result.to_dtype(sample.dtype())?)
        } else {
            Ok(result)
        }
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[derive(Debug, Clone)]
pub struct SDXLConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub cross_attention_dim: usize,
    pub use_flash_attn: bool,
}

impl Default for SDXLConfig {
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            cross_attention_dim: 2048, // SDXL uses concatenated CLIP embeddings
            use_flash_attn: false,
        }
    }
}

/// Get SDXL-specific UNet configuration
pub fn get_sdxl_unet_config() -> unet_2d::UNet2DConditionModelConfig {
    unet_2d::UNet2DConditionModelConfig {
        blocks: vec![
            // First block - 320 channels (no cross attention for SDXL)
            unet_2d::BlockConfig {
                out_channels: 320,
                use_cross_attn: None,
                attention_head_dim: 5,
            },
            // Second block - 640 channels (2 cross attention layers)
            unet_2d::BlockConfig {
                out_channels: 640,
                use_cross_attn: Some(2),
                attention_head_dim: 10,
            },
            // Third block - 1280 channels (10 cross attention layers)
            unet_2d::BlockConfig {
                out_channels: 1280,
                use_cross_attn: Some(10),
                attention_head_dim: 20,
            },
        ],
        center_input_sample: false,
        cross_attention_dim: 2048, // Concatenated CLIP embeddings
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        use_linear_projection: true, // SDXL requires linear projection
        sliced_attention_size: None, // Optional sliced attention
    }
}

/// Load SDXL UNet from safetensors file
pub fn load_sdxl_unet(
    model_path: &str,
    device: &Device,
    dtype: DType,
) -> Result<SDXLUNet2DConditionModel> {
    println!("Loading SDXL UNet from: {}", model_path);
    
    // For now, let's use a simpler approach - we know the model has "model.diffusion_model" prefix
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], dtype, device)?
    };
    
    // Try with the known prefix first
    let vb_with_prefix = vb.pp("model.diffusion_model");
    let config = SDXLConfig::default();
    
    match SDXLUNet2DConditionModel::new(vb_with_prefix, config.clone()) {
        Ok(model) => {
            println!("Successfully loaded UNet with prefix: 'model.diffusion_model'");
            return Ok(model);
        }
        Err(e) => {
            println!("Failed to load with standard prefix: {}", e);
            
            // Try without any prefix as fallback
            match SDXLUNet2DConditionModel::new(vb, config) {
                Ok(model) => {
                    println!("Successfully loaded UNet without prefix");
                    return Ok(model);
                }
                Err(e2) => {
                    println!("Failed to load without prefix: {}", e2);
                }
            }
        }
    }
    
    println!("\nFailed to load UNet. The model file may:");
    println!("1. Not be an SDXL model");
    println!("2. Have a different weight structure");
    println!("3. Be corrupted or incomplete");
    
    Err(anyhow::anyhow!("Failed to load SDXL UNet. Please verify the model file is a valid SDXL checkpoint."))
}

/// Load SDXL UNet with custom config
pub fn load_sdxl_unet_with_config(
    model_path: &str,
    device: &Device,
    dtype: DType,
    config: SDXLConfig,
) -> Result<SDXLUNet2DConditionModel> {
    println!("Loading SDXL UNet from: {}", model_path);
    
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[model_path], dtype, device)? 
    };
    
    SDXLUNet2DConditionModel::new(vb, config)
}