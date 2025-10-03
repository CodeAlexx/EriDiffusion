use anyhow::anyhow;
use crate::ops::{Linear, LayerNorm, Conv2d, GroupNorm};
use flame_core::device::Device;
use std::sync::Arc;
use std::collections::HashMap;
use flame_core::{Result, Tensor, Shape, Parameter};

// Import the existing helper traits and structs
use super::flux_model_complete::{
    TensorExt, FluxModelConfig, QKNorm, Attention, FeedForward,
    FluxRoPE, DoubleStreamBlock, SingleStreamBlock, MlpEmbedder,
    rms_norm_simple, compute_axial_freqs, get_timestep_embedding, cat
};

/// Helper to create a Linear layer with pre-loaded weights
fn create_linear_with_weights(
    weights: &HashMap<String, Tensor>,
    weight_key: &str,
    bias_key: Option<&str>,
    device: &flame_core::device::CudaDevice,
) -> Result<Linear> {
    let weight = weights.get(weight_key)
        .ok_or_else(|| flame_core::Error::InvalidOperation(
            format!("Missing weight: {}", weight_key)
        ))?;
    
    let weight_shape = weight.shape().dims();
    let out_features = weight_shape[0];
    let in_features = weight_shape[1];
    
    // Create linear layer
    let has_bias = bias_key.map(|k| weights.contains_key(k)).unwrap_or(false);
    let mut linear = Linear::new(in_features, out_features, has_bias, device)?;
    
    // Load the weights
    // Convert weight tensor to Parameter
    let weight_param = Parameter::from_tensor(weight.clone())?;
    linear.set_weight(weight_param)?;
    
    // Load bias if present
    if let Some(bias_key) = bias_key {
        if let Some(bias) = weights.get(bias_key) {
            let bias_param = Parameter::from_tensor(bias.clone())?;
            linear.set_bias(Some(bias_param))?;
        }
    }
    
    Ok(linear)
}

/// Fixed MLP embedder that loads weights properly
struct MlpEmbedderFixed {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedderFixed {
    fn new(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        device: &flame_core::device::CudaDevice,
    ) -> Result<Self> {
        let in_layer = create_linear_with_weights(
            weights,
            &format!("{}.in_layer.weight", prefix),
            Some(&format!("{}.in_layer.bias", prefix)),
            device,
        )?;
        
        let out_layer = create_linear_with_weights(
            weights,
            &format!("{}.out_layer.weight", prefix),
            Some(&format!("{}.out_layer.bias", prefix)),
            device,
        )?;
        
        Ok(Self { in_layer, out_layer })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.in_layer.forward(x)?;
        let x = x.silu()?;
        self.out_layer.forward(&x)
    }
}

/// Fixed Flux model that properly loads weights
pub struct FluxModelFixed {
    config: FluxModelConfig,
    
    // Input layers
    img_in: Linear,
    txt_in: Linear,
    time_in: MlpEmbedderFixed,
    vector_in: MlpEmbedderFixed,
    guidance_in: Option<MlpEmbedderFixed>,
    
    // Double stream blocks (process image and text separately)
    double_blocks: Vec<DoubleStreamBlock>,
    
    // Single stream blocks (process concatenated features)
    single_blocks: Vec<SingleStreamBlock>,
    
    // Output
    final_layer: Linear,
    
    device: flame_core::device::Device,
}

impl FluxModelFixed {
    pub fn new(
        config: FluxModelConfig,
        device: flame_core::device::Device,
        weights: HashMap<String, Tensor>,
    ) -> Result<Self> {
        println!("Creating FluxModelFixed with proper weight loading...");
        let cuda_device = device.cuda_device();
        
        // Create input projections with loaded weights
        let img_in = create_linear_with_weights(
            &weights,
            "img_in.weight",
            Some("img_in.bias"),
            cuda_device,
        )?;
        println!("✅ Loaded img_in weights");
        
        let txt_in = create_linear_with_weights(
            &weights,
            "txt_in.weight",
            Some("txt_in.bias"),
            cuda_device,
        )?;
        println!("✅ Loaded txt_in weights");
        
        // Time embedding
        let time_in = MlpEmbedderFixed::new(&weights, "time_in", cuda_device)?;
        println!("✅ Loaded time_in weights");
        
        // Vector embedding for pooled text
        let vector_in = MlpEmbedderFixed::new(&weights, "vector_in", cuda_device)?;
        println!("✅ Loaded vector_in weights");
        
        // Guidance embedding (optional)
        let guidance_in = if config.guidance_embed {
            Some(MlpEmbedderFixed::new(&weights, "guidance_in", cuda_device)?)
        } else {
            None
        };
        if guidance_in.is_some() {
            println!("✅ Loaded guidance_in weights");
        }
        
        // Create double stream blocks with loaded weights
        let mut double_blocks = Vec::new();
        for i in 0..config.depth {
            double_blocks.push(create_double_stream_block_with_weights(
                &weights,
                &config,
                i,
                cuda_device,
            )?);
            if i % 5 == 0 {
                println!("  Loading double block {}/{}", i + 1, config.depth);
            }
        }
        println!("✅ Loaded {} double blocks", config.depth);
        
        // Create single stream blocks with loaded weights
        let mut single_blocks = Vec::new();
        for i in 0..config.depth_single_blocks {
            single_blocks.push(create_single_stream_block_with_weights(
                &weights,
                &config,
                i,
                cuda_device,
            )?);
            if i % 10 == 0 {
                println!("  Loading single block {}/{}", i + 1, config.depth_single_blocks);
            }
        }
        println!("✅ Loaded {} single blocks", config.depth_single_blocks);
        
        // Final output layer
        let final_layer = create_linear_with_weights(
            &weights,
            "final_layer.weight",
            Some("final_layer.bias"),
            cuda_device,
        )?;
        println!("✅ Loaded final_layer weights");
        
        Ok(Self {
            config,
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            double_blocks,
            single_blocks,
            final_layer,
            device,
        })
    }
    
    pub fn forward(
        &self,
        img: &Tensor,          // Patchified image latents [B, num_patches, patch_size^2 * C]
        txt: &Tensor,          // Text embeddings [B, seq_len, hidden_size]
        timesteps: &Tensor,    // Timesteps [B]
        y: &Tensor,            // Pooled text embeddings [B, pooled_dim]
        guidance: Option<&Tensor>,  // Guidance scale [B] (optional)
    ) -> Result<Tensor> {
        let batch_size = img.shape().dims()[0];
        let img_seq_len = img.shape().dims()[1];
        let txt_seq_len = txt.shape().dims()[1];
        
        // 1. Input projections
        let img = self.img_in.forward(img)?;
        let txt = self.txt_in.forward(txt)?;
        
        // 2. Time and vector embeddings
        let vec_emb = self.vector_in.forward(y)?;
        let timestep_emb = get_timestep_embedding(timesteps, 256)?;
        let time_emb = self.time_in.forward(&timestep_emb)?;
        
        // Combine embeddings
        let mut c = time_emb.add(&vec_emb)?;
        
        // Add guidance if provided
        if let (Some(guidance), Some(guidance_in)) = (guidance, &self.guidance_in) {
            let g_emb = get_timestep_embedding(guidance, 256)?;
            let g_emb = guidance_in.forward(&g_emb)?;
            c = c.add(&g_emb)?;
        }
        
        // 3. Create RoPE embeddings
        let pe = self.create_rope_embeddings(batch_size, img_seq_len, txt_seq_len)?;
        
        // 4. Process through double stream blocks
        let mut img = img;
        let mut txt = txt;
        
        for block in &self.double_blocks {
            let (new_img, new_txt) = block.forward_with_modulation(&img, &txt, &c, &pe)?;
            img = new_img;
            txt = new_txt;
        }
        
        // 5. Concatenate for single stream
        let mut x = cat(&[&img, &txt], 1)?;
        
        // 6. Process through single stream blocks
        for block in &self.single_blocks {
            x = block.forward(&x, &c, &pe)?;
        }
        
        // 7. Extract image part and final projection
        let img_out = x.slice(&[(0, 0 + img_seq_len)])?;
        let out = self.final_layer.forward(&img_out)?;
        
        // 8. Unpatchify
        self.unpatchify(&out)
    }
    
    fn create_rope_embeddings(
        &self,
        batch_size: usize,
        img_seq_len: usize,
        txt_seq_len: usize,
    ) -> Result<FluxRoPE> {
        let device = &self.device;
        let head_dim = self.config.hidden_size / self.config.num_heads;
        
        // Create frequency basis
        let freqs = compute_axial_freqs(
            head_dim,
            self.config.axes_dim.clone(),
            self.config.theta,
            device,
        )?;
        
        Ok(FluxRoPE {
            freqs,
            img_seq_len,
            txt_seq_len,
        })
    }
    
    fn unpatchify(&self, x: &Tensor) -> Result<Tensor> {
        // Assuming x is [B, num_patches, patch_size^2 * out_channels]
        // Need to reshape back to [B, out_channels, H, W]
        let x_dims = x.shape().dims();
        let batch = x_dims[0];
        let num_patches = x_dims[1];
        let patch_dim = x_dims[2];
        
        let p = self.config.patch_size;
        let c = self.config.out_channels;
        let h = (num_patches as f32).sqrt() as usize;
        let w = h;  // Assuming square
        
        // Reshape to [B, h, w, p, p, c]
        let x = x.reshape(&[batch, h, w, p, p, c])?;
        
        // Rearrange to [B, c, h*p, w*p]
        let x = x.transpose_dims(1, 5)?   // [B, c, w, p, p, h]
            .transpose_dims(2, 4)?         // [B, c, p, p, w, h]
            .transpose_dims(4, 5)?;        // [B, c, p, p, h, w]
        
        let x = x.reshape(&[batch, c, p * h, p * w])?;
        
        Ok(x)
    }
}

// Helper functions to create blocks with loaded weights
fn create_double_stream_block_with_weights(
    weights: &HashMap<String, Tensor>,
    config: &FluxModelConfig,
    block_idx: usize,
    device: &flame_core::device::CudaDevice,
) -> Result<DoubleStreamBlock> {
    // This would need to be implemented to properly load all the weights
    // For now, returning the existing implementation
    DoubleStreamBlock::new_from_weights(
        config.hidden_size,
        config.num_heads,
        config.mlp_ratio,
        config.qkv_bias,
        &Device::from_cuda_device(device.clone()),
        weights,
        block_idx,
    )
}

fn create_single_stream_block_with_weights(
    weights: &HashMap<String, Tensor>,
    config: &FluxModelConfig,
    block_idx: usize,
    device: &flame_core::device::CudaDevice,
) -> Result<SingleStreamBlock> {
    // This would need to be implemented to properly load all the weights
    // For now, returning the existing implementation
    SingleStreamBlock::from_weights(
        config.hidden_size,
        config.num_heads,
        config.mlp_ratio,
        &Device::from_cuda_device(device.clone()),
        weights,
        block_idx,
    )
}