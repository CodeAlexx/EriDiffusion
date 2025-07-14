//! Direct Flux model loading without VarBuilder
//! 
//! This bypasses all VarBuilder issues by constructing the model directly.

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{Linear, LayerNorm, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

use crate::models::flux_custom::{
    FluxConfig, FluxModelWithLoRA,
    blocks::{FluxDoubleBlockWithLoRA, FluxSingleBlockWithLoRA, FluxAttentionWithLoRA, MLPWithLoRA},
    utils::{MLP, EmbedND},
    lora::{LinearWithLoRA, LoRAConfig},
};
use crate::models::flux_adaptive_loader::FluxAdaptiveLoader;
use crate::models::flux_lora::modulation::{Modulation1, Modulation2};
use candle_nn::Activation;

/// Load Flux model by manually creating all components
pub fn load_flux_model_direct(
    model_path: &Path,
    config: &FluxConfig,
    device: &Device,
    dtype: DType,
) -> Result<FluxModelWithLoRA> {
    println!("Loading Flux model with direct component creation...");
    
    // Load and adapt all tensors
    let loader = FluxAdaptiveLoader::from_file(model_path, device.clone(), dtype)?;
    let mut adapted = HashMap::new();
    loader.adapt_all_tensors(&mut adapted)?;
    
    println!("Adapted {} tensors, building model components...", adapted.len());
    
    // Helper to create a Linear layer from specific tensors
    let create_linear = |weight_name: &str, bias_name: Option<&str>| -> Result<Linear> {
        let weight = adapted.get(weight_name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing weight: {}", weight_name)))?;
        
        let var_map = VarMap::new();
        var_map.set_one("weight", weight.clone())?;
        
        if let Some(bias_name) = bias_name {
            if let Some(bias) = adapted.get(bias_name) {
                var_map.set_one("bias", bias.clone())?;
            }
        }
        
        let (out_features, in_features) = weight.dims2()?;
        let vb = VarBuilder::from_varmap(&var_map, dtype, device);
        candle_nn::linear(in_features, out_features, vb)
    };
    
    // Helper to create LinearWithLoRA
    let create_linear_with_lora = |name: &str, weight_name: &str, bias_name: Option<&str>| -> Result<LinearWithLoRA> {
        let base = create_linear(weight_name, bias_name)?;
        Ok(LinearWithLoRA {
            base,
            lora: None,
            name: name.to_string(),
            trainable: false,
        })
    };
    
    // Build input projections
    let img_in = create_linear("img_in.weight", Some("img_in.bias"))?;
    let txt_in = create_linear("txt_in.weight", Some("txt_in.bias"))?;
    
    // Build time/vector embeddings
    let time_in = MLP {
        fc1: create_linear("time_in.mlp.0.fc1.weight", Some("time_in.mlp.0.fc1.bias"))?,
        fc2: create_linear("time_in.mlp.0.fc2.weight", Some("time_in.mlp.0.fc2.bias"))?,
    };
    
    let vector_in = MLP {
        fc1: create_linear("vector_in.mlp.0.fc1.weight", Some("vector_in.mlp.0.fc1.bias"))?,
        fc2: create_linear("vector_in.mlp.0.fc2.weight", Some("vector_in.mlp.0.fc2.bias"))?,
    };
    
    // Build positional embedder (usually initialized fresh)
    let pe_embedder = {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, dtype, device);
        EmbedND::new(
            config.hidden_size / config.axes_dim.len(),
            &config.axes_dim,
            vb,
        )?
    };
    
    // Build double blocks
    let mut double_blocks = Vec::new();
    for i in 0..config.depth {
        let block = build_double_block(i, &adapted, config, device, dtype)?;
        double_blocks.push(block);
    }
    
    // Build single blocks
    let mut single_blocks = Vec::new();
    for i in 0..config.depth_single_blocks {
        let block = build_single_block(i, &adapted, config, device, dtype)?;
        single_blocks.push(block);
    }
    
    // Build final layer
    let final_layer = create_linear_with_lora(
        "final_layer",
        "final_layer.weight",
        Some("final_layer.bias"),
    )?;
    
    // Build guidance (if enabled)
    let guidance_in = if config.guidance_embed {
        Some(MLP {
            fc1: create_linear("guidance_in.mlp.0.fc1.weight", Some("guidance_in.mlp.0.fc1.bias"))?,
            fc2: create_linear("guidance_in.mlp.0.fc2.weight", Some("guidance_in.mlp.0.fc2.bias"))?,
        })
    } else {
        None
    };
    
    println!("Successfully built all Flux model components!");
    
    Ok(FluxModelWithLoRA {
        config: config.clone(),
        img_in,
        txt_in,
        time_in,
        vector_in,
        pe_embedder,
        double_blocks,
        single_blocks,
        final_layer,
        guidance_in,
        hidden_size: config.hidden_size,
        num_heads: config.num_heads,
    })
}

fn build_double_block(
    idx: usize,
    tensors: &HashMap<String, Tensor>,
    config: &FluxConfig,
    device: &Device,
    dtype: DType,
) -> Result<FluxDoubleBlockWithLoRA> {
    let prefix = format!("double_blocks.{}", idx);
    let hidden_size = config.hidden_size;
    let num_heads = config.num_heads;
    let dim_head = hidden_size / num_heads;
    
    // Build modulation layers
    let img_mod = build_modulation2(&format!("{}.img_mod", prefix), tensors, hidden_size, device, dtype)?;
    let txt_mod = build_modulation2(&format!("{}.txt_mod", prefix), tensors, hidden_size, device, dtype)?;
    
    // Build attention layers
    let img_attn = build_attention(
        &format!("{}.img_attn", prefix),
        &format!("{}.img_attn", prefix),
        tensors,
        hidden_size,
        num_heads,
        device,
        dtype,
    )?;
    
    let txt_attn = build_attention(
        &format!("{}.txt_attn", prefix),
        &format!("{}.txt_attn", prefix),
        tensors,
        hidden_size,
        num_heads,
        device,
        dtype,
    )?;
    
    // Build MLPs
    let img_mlp = build_mlp(
        &format!("{}.img_mlp", prefix),
        &format!("{}.img_mlp", prefix),
        tensors,
        hidden_size,
        config.mlp_ratio,
        false, // double blocks use 0/2 naming
        device,
        dtype,
    )?;
    
    let txt_mlp = build_mlp(
        &format!("{}.txt_mlp", prefix),
        &format!("{}.txt_mlp", prefix),
        tensors,
        hidden_size,
        config.mlp_ratio,
        false, // double blocks use 0/2 naming
        device,
        dtype,
    )?;
    
    // Build layer norms
    let img_norm1 = build_layer_norm(&format!("{}.img_norm1", prefix), tensors, hidden_size, device, dtype)?;
    let img_norm2 = build_layer_norm(&format!("{}.img_norm2", prefix), tensors, hidden_size, device, dtype)?;
    let txt_norm1 = build_layer_norm(&format!("{}.txt_norm1", prefix), tensors, hidden_size, device, dtype)?;
    let txt_norm2 = build_layer_norm(&format!("{}.txt_norm2", prefix), tensors, hidden_size, device, dtype)?;
    
    Ok(FluxDoubleBlockWithLoRA {
        img_mod,
        txt_mod,
        img_attn,
        txt_attn,
        img_mlp,
        txt_mlp,
        img_norm1,
        img_norm2,
        txt_norm1,
        txt_norm2,
        block_idx: idx,
    })
}

fn build_single_block(
    idx: usize,
    tensors: &HashMap<String, Tensor>,
    config: &FluxConfig,
    device: &Device,
    dtype: DType,
) -> Result<FluxSingleBlockWithLoRA> {
    let prefix = format!("single_blocks.{}", idx);
    let hidden_size = config.hidden_size;
    let num_heads = config.num_heads;
    
    // Build modulation
    let modulation = build_modulation1(&format!("{}.modulation", prefix), tensors, hidden_size, device, dtype)?;
    
    // Build attention
    let attn = build_attention(
        &format!("{}.attn", prefix),
        &format!("{}.attn", prefix),
        tensors,
        hidden_size,
        num_heads,
        device,
        dtype,
    )?;
    
    // Build MLP
    let mlp = build_mlp(
        &format!("{}.mlp", prefix),
        &format!("{}.mlp", prefix),
        tensors,
        hidden_size,
        config.mlp_ratio,
        true, // single blocks use fc1/fc2 naming
        device,
        dtype,
    )?;
    
    // Build layer norms
    let norm1 = build_layer_norm(&format!("{}.norm1", prefix), tensors, hidden_size, device, dtype)?;
    let norm2 = build_layer_norm(&format!("{}.norm2", prefix), tensors, hidden_size, device, dtype)?;
    
    Ok(FluxSingleBlockWithLoRA {
        modulation,
        attn,
        mlp,
        norm1,
        norm2,
        block_idx: idx,
    })
}

// Helper functions...

fn build_modulation1(
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
    hidden_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<Modulation1> {
    let weight = tensors.get(&format!("{}.lin.weight", prefix))
        .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.lin.weight", prefix)))?;
    let bias = tensors.get(&format!("{}.lin.bias", prefix));
    
    let var_map = VarMap::new();
    var_map.set_one("weight", weight.clone())?;
    if let Some(b) = bias {
        var_map.set_one("bias", b.clone())?;
    }
    
    let vb = VarBuilder::from_varmap(&var_map, dtype, device);
    let lin = candle_nn::linear(hidden_size, 3 * hidden_size, vb)?;
    
    Ok(Modulation1 { lin })
}

fn build_modulation2(
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
    hidden_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<Modulation2> {
    let weight = tensors.get(&format!("{}.lin.weight", prefix))
        .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.lin.weight", prefix)))?;
    let bias = tensors.get(&format!("{}.lin.bias", prefix));
    
    let var_map = VarMap::new();
    var_map.set_one("weight", weight.clone())?;
    if let Some(b) = bias {
        var_map.set_one("bias", b.clone())?;
    }
    
    let vb = VarBuilder::from_varmap(&var_map, dtype, device);
    let lin = candle_nn::linear(hidden_size, 6 * hidden_size, vb)?;
    
    Ok(Modulation2 { lin })
}

fn build_attention(
    name: &str,
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
    hidden_size: usize,
    num_heads: usize,
    device: &Device,
    dtype: DType,
) -> Result<FluxAttentionWithLoRA> {
    let dim_head = hidden_size / num_heads;
    
    // Build Q, K, V projections
    let to_q = build_linear_with_lora(
        &format!("{}.to_q", name),
        &format!("{}.to_q.weight", prefix),
        Some(&format!("{}.to_q.bias", prefix)),
        tensors,
        device,
        dtype,
    )?;
    
    let to_k = build_linear_with_lora(
        &format!("{}.to_k", name),
        &format!("{}.to_k.weight", prefix),
        Some(&format!("{}.to_k.bias", prefix)),
        tensors,
        device,
        dtype,
    )?;
    
    let to_v = build_linear_with_lora(
        &format!("{}.to_v", name),
        &format!("{}.to_v.weight", prefix),
        Some(&format!("{}.to_v.bias", prefix)),
        tensors,
        device,
        dtype,
    )?;
    
    let to_out = build_linear_with_lora(
        &format!("{}.to_out.0", name),
        &format!("{}.to_out.0.weight", prefix),
        Some(&format!("{}.to_out.0.bias", prefix)),
        tensors,
        device,
        dtype,
    )?;
    
    Ok(FluxAttentionWithLoRA {
        to_q,
        to_k,
        to_v,
        to_out,
        heads: num_heads,
        dim_head,
        dropout: None,
    })
}

fn build_mlp(
    name: &str,
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
    hidden_size: usize,
    mlp_ratio: f32,
    use_fc_names: bool,
    device: &Device,
    dtype: DType,
) -> Result<MLPWithLoRA> {
    let mlp_hidden = (hidden_size as f32 * mlp_ratio) as usize;
    
    let (fc1_suffix, fc2_suffix) = if use_fc_names {
        ("fc1", "fc2")
    } else {
        ("0", "2")
    };
    
    let fc1 = build_linear_with_lora(
        &format!("{}.{}", name, fc1_suffix),
        &format!("{}.{}.weight", prefix, fc1_suffix),
        Some(&format!("{}.{}.bias", prefix, fc1_suffix)),
        tensors,
        device,
        dtype,
    )?;
    
    let fc2 = build_linear_with_lora(
        &format!("{}.{}", name, fc2_suffix),
        &format!("{}.{}.weight", prefix, fc2_suffix),
        Some(&format!("{}.{}.bias", prefix, fc2_suffix)),
        tensors,
        device,
        dtype,
    )?;
    
    Ok(MLPWithLoRA {
        fc1,
        fc2,
        activation: Activation::Gelu,
        dropout: None,
    })
}

fn build_layer_norm(
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
    hidden_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<LayerNorm> {
    let weight = tensors.get(&format!("{}.weight", prefix))
        .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.weight", prefix)))?;
    
    let var_map = VarMap::new();
    var_map.set_one("weight", weight.clone())?;
    
    let vb = VarBuilder::from_varmap(&var_map, dtype, device);
    candle_nn::layer_norm(hidden_size, 1e-6, vb)
}

fn build_linear_with_lora(
    name: &str,
    weight_name: &str,
    bias_name: Option<&str>,
    tensors: &HashMap<String, Tensor>,
    device: &Device,
    dtype: DType,
) -> Result<LinearWithLoRA> {
    let weight = tensors.get(weight_name)
        .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}", weight_name)))?;
    
    let var_map = VarMap::new();
    var_map.set_one("weight", weight.clone())?;
    
    if let Some(bias_name) = bias_name {
        if let Some(bias) = tensors.get(bias_name) {
            var_map.set_one("bias", bias.clone())?;
        }
    }
    
    let (out_features, in_features) = weight.dims2()?;
    let vb = VarBuilder::from_varmap(&var_map, dtype, device);
    let base = candle_nn::linear(in_features, out_features, vb)?;
    
    Ok(LinearWithLoRA {
        base,
        lora: None,
        name: name.to_string(),
        trainable: false,
    })
}