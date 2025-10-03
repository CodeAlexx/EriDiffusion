#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

//! Profile memory usage with QK-Norm implementation

use eridiffusion::models::flux_blocks::{DoubleStreamBlock, SingleStreamBlock};
use eridiffusion::ops::streaming_rms_norm::extract_rms_norm_weights;
use flame_core::{DType, Device, Result, Shape, Tensor};
use std::collections::HashMap;
use std::time::Instant;

/// Get current GPU memory usage in MB
fn get_gpu_memory_mb() -> f32 {
    // Use nvidia-ml to get memory info
    if let Ok(nvml) = nvml_wrapper::Nvml::init() {
        if let Ok(device) = nvml.device_by_index(0) {
            if let Ok(memory_info) = device.memory_info() {
                return memory_info.used as f32 / 1e6;
            }
        }
    }
    0.0
}

fn main() -> Result<()> {
    env_logger::init();

    println!("🔍 Profiling QK-Norm memory usage...\n");

    let device = Device::cuda(0)?;
    let cuda_device = device.cuda_device().clone();

    // Test configurations
    let configs = vec![
        ("Small", 1024, 16, 1, 4), // hidden_size, num_heads, batch, blocks
        ("Medium", 2048, 16, 1, 8),
        ("Large", 3072, 24, 1, 19), // Flux-like config
        ("XL", 4096, 32, 1, 19),
    ];

    for (name, hidden_size, num_heads, batch_size, num_blocks) in configs {
        println!("📊 Testing {} configuration:", name);
        println!("  Hidden size: {}", hidden_size);
        println!("  Num heads: {}", num_heads);
        println!("  Batch size: {}", batch_size);
        println!("  Num blocks: {}", num_blocks);

        let seq_len = 256;
        let head_dim = hidden_size / num_heads;
        let mlp_ratio = 4.0;

        // Measure baseline memory
        let baseline_mem = get_gpu_memory_mb();
        println!("  Baseline GPU memory: {:.2} MB", baseline_mem);

        // Create inputs
        let img = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len, hidden_size]),
            0.0,
            1.0,
            cuda_device.clone(),
        )?;
        let txt = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len, hidden_size]),
            0.0,
            1.0,
            cuda_device.clone(),
        )?;
        let img_ids =
            Tensor::zeros(Shape::from_dims(&[batch_size, seq_len, 3]), cuda_device.clone())?;
        let txt_ids =
            Tensor::zeros(Shape::from_dims(&[batch_size, seq_len, 3]), cuda_device.clone())?;

        let after_inputs_mem = get_gpu_memory_mb();
        println!(
            "  After inputs: {:.2} MB (+{:.2} MB)",
            after_inputs_mem,
            after_inputs_mem - baseline_mem
        );

        // Create weights for all blocks
        let mut all_weights = Vec::new();
        let mut peak_memory = after_inputs_mem;

        for i in 0..num_blocks {
            let mut weights = HashMap::new();

            // QKV weights
            weights.insert(
                "img_attn.qkv.weight".to_string(),
                Tensor::randn(
                    Shape::from_dims(&[3 * hidden_size, hidden_size]),
                    0.0,
                    0.02,
                    cuda_device.clone(),
                )?,
            );
            weights.insert(
                "txt_attn.qkv.weight".to_string(),
                Tensor::randn(
                    Shape::from_dims(&[3 * hidden_size, hidden_size]),
                    0.0,
                    0.02,
                    cuda_device.clone(),
                )?,
            );

            // Projection weights
            weights.insert(
                "img_attn.proj.weight".to_string(),
                Tensor::randn(
                    Shape::from_dims(&[hidden_size, hidden_size]),
                    0.0,
                    0.02,
                    cuda_device.clone(),
                )?,
            );
            weights.insert(
                "txt_attn.proj.weight".to_string(),
                Tensor::randn(
                    Shape::from_dims(&[hidden_size, hidden_size]),
                    0.0,
                    0.02,
                    cuda_device.clone(),
                )?,
            );

            // QK-Norm weights
            weights.insert(
                "img_attn.norm.query_norm.scale".to_string(),
                Tensor::ones(Shape::from_dims(&[head_dim]), cuda_device.clone())?,
            );
            weights.insert(
                "img_attn.norm.key_norm.scale".to_string(),
                Tensor::ones(Shape::from_dims(&[head_dim]), cuda_device.clone())?,
            );
            weights.insert(
                "txt_attn.norm.query_norm.scale".to_string(),
                Tensor::ones(Shape::from_dims(&[head_dim]), cuda_device.clone())?,
            );
            weights.insert(
                "txt_attn.norm.key_norm.scale".to_string(),
                Tensor::ones(Shape::from_dims(&[head_dim]), cuda_device.clone())?,
            );

            // MLP weights
            let mlp_hidden = (hidden_size as f32 * mlp_ratio) as usize;
            weights.insert(
                "img_mlp.0.weight".to_string(),
                Tensor::randn(
                    Shape::from_dims(&[mlp_hidden, hidden_size]),
                    0.0,
                    0.02,
                    cuda_device.clone(),
                )?,
            );
            weights.insert(
                "img_mlp.2.weight".to_string(),
                Tensor::randn(
                    Shape::from_dims(&[hidden_size, mlp_hidden]),
                    0.0,
                    0.02,
                    cuda_device.clone(),
                )?,
            );
            weights.insert(
                "txt_mlp.0.weight".to_string(),
                Tensor::randn(
                    Shape::from_dims(&[mlp_hidden, hidden_size]),
                    0.0,
                    0.02,
                    cuda_device.clone(),
                )?,
            );
            weights.insert(
                "txt_mlp.2.weight".to_string(),
                Tensor::randn(
                    Shape::from_dims(&[hidden_size, mlp_hidden]),
                    0.0,
                    0.02,
                    cuda_device.clone(),
                )?,
            );

            all_weights.push(weights);

            let current_mem = get_gpu_memory_mb();
            if current_mem > peak_memory {
                peak_memory = current_mem;
            }
        }

        let after_weights_mem = get_gpu_memory_mb();
        println!(
            "  After creating {} blocks: {:.2} MB (+{:.2} MB)",
            num_blocks,
            after_weights_mem,
            after_weights_mem - after_inputs_mem
        );

        // Run forward passes and measure memory
        let mut img_hidden = img.clone();
        let mut txt_hidden = txt.clone();
        let mut forward_peak_memory = after_weights_mem;

        let start_time = Instant::now();

        for (i, weights) in all_weights.iter().enumerate() {
            // Extract norm weights
            let norm_weights = extract_rms_norm_weights(weights, &format!("double_blocks.{}", i));

            // Create block
            let block = DoubleStreamBlock::from_weights(
                weights,
                hidden_size,
                num_heads,
                mlp_ratio,
                device.clone(),
            )?;

            // Forward pass
            let (new_img, new_txt) = block.forward_with_norm(
                &img_hidden,
                &img_ids,
                &txt_hidden,
                &txt_ids,
                None,
                &norm_weights,
                1e-6,
            )?;

            img_hidden = new_img;
            txt_hidden = new_txt;

            // Check memory
            let current_mem = get_gpu_memory_mb();
            if current_mem > forward_peak_memory {
                forward_peak_memory = current_mem;
            }
        }

        let forward_time = start_time.elapsed();

        println!(
            "  Peak memory during forward: {:.2} MB (+{:.2} MB from weights)",
            forward_peak_memory,
            forward_peak_memory - after_weights_mem
        );
        println!("  Forward pass time: {:.2}ms", forward_time.as_millis());
        println!("  Time per block: {:.2}ms", forward_time.as_millis() as f32 / num_blocks as f32);

        // Calculate memory efficiency
        let weight_memory_mb = (after_weights_mem - after_inputs_mem);
        let activation_memory_mb = (forward_peak_memory - after_weights_mem);
        let memory_efficiency = weight_memory_mb / (weight_memory_mb + activation_memory_mb);

        println!("  Weight memory: {:.2} MB", weight_memory_mb);
        println!("  Activation memory: {:.2} MB", activation_memory_mb);
        println!("  Memory efficiency: {:.1}%", memory_efficiency * 100.0);

        // Check output statistics
        let img_min = img_hidden.min_all()?;
        let img_max = img_hidden.max_all()?;
        let txt_min = txt_hidden.min_all()?;
        let txt_max = txt_hidden.max_all()?;

        let has_nan_inf = !img_min.is_finite()
            || !img_max.is_finite()
            || !txt_min.is_finite()
            || !txt_max.is_finite();

        if has_nan_inf {
            println!("  ⚠️  NaN/Inf detected in outputs!");
        } else {
            println!(
                "  ✅ Outputs are stable (img: [{:.3}, {:.3}], txt: [{:.3}, {:.3}])",
                img_min, img_max, txt_min, txt_max
            );
        }

        println!();
    }

    Ok(())
}
