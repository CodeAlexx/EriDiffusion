#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

//! Simple memory profiling for QK-Norm

use eridiffusion::models::flux_blocks::{DoubleStreamBlock, SingleStreamBlock};
use eridiffusion::ops::streaming_rms_norm::extract_rms_norm_weights;
use flame_core::{DType, Device, Result, Shape, Tensor};
use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;

/// Get GPU memory usage via nvidia-smi
fn get_gpu_memory_mb() -> f32 {
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .expect("Failed to run nvidia-smi");

    let mem_str = String::from_utf8_lossy(&output.stdout);
    mem_str.trim().parse::<f32>().unwrap_or(0.0)
}

fn main() -> Result<()> {
    env_logger::init();

    println!("🔍 Profiling QK-Norm memory usage...\n");

    let device = Device::cuda(0)?;
    let cuda_device = device.cuda_device().clone();

    // Test with Flux-like configuration
    let hidden_size = 3072;
    let num_heads = 24;
    let batch_size = 1;
    let seq_len = 256;
    let head_dim = hidden_size / num_heads;
    let mlp_ratio = 4.0;

    println!("📊 Configuration:");
    println!("  Hidden size: {}", hidden_size);
    println!("  Num heads: {}", num_heads);
    println!("  Batch size: {}", batch_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Head dimension: {}", head_dim);
    println!("  MLP ratio: {}", mlp_ratio);
    println!();

    // Measure baseline memory
    std::thread::sleep(std::time::Duration::from_millis(100));
    let baseline_mem = get_gpu_memory_mb();
    println!("📊 Baseline GPU memory: {:.2} MB", baseline_mem);

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
    let img_ids = Tensor::zeros(Shape::from_dims(&[batch_size, seq_len, 3]), cuda_device.clone())?;
    let txt_ids = Tensor::zeros(Shape::from_dims(&[batch_size, seq_len, 3]), cuda_device.clone())?;

    std::thread::sleep(std::time::Duration::from_millis(100));
    let after_inputs_mem = get_gpu_memory_mb();
    println!(
        "📊 After creating inputs: {:.2} MB (+{:.2} MB)",
        after_inputs_mem,
        after_inputs_mem - baseline_mem
    );

    // Create weights for one block
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

    std::thread::sleep(std::time::Duration::from_millis(100));
    let after_weights_mem = get_gpu_memory_mb();
    println!(
        "📊 After creating block weights: {:.2} MB (+{:.2} MB)",
        after_weights_mem,
        after_weights_mem - after_inputs_mem
    );

    // Calculate weight memory size
    let weight_params: usize = weights.values().map(|t| t.shape().elem_count()).sum();
    let weight_size_mb = weight_params as f32 * 4.0 / 1e6; // f32 = 4 bytes
    println!("📊 Weight parameters: {} ({:.2} MB theoretical)", weight_params, weight_size_mb);

    // Extract norm weights
    let norm_weights = extract_rms_norm_weights(&weights, "double_blocks.0");
    println!("📊 Found {} QK-Norm weights", norm_weights.len());

    // Create block
    let block = DoubleStreamBlock::from_weights(
        &weights,
        hidden_size,
        num_heads,
        mlp_ratio,
        device.clone(),
    )?;

    // Memory before forward pass
    std::thread::sleep(std::time::Duration::from_millis(100));
    let before_forward_mem = get_gpu_memory_mb();
    println!("\n📊 Before forward pass: {:.2} MB", before_forward_mem);

    // Run forward pass
    println!("\n🔄 Running forward pass with QK-Norm...");
    let start_time = Instant::now();

    let (img_out, txt_out) =
        block.forward_with_norm(&img, &img_ids, &txt, &txt_ids, None, &norm_weights, 1e-6)?;

    let forward_time = start_time.elapsed();

    // Memory after forward pass
    std::thread::sleep(std::time::Duration::from_millis(100));
    let after_forward_mem = get_gpu_memory_mb();
    println!(
        "📊 After forward pass: {:.2} MB (+{:.2} MB)",
        after_forward_mem,
        after_forward_mem - before_forward_mem
    );

    println!("\n⏱️  Forward pass time: {:.2}ms", forward_time.as_millis());

    // Check output statistics
    let img_min = img_out.min_all()?;
    let img_max = img_out.max_all()?;
    let txt_min = txt_out.min_all()?;
    let txt_max = txt_out.max_all()?;

    println!("\n📊 Output statistics:");
    println!("  img_out: min={:.3}, max={:.3}", img_min, img_max);
    println!("  txt_out: min={:.3}, max={:.3}", txt_min, txt_max);

    let has_nan_inf = !img_min.is_finite()
        || !img_max.is_finite()
        || !txt_min.is_finite()
        || !txt_max.is_finite();

    if has_nan_inf {
        println!("  ⚠️  NaN/Inf detected in outputs!");
    } else {
        println!("  ✅ Outputs are stable");
    }

    // Memory efficiency analysis
    println!("\n📊 Memory Efficiency Analysis:");
    let activation_memory_mb = after_forward_mem - before_forward_mem;
    let total_memory_mb = after_forward_mem - baseline_mem;
    let efficiency = weight_size_mb / total_memory_mb * 100.0;

    println!("  Input memory: {:.2} MB", after_inputs_mem - baseline_mem);
    println!("  Weight memory: {:.2} MB (actual GPU usage)", after_weights_mem - after_inputs_mem);
    println!("  Activation memory: {:.2} MB", activation_memory_mb);
    println!("  Total memory: {:.2} MB", total_memory_mb);
    println!("  Memory efficiency: {:.1}%", efficiency);

    // Test multiple forward passes
    println!("\n🔄 Running 10 forward passes to test stability...");
    let mut times = Vec::new();
    let mut stable = true;

    for i in 0..10 {
        let start = Instant::now();
        let (img_i, txt_i) =
            block.forward_with_norm(&img, &img_ids, &txt, &txt_ids, None, &norm_weights, 1e-6)?;
        let elapsed = start.elapsed();
        times.push(elapsed.as_micros() as f32 / 1000.0);

        // Check stability
        let img_min_i = img_i.min_all()?;
        let img_max_i = img_i.max_all()?;
        if !img_min_i.is_finite() || !img_max_i.is_finite() {
            stable = false;
            println!("  ⚠️  Iteration {}: NaN/Inf detected!", i + 1);
        }
    }

    if stable {
        let avg_time = times.iter().sum::<f32>() / times.len() as f32;
        let min_time = times.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_time = times.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        println!("  ✅ All iterations stable!");
        println!("  Average time: {:.2}ms", avg_time);
        println!("  Min time: {:.2}ms", min_time);
        println!("  Max time: {:.2}ms", max_time);
    }

    Ok(())
}
