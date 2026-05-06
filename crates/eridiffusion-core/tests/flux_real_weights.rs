//! Test Flux/Klein model forward+backward with REAL weights.
//! Verifies gradient flow through the full DiT.
use std::path::Path;
use flame_core::{self, autograd::AutogradContext, DType, Shape, Tensor};
use eridiffusion_core::config::TrainConfig;
use eridiffusion_core::models::FluxModel;
use eridiffusion_core::models::TrainableModel;

#[test]
fn klein9b_real_forward_backward() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);

    let path = Path::new("/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors");

    // Load model
    let config = TrainConfig::default();
    let model = FluxModel::load(path, &config, device.clone());

    let model = match model {
        Ok(m) => {
            println!("Loaded: {} shared, {} db, {} sb, lora={}, guidance={}",
                m.shared_weights.len(), m.double_block_weights.len(),
                m.single_block_weights.len(), m.bundle.is_some(), m.has_guidance);
            println!("Sample shared keys:");
            for (k, v) in m.shared_weights.iter().take(5) {
                println!("  {}: {:?} dtype={:?}", k, v.shape().dims(), v.dtype());
            }
            m
        }
        Err(e) => {
            println!("Load failed: {}", e);
            // Try with LoRA mode
            let mut cfg = TrainConfig::default();
            cfg.lora_rank = 16;
            cfg.lora_alpha = 16.0;
            let m2 = FluxModel::load(path, &cfg, device.clone()).expect("Should load with LoRA");
            println!("LoRA mode: {} shared, {} lora adapters, {} params",
                m2.shared_weights.len(), m2.bundle.as_ref().map(|b| b.double_adapters.len() + b.single_adapters.len()).unwrap_or(0), m2.parameters().len());
            m2
        }
    };

    // Forward pass: random noise + zeros embeddings
    let b = 1usize;
    let n_img = 1024usize; // 64x64 patches for 1024x1024 image
    let n_txt = 256usize;  // T5 sequence length
    let in_c = eridiffusion_core::models::flux::IN_CHANNELS;
    let t5_d = eridiffusion_core::models::flux::T5_DIM;
    let v_dim = eridiffusion_core::models::flux::VECTOR_DIM;

    let noisy = Tensor::randn(Shape::from_dims(&[b, n_img, in_c]), 0.0, 1.0, device.clone()).unwrap();
    let t5 = Tensor::randn(Shape::from_dims(&[b, n_txt, t5_d]), 0.0, 1.0, device.clone()).unwrap();
    let timestep = Tensor::from_vec(vec![0.5], Shape::from_dims(&[1]), device.clone()).unwrap();
    let pooled = Tensor::zeros(Shape::from_dims(&[b, v_dim]), device.clone()).unwrap();

    // Default position IDs
    let img_ids = Tensor::zeros(Shape::from_dims(&[n_img, 3]), device.clone()).unwrap();
    let txt_ids = Tensor::zeros(Shape::from_dims(&[n_txt, 3]), device.clone()).unwrap();

    println!("Forward: img={:?} txt={:?}", noisy.shape().dims(), t5.shape().dims());

    // Forward pass (call model's full forward, not TrainableModel trait)
    let guidance = if model.has_guidance {
        Some(Tensor::from_vec(vec![3.5], Shape::from_dims(&[1]), device.clone()).unwrap())
    } else { None };

    let pred = model.forward(&noisy, &t5, &timestep, &img_ids, &txt_ids, guidance.as_ref(), &pooled);
    match pred {
        Ok(p) => {
            println!("Forward output: {:?}", p.shape().dims());

            // Verify output is finite
            let out_sample = p.to_dtype(DType::F32).unwrap().to_vec().unwrap();
            let (min, max, mean) = (out_sample.iter().fold(f32::MAX, |a,b| a.min(*b)),
                                     out_sample.iter().fold(f32::MIN, |a,b| a.max(*b)),
                                     out_sample.iter().sum::<f32>() / out_sample.len() as f32);
            println!("Output stats: min={:.4} max={:.4} mean={:.4}", min, max, mean);
            assert!(min.is_finite() && max.is_finite(), "Forward output has NaN/inf");

            // Loss
            let target = Tensor::zeros(p.shape().clone(), device.clone()).unwrap();
            let diff = p.sub(&target).unwrap();
            let loss = diff.square().unwrap().mean().unwrap();
            let loss_val = loss.to_dtype(DType::F32).unwrap().to_vec().unwrap()[0];
            println!("Loss: {:.6}", loss_val);
            assert!(loss_val.is_finite(), "Loss is NaN/inf");

            // Backward
            println!("Running backward...");
            let grads = loss.backward().unwrap();
            println!("Backward: {} gradients", grads.len());

            // Check gradients for a few parameters
            let params = model.parameters();
            let mut nan_count = 0;
            let mut zero_count = 0;
            let mut max_grad = 0f32;
            let mut min_grad = f32::MAX;

            for param in params.iter().take(20) {
                let tid = param.id();
                if let Some(g) = grads.get(tid) {
                    let g_f32 = g.to_dtype(DType::F32).unwrap().to_vec().unwrap();
                    let g_abs: f32 = g_f32.iter().map(|x| x.abs()).sum();
                    let g_norm = (g_f32.iter().map(|x| x*x).sum::<f32>() / g_f32.len() as f32).sqrt();

                    if g_abs < 1e-10 { zero_count += 1; }
                    for &v in &g_f32 {
                        if v.is_nan() || v.is_infinite() { nan_count += 1; break; }
                        max_grad = max_grad.max(v);
                        min_grad = min_grad.min(v);
                    }
                }
            }
            println!("Grad check (first 20 params): nan={} zero={} min={:.6} max={:.6}",
                nan_count, zero_count, min_grad, max_grad);
            assert_eq!(nan_count, 0, "NaN gradients found!");
            assert!(max_grad.abs() > 1e-10, "All gradients near zero — no gradient flow!");

            println!("\n=== PASS: Forward+backward with real Klein 9B weights ===");
        }
        Err(e) => {
            println!("Forward failed: {}", e);
            panic!("Forward should succeed: {}", e);
        }
    }
    AutogradContext::clear();
}
