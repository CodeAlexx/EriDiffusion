//! Smoke tests: verify flame-core autograd, matmul, activations, layernorm.
//! Run with: LD_LIBRARY_PATH=<cudnn_path> cargo test --test smoke_test -- --nocapture

use flame_core::{self, autograd::AutogradContext, DType, Shape, Tensor};

#[test]
fn test_matmul_autograd() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    // Two weight tensors (trainable)
    let w1 = Tensor::randn(Shape::from_dims(&[64, 32]), 0.0, 0.02, device.clone())
        .unwrap().requires_grad_(true);
    let w2 = Tensor::randn(Shape::from_dims(&[32, 64]), 0.0, 0.02, device.clone())
        .unwrap().requires_grad_(true);

    let input = Tensor::randn(Shape::from_dims(&[2, 32]), 0.0, 1.0, device.clone()).unwrap();
    let target = Tensor::zeros(Shape::from_dims(&[2, 64]), device.clone()).unwrap();

    // Forward
    let w1t = w1.transpose().unwrap();
    let h = input.matmul(&w1t).unwrap().relu().unwrap();
    let w2t = w2.transpose().unwrap();
    let out = h.matmul(&w2t).unwrap();

    let diff = out.sub(&target).unwrap();
    let loss = diff.square().unwrap().mean().unwrap();
    let loss_f32 = loss.to_vec().unwrap()[0];
    println!("Initial loss: {:.6}", loss_f32);
    assert!(loss_f32 > 0.0 && loss_f32.is_finite());

    // Backward
    let grads = loss.backward().unwrap();
    println!("Backward: {} gradients", grads.len());
    assert!(grads.len() >= 2, "Expected >=2 gradients, got {}", grads.len());

    // Check W1 gradient
    let g1 = grads.get(w1.id()).expect("W1 gradient missing");
    let g1_abs: f32 = g1.to_vec().unwrap().iter().map(|x| x.abs()).sum();
    println!("W1 grad abs sum: {:.6}", g1_abs);
    assert!(g1_abs > 0.0, "W1 gradient should be non-zero");

    // Check W2 gradient
    let g2 = grads.get(w2.id()).expect("W2 gradient missing");
    let g2_abs: f32 = g2.to_vec().unwrap().iter().map(|x| x.abs()).sum();
    println!("W2 grad abs sum: {:.6}", g2_abs);
    assert!(g2_abs > 0.0, "W2 gradient should be non-zero");

    AutogradContext::clear();
    println!("matmul_autograd: PASS\n");
}

#[test]
fn test_layernorm_autograd() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    // Create tensor that requires grad
    let x = Tensor::randn(Shape::from_dims(&[2, 64]), 0.0, 1.0, device.clone())
        .unwrap().requires_grad_(true);

    let normed = flame_core::layer_norm::layer_norm(&x, &[64], None, None, 1e-6).unwrap();
    let loss = normed.square().unwrap().mean().unwrap();
    let grads = loss.backward().unwrap();

    let g = grads.get(x.id()).expect("LayerNorm gradient missing");
    let g_abs: f32 = g.to_vec().unwrap().iter().map(|x| x.abs()).sum();
    println!("LayerNorm grad abs sum: {:.6}", g_abs);
    assert!(g_abs > 0.0, "LayerNorm gradient should be non-zero");

    AutogradContext::clear();
    println!("layernorm_autograd: PASS\n");
}

#[test]
fn test_gelu_autograd() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    let x = Tensor::randn(Shape::from_dims(&[2, 64]), 0.0, 1.0, device.clone())
        .unwrap().requires_grad_(true);
    let out = x.gelu().unwrap();
    let loss = out.square().unwrap().mean().unwrap();
    let grads = loss.backward().unwrap();
    let g = grads.get(x.id()).expect("GELU gradient missing");
    let g_abs: f32 = g.to_vec().unwrap().iter().map(|x| x.abs()).sum();
    println!("GELU grad abs sum: {:.6}", g_abs);
    assert!(g_abs > 0.0);
    AutogradContext::clear();
    println!("gelu_autograd: PASS\n");
}

#[test]
fn test_silu_autograd() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    let x = Tensor::randn(Shape::from_dims(&[2, 64]), 0.0, 1.0, device.clone())
        .unwrap().requires_grad_(true);
    let out = x.silu().unwrap();
    let loss = out.square().unwrap().mean().unwrap();
    let grads = loss.backward().unwrap();
    let g = grads.get(x.id()).expect("SiLU gradient missing");
    let g_abs: f32 = g.to_vec().unwrap().iter().map(|x| x.abs()).sum();
    println!("SiLU grad abs sum: {:.6}", g_abs);
    assert!(g_abs > 0.0);
    AutogradContext::clear();
    println!("silu_autograd: PASS\n");
}

#[test]
fn test_rope_autograd() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    let q = Tensor::randn(Shape::from_dims(&[1, 4, 16, 128]), 0.0, 1.0, device.clone())
        .unwrap().requires_grad_(true);
    let k = Tensor::randn(Shape::from_dims(&[1, 4, 16, 128]), 0.0, 1.0, device.clone())
        .unwrap().requires_grad_(true);

    // Identity RoPE (cos=1, sin=0)
    let cos = Tensor::ones_dtype(Shape::from_dims(&[1, 1, 16, 64]), DType::F32, device.clone()).unwrap();
    let sin = Tensor::zeros_dtype(Shape::from_dims(&[1, 1, 16, 64]), DType::F32, device.clone()).unwrap();

    let q_out = flame_core::bf16_ops::rope_fused_bf16(&q.to_dtype(DType::BF16).unwrap(), &cos, &sin).unwrap();
    let loss = q_out.square().unwrap().mean().unwrap();
    let grads = loss.backward().unwrap();

    let g = grads.get(q.id()).expect("RoPE Q gradient missing");
    let g_abs: f32 = g.to_vec().unwrap().iter().map(|x| x.abs()).sum();
    println!("RoPE Q grad abs sum: {:.6}", g_abs);
    assert!(g_abs > 0.0, "RoPE gradient should be non-zero");

    AutogradContext::clear();
    println!("rope_autograd: PASS\n");
}

#[test]
fn test_optimizer_step() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    let w = Tensor::randn(Shape::from_dims(&[32, 32]), 0.0, 0.02, device.clone())
        .unwrap().requires_grad_(true);
    let w_id = w.id();
    let input = Tensor::ones(Shape::from_dims(&[4, 32]), device.clone()).unwrap();
    let target = Tensor::zeros(Shape::from_dims(&[4, 32]), device.clone()).unwrap();

    // Forward
    let out = input.matmul(&w.transpose().unwrap()).unwrap();
    let loss_before = out.sub(&target).unwrap().square().unwrap().mean().unwrap();
    let before = loss_before.to_vec().unwrap()[0];
    println!("Loss before: {:.6}", before);

    // Backward + step
    let grads = loss_before.backward().unwrap();
    let g = grads.get(w_id).unwrap().clone();

    // Manual gradient descent step: w -= lr * g
    let lr = 0.01;
    let w_new = w.sub(&g.mul_scalar(lr).unwrap()).unwrap();
    drop(w); // Can't reuse moved w

    // Second forward with w_new
    let out2 = input.matmul(&w_new.transpose().unwrap()).unwrap();
    let loss_after = out2.sub(&target).unwrap().square().unwrap().mean().unwrap();
    let after = loss_after.to_vec().unwrap()[0];
    println!("Loss after: {:.6}", after);
    assert!(after < before, "Loss should decrease after step");

    AutogradContext::clear();
    println!("optimizer_step: PASS\n");
}
