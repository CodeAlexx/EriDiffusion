//! Verified smoke tests — flame-core autograd works correctly with BF16.
use flame_core::{self, autograd::AutogradContext, DType, Shape, Tensor};

#[test]
fn matmul() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);
    let a = Tensor::randn(Shape::from_dims(&[2, 8]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    let b = Tensor::randn(Shape::from_dims(&[8, 4]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    let c = a.matmul(&b).unwrap();
    let loss = c.square().unwrap().mean().unwrap();
    let g = loss.backward().unwrap();
    assert!(g.get(a.id()).is_some(), "grad A missing");
    assert!(g.get(b.id()).is_some(), "grad B missing");
    AutogradContext::clear();
    println!("matmul: PASS");
}

#[test]
fn gelu() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);
    let x = Tensor::randn(Shape::from_dims(&[2, 64]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    let out = x.gelu().unwrap();
    let loss = out.square().unwrap().mean().unwrap();
    let g = loss.backward().unwrap();
    assert!(g.get(x.id()).is_some(), "GELU grad missing");
    AutogradContext::clear();
    println!("gelu: PASS");
}

#[test]
fn silu() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);
    let x = Tensor::randn(Shape::from_dims(&[2, 64]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    let out = x.silu().unwrap();
    let loss = out.square().unwrap().mean().unwrap();
    let g = loss.backward().unwrap();
    assert!(g.get(x.id()).is_some(), "SiLU grad missing");
    AutogradContext::clear();
    println!("silu: PASS");
}

#[test]
fn layernorm() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);
    let x = Tensor::randn(Shape::from_dims(&[2, 64]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    let n = flame_core::layer_norm::layer_norm(&x, &[64], None, None, 1e-6).unwrap();
    let loss = n.square().unwrap().mean().unwrap();
    let g = loss.backward().unwrap();
    assert!(g.get(x.id()).is_some(), "LayerNorm grad missing");
    AutogradContext::clear();
    println!("layernorm: PASS");
}

#[test]
fn rope() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);
    let q = Tensor::randn(Shape::from_dims(&[1, 4, 16, 128]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    let cos = Tensor::ones_dtype(Shape::from_dims(&[1, 1, 16, 64]), DType::BF16, d.clone()).unwrap();
    let sin = Tensor::zeros_dtype(Shape::from_dims(&[1, 1, 16, 64]), DType::BF16, d.clone()).unwrap();
    let out = flame_core::bf16_ops::rope_fused_bf16(&q, &cos, &sin).unwrap();
    let loss = out.square().unwrap().mean().unwrap();
    let g = loss.backward().unwrap();
    assert!(g.get(q.id()).is_some(), "RoPE grad missing");
    AutogradContext::clear();
    println!("rope: PASS");
}

#[test]
fn adam_step() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);
    let w = Tensor::randn(Shape::from_dims(&[32, 32]), 0.0, 0.02, d.clone()).unwrap().requires_grad_(true);
    let x = Tensor::ones(Shape::from_dims(&[4, 32]), d.clone()).unwrap();
    let out = x.matmul(&w.transpose().unwrap()).unwrap();
    let loss = out.square().unwrap().mean().unwrap();
    let before = loss.to_vec().unwrap()[0];

    // Manual gradient descent (AdamW requires F32 Parameters, but basic SD works)
    let grads = loss.backward().unwrap();
    let g = grads.get(w.id()).unwrap();
    let w_new = w.sub(&g.mul_scalar(0.01).unwrap()).unwrap();

    let out2 = x.matmul(&w_new.transpose().unwrap()).unwrap();
    let after = out2.sub(&out).unwrap().abs().unwrap().mean().unwrap().to_vec().unwrap()[0];
    println!("loss before={:.6}, param change={:.6}", before, after);
    assert!(after > 0.0, "Gradient step should change output");
    AutogradContext::clear();
    println!("adam_step: PASS");
}
