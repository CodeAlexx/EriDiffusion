//! Debug: print gradient map IDs to find why lookups fail.
use flame_core::{self, autograd::AutogradContext, DType, Shape, Tensor, gradient::GradientMap};

#[test]
fn debug_gelu_grads() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    let x = Tensor::randn(Shape::from_dims(&[2, 8]), 0.0, 1.0, device.clone())
        .unwrap().requires_grad_(true);
    let x_id = x.id();
    println!("GELU: x_id={}", x_id.0);

    let out = x.gelu().unwrap();
    println!("GELU: out_id={}", out.id().0);
    let loss = out.square().unwrap().mean().unwrap();
    println!("GELU: loss_id={}", loss.id().0);

    let grads = loss.backward().unwrap();
    println!("GELU: {} grads in map", grads.len());
    let mut found = false;
    for (tid, _) in grads.iter() {
        println!("  grad_ids={} (matches x: {})", tid.0, tid == x_id);
        if tid == x_id { found = true; }
    }
    assert!(found, "GELU gradient for x not found in gradient map");
    AutogradContext::clear();
}

#[test]
fn debug_matmul_grads() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    let a = Tensor::randn(Shape::from_dims(&[2, 8]), 0.0, 1.0, device.clone())
        .unwrap().requires_grad_(true);
    let a_id = a.id();
    let b = Tensor::randn(Shape::from_dims(&[8, 4]), 0.0, 1.0, device.clone())
        .unwrap().requires_grad_(true);
    let b_id = b.id();
    let c = a.matmul(&b).unwrap();
    println!("MATMUL: a_id={} b_id={} c_id={}", a_id.0, b_id.0, c.id().0);
    let loss = c.square().unwrap().mean().unwrap();

    let grads = loss.backward().unwrap();
    println!("MATMUL: {} grads in map", grads.len());
    let (mut found_a, mut found_b) = (false, false);
    for (tid, _) in grads.iter() {
        println!("  grad_ids={} (a:{} b:{})", tid.0, tid == a_id, tid == b_id);
        if tid == a_id { found_a = true; }
        if tid == b_id { found_b = true; }
    }
    assert!(found_a, "MatMul gradient for A not found");
    assert!(found_b, "MatMul gradient for B not found");
    AutogradContext::clear();
}
