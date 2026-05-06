use flame_core::{self, autograd::AutogradContext, DType, Shape, Tensor};

#[test]
fn trace_matmul() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    let a = Tensor::randn(Shape::from_dims(&[2, 8]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    let b = Tensor::randn(Shape::from_dims(&[8, 4]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    println!("[trace] a_id={} b_id={}", a.id().0, b.id().0);

    let c = a.matmul(&b).unwrap();
    println!("[trace] c_id={}", c.id().0);
    let loss = c.square().unwrap().mean().unwrap();

    let grads = loss.backward().unwrap();
    println!("[trace] {} grads", grads.len());
    for (tid, _) in grads.iter() {
        println!("[trace]   grad_id={} (a:{} b:{})", tid.0, tid == a.id(), tid == b.id());
    }
    assert!(grads.get(a.id()).is_some(), "MatMul grad for A missing");
    assert!(grads.get(b.id()).is_some(), "MatMul grad for B missing");
    AutogradContext::clear();
}

#[test]
fn trace_layernorm() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);

    let x = Tensor::randn(Shape::from_dims(&[2, 64]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    println!("x_id={}", x.id().0);

    let n = flame_core::layer_norm::layer_norm(&x.to_dtype(DType::BF16).unwrap(), &[64], None, None, 1e-6).unwrap();
    println!("n_id={}", n.id().0);
    let loss = n.square().unwrap().mean().unwrap();

    let grads = loss.backward().unwrap();
    println!("{} grads", grads.len());
    assert!(grads.get(x.id()).is_some(), "LayerNorm grad missing");
    println!("layernorm: PASS");
    AutogradContext::clear();
}

#[test]
fn trace_rope() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);

    let q = Tensor::randn(Shape::from_dims(&[1, 4, 16, 128]), 0.0, 1.0, d.clone()).unwrap().requires_grad_(true);
    println!("q_id={}", q.id().0);

    let cos = Tensor::ones_dtype(Shape::from_dims(&[1, 1, 16, 64]), DType::BF16, d.clone()).unwrap();
    let sin = Tensor::zeros_dtype(Shape::from_dims(&[1, 1, 16, 64]), DType::BF16, d.clone()).unwrap();

    let q_out = flame_core::bf16_ops::rope_fused_bf16(&q.to_dtype(DType::BF16).unwrap(), &cos, &sin).unwrap();
    println!("q_out_id={}", q_out.id().0);
    let loss = q_out.square().unwrap().mean().unwrap();

    let grads = loss.backward().unwrap();
    println!("{} grads", grads.len());
    assert!(grads.get(q.id()).is_some(), "RoPE grad missing");
    println!("rope: PASS");
    AutogradContext::clear();
}
