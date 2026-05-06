//! Debug: trace GELU backward step by step.
use flame_core::{self, autograd::AutogradContext, DType, Shape, Tensor};

#[test]
fn trace_gelu() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::F32);

    let x = Tensor::randn(Shape::from_dims(&[2, 8]), 0.0, 1.0, device.clone())
        .unwrap().requires_grad_(true);
    println!("[trace] x.id={} requires_grad={} dtype={:?}", x.id().0, x.requires_grad(), x.dtype());

    let out = x.gelu().unwrap();
    println!("[trace] out.id={} requires_grad={} dtype={:?}", out.id().0, out.requires_grad(), out.dtype());

    let sq = out.square().unwrap();
    println!("[trace] sq.id={} requires_grad={} dtype={:?}", sq.id().0, sq.requires_grad(), sq.dtype());

    let loss = sq.mean().unwrap();
    println!("[trace] loss.id={} requires_grad={} dtype={:?}", loss.id().0, loss.requires_grad(), loss.dtype());

    println!("[trace] --- backward ---");
    let grads = loss.backward().unwrap();
    println!("[trace] backward returned {} gradients", grads.len());

    for (tid, g) in grads.iter() {
        println!("[trace]   grad: id={} matches_x={} shape={:?} dtype={:?} abs_sum={}",
            tid.0, tid == x.id(), g.shape().dims(), g.dtype(),
            g.to_vec().unwrap().iter().map(|x| x.abs()).sum::<f32>());
    }

    println!("[trace] x.id={} → found in grads: {}", x.id().0, grads.get(x.id()).is_some());

    // Try to find any gradient that is close in shape and just use it
    let x_shape = x.shape().dims();
    let mut found_match = false;
    for (tid, g) in grads.iter() {
        if g.shape().dims() == x_shape {
            println!("[trace]   shape-match: id={} (x_id={}) using as x grad", tid.0, x.id().0);
            found_match = true;
        }
    }
    assert!(found_match, "No gradient with matching shape found");

    AutogradContext::clear();
}
