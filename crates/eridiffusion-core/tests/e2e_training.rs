//! End-to-end training test: model -> loss -> backward -> optimizer.step -> verify loss decreases.
//! Tests the CRITICAL path that gradient pipeline actually works.
use flame_core::{self, autograd::AutogradContext, DType, Shape, Tensor, parameter::Parameter};
use flame_core::adam::AdamW;

/// Test that Parameter gradients flow correctly through backward → optimizer.step
#[test]
fn e2e_training_converges() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);

    // Create a small 2-layer FC model with Parameters (F32 storage for AdamW)
    let w1 = Parameter::new(
        Tensor::randn(Shape::from_dims(&[64, 32]), 0.0, 0.02, device.clone())
            .unwrap().requires_grad_(true)
    );
    let b1 = Parameter::new(
        Tensor::zeros_dtype(Shape::from_dims(&[64]), DType::F32, device.clone())
            .unwrap().requires_grad_(true)
    );
    let w2 = Parameter::new(
        Tensor::randn(Shape::from_dims(&[32, 64]), 0.0, 0.02, device.clone())
            .unwrap().requires_grad_(true)
    );
    let b2 = Parameter::new(
        Tensor::zeros_dtype(Shape::from_dims(&[32]), DType::F32, device.clone())
            .unwrap().requires_grad_(true)
    );

    let params = vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()];

    // Fixed input/target
    let input = Tensor::randn(Shape::from_dims(&[4, 32]), 0.0, 1.0, device.clone()).unwrap();
    let target = Tensor::randn(Shape::from_dims(&[4, 32]), 0.0, 1.0, device.clone()).unwrap();

    let mut opt = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0);
    let mut losses = Vec::new();
    let mut grad_norms = Vec::new();

    for step in 0..20 {
        // ---- Forward ----
        let w1t = w1.tensor().unwrap();
        let h = input.matmul(&w1t.transpose().unwrap()).unwrap();
        let h = h.add(&b1.tensor().unwrap().unsqueeze(0).unwrap()).unwrap();
        let h = h.relu().unwrap();

        let w2t = w2.tensor().unwrap();
        let out = h.matmul(&w2t.transpose().unwrap()).unwrap();
        let out = out.add(&b2.tensor().unwrap().unsqueeze(0).unwrap()).unwrap();

        let diff = out.sub(&target).unwrap();
        let loss = diff.square().unwrap().mean().unwrap();
        let loss_val = loss.to_dtype(DType::F32).unwrap().to_vec().unwrap()[0];
        losses.push(loss_val);

        // ---- Backward ----
        let grads = loss.backward().unwrap();
        let grad_count = grads.len();

        // Track gradient norm
        let mut total_norm = 0.0f32;
        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                let g_f32 = g.to_dtype(DType::F32).unwrap();
                let sq = g_f32.square().unwrap().sum().unwrap().to_vec().unwrap()[0];
                total_norm += sq;
                param.set_grad(g.to_dtype(DType::F32).unwrap()).unwrap();
            }
        }
        let gnorm = total_norm.sqrt();
        grad_norms.push(gnorm);
        assert!(gnorm.is_finite(), "Gradient norm NaN/inf at step {}", step);
        assert!(gnorm < 1000.0, "Gradient explosion at step {}: norm={}", step, gnorm);

        // ---- Optimizer step ----
        {
            let _guard = AutogradContext::no_grad();
            opt.step(&params).unwrap();
            opt.zero_grad(&params);
        }
        AutogradContext::clear();

        if step % 5 == 0 || step == 19 {
            println!("step {:2}: loss={:.6} grad_count={}", step, loss_val, grad_count);
        }
    }

    println!("loss progression: {:?}", losses);
    assert!(losses[0].is_finite(), "Initial loss should be finite");
    assert!(losses[19] < losses[0] * 0.95, "Loss should decrease by >5% over 20 steps");
    assert!(losses[19] < losses[0], "Loss should decrease");
    for i in 1..losses.len() {
        assert!(losses[i].is_finite(), "Loss at step {} should be finite", i);
    }
    println!("e2e_training_converges: PASS (loss: {:.6} -> {:.6})", losses[0], losses[19]);
}
