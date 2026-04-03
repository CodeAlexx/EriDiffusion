use flame_core::{Tensor, Shape, DType, device::Device, autograd::AutogradContext};
use eridiffusion_core::layers::{norms, mlp::{self, ActivationKind}};

fn cuda() -> Device { Device::cuda(0).unwrap() }

#[test]
fn norm_backward_parity() -> anyhow::Result<()> {
    let dev = cuda();
    let shape = Shape::from_dims(&[2, 4, 8]); // normalize over last dim (8)
    let x = Tensor::randn(shape.clone(), 0.0, 1.0, dev.clone())?.to_dtype(DType::BF16)?.requires_grad_(true);
    let gamma = Tensor::randn(Shape::from_dims(&[8]), 0.0, 1.0, dev.clone())?.requires_grad_(true);
    let beta = Tensor::randn(Shape::from_dims(&[8]), 0.0, 1.0, dev.clone())?.requires_grad_(true);

    // Unfused
    std::env::remove_var("ERID_FUSE");
    let y_ref = norms::layer_norm(&x, &gamma, &beta, 1e-5)?;
    let loss_ref = y_ref.sum()?;
    let grads_ref = AutogradContext::backward(&loss_ref)?;

    // Fused
    std::env::set_var("ERID_FUSE", "1");
    let y_fused = norms::layer_norm(&x, &gamma, &beta, 1e-5)?;
    let loss_fused = y_fused.sum()?;
    let grads_fused = AutogradContext::backward(&loss_fused)?;

    // Compare grads: x, gamma, beta
    let gx_ref = grads_ref.get(&x).unwrap().to_dtype(DType::F32)?;
    let gx_fused = grads_fused.get(&x).unwrap().to_dtype(DType::F32)?;
    let dx = gx_ref.sub(&gx_fused)?.abs()?.max_all().unwrap_or(0.0);
    assert!(dx < 1e-4, "dx max diff {}", dx);

    let gg_ref = grads_ref.get(&gamma).unwrap().to_dtype(DType::F32)?;
    let gg_fused = grads_fused.get(&gamma).unwrap().to_dtype(DType::F32)?;
    let dg = gg_ref.sub(&gg_fused)?.abs()?.max_all().unwrap_or(0.0);
    assert!(dg < 1e-4, "dg max diff {}", dg);

    let gb_ref = grads_ref.get(&beta).unwrap().to_dtype(DType::F32)?;
    let gb_fused = grads_fused.get(&beta).unwrap().to_dtype(DType::F32)?;
    let db = gb_ref.sub(&gb_fused)?.abs()?.max_all().unwrap_or(0.0);
    assert!(db < 1e-4, "db max diff {}", db);
    Ok(())
}

#[test]
fn mlp_fused_matches_unfused() -> anyhow::Result<()> {
    let dev = cuda();
    let b = 3usize; let d_in = 7usize; let d_out = 9usize;
    let x = Tensor::randn(Shape::from_dims(&[b, d_in]), 0.0, 1.0, dev.clone())?.requires_grad_(true);
    let w = Tensor::randn(Shape::from_dims(&[d_in, d_out]), 0.0, 0.5, dev.clone())?.requires_grad_(true);
    let bvec = Tensor::randn(Shape::from_dims(&[d_out]), 0.0, 0.5, dev.clone())?.requires_grad_(true);

    // Unfused
    std::env::remove_var("ERID_FUSE");
    let y_ref = mlp::mlp(&x, &w, Some(&bvec), ActivationKind::Gelu, 0.0, true, None)?;
    let l_ref = y_ref.sum()?;
    let g_ref = AutogradContext::backward(&l_ref)?;

    // Fused
    std::env::set_var("ERID_FUSE", "1");
    let y_fused = mlp::mlp(&x, &w, Some(&bvec), ActivationKind::Gelu, 0.0, true, None)?;
    let l_fused = y_fused.sum()?;
    let g_fused = AutogradContext::backward(&l_fused)?;

    let dy = y_ref.sub(&y_fused)?.abs()?.max_all().unwrap_or(0.0);
    assert!(dy < 1e-5, "y max diff {}", dy);

    let gx_ref = g_ref.get(&x).unwrap();
    let gx_fused = g_fused.get(&x).unwrap();
    let dx = gx_ref.sub(&gx_fused)?.abs()?.max_all().unwrap_or(0.0);
    assert!(dx < 1e-4, "dx max diff {}", dx);
    Ok(())
}

