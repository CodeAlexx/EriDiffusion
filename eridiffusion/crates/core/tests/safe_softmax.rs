use std::sync::Arc;
use eridiffusion_core::{safe_ops::softmax_stable, Result};
use flame_core::{Tensor, DType, Shape, CudaDevice};

fn dev() -> Arc<CudaDevice> { CudaDevice::new(0).expect("cuda").into() }

#[test]
fn softmax_stable_is_finite_and_rows_sum_to_one() -> Result<()> {
    let d = dev();
    // Build logits linearly spaced from -1e6..1e6 across 1024 elements
    let n = 32 * 32;
    let mut v = Vec::with_capacity(n);
    for i in 0..n { let t = i as f32 / (n - 1) as f32; v.push(-1e6 + 2e6 * t); }
    let x = Tensor::from_vec(v, Shape::from_dims(&[1, 32, 32]), d.clone())?
        .to_dtype(DType::BF16)?;

    let y = softmax_stable(&x, -1)?;
    // Check finiteness via host read (test-only)
    let yv = y.to_dtype(DType::F32)?.to_vec()?;
    assert!(yv.iter().all(|f| f.is_finite()));

    // Row sums close to 1
    let s = y.to_dtype(DType::F32)?.sum_dim_keepdim(2)?; // sum over last dim
    let sv = s.to_vec()?;
    for val in sv { assert!((val - 1.0).abs() < 1e-3, "row sum deviates: {}", val); }
    Ok(())
}

