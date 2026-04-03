use std::sync::Arc;
use eridiffusion_core::{TensorDebugExt, Result};
use flame_core::{Tensor, DType, Shape, CudaDevice};

fn dev() -> Arc<CudaDevice> { CudaDevice::new(0).expect("cuda").into() }

#[test]
fn debug_check_catches_nonfinite() -> Result<()> {
    std::env::set_var("TRIPWIRES", "1");
    let d = dev();
    let a = Tensor::from_vec(vec![1.0f32; 4], Shape::from_dims(&[4]), d.clone())?;
    let z = Tensor::from_vec(vec![0.0f32; 4], Shape::from_dims(&[4]), d.clone())?;
    // Force division by zero
    let bad = a.div(&z)?;
    assert!(bad.debug_check("div.zero").is_err());
    Ok(())
}

