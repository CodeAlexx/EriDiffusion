use anyhow::Result;
use flame_core::{Device, Tensor, Shape, DType};
use eridiffusion_common_vae as vae;

#[test]
fn roundtrip_smoke() -> Result<()> {
    let dev = Device::cuda(0)?;
    let (b,h,w,c) = (1, 64, 64, 3);
    let spec = vae::VaeSpec { kind: vae::VaeKind::Sdxl, path: "stub".into(), latent_div: 8, latent_channels: 4, latent_scale: 0.18215 };
    // Random-ish input
    let x = Tensor::randn(Shape::from(vec![b,h,w,c]), 0.0, 1.0, dev.cuda_device().clone())?
        .to_dtype(DType::BF16)?;
    let z = vae::encode(&spec, &x, vae::VaePolicy::GpuFirst)?;
    let y = vae::decode(&spec, &z, vae::VaePolicy::GpuFirst)?;
    // MSE finite and reasonably small for bilinear down/up path
    let xv = x.to_dtype(DType::F32)?.to_vec()?;
    let yv = y.to_dtype(DType::F32)?.to_vec()?;
    let mut mse = 0.0f32;
    for i in 0..xv.len() { let d = xv[i]-yv[i]; mse += d*d; }
    mse /= xv.len() as f32;
    assert!(mse.is_finite());
    assert!(mse < 0.2, "MSE too large: {}", mse);
    Ok(())
}

