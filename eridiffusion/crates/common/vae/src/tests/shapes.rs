use anyhow::Result;
use flame_core::{Device, Tensor, Shape, DType};
use eridiffusion_common_vae as vae;

#[test]
fn encode_decode_shapes_sdxl() -> Result<()> {
    let dev = Device::cuda(0)?;
    let (b,h,w,c) = (2, 128, 128, 3);
    let spec = vae::VaeSpec { kind: vae::VaeKind::Sdxl, path: "stub".into(), latent_div: 8, latent_channels: 4, latent_scale: 0.18215 };
    let imgs = Tensor::zeros_dtype(Shape::from(vec![b,h,w,c]), DType::BF16, dev.cuda_device().clone())?;
    let lat = vae::encode(&spec, &imgs, vae::VaePolicy::GpuFirst)?;
    assert_eq!(lat.shape().dims().to_vec(), vec![b, h/8, w/8, 4]);
    let rec = vae::decode(&spec, &lat, vae::VaePolicy::GpuFirst)?;
    assert_eq!(rec.shape().dims().to_vec(), vec![b, h, w, 3]);
    Ok(())
}

