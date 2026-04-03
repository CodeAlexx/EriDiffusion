use anyhow::Result;
use flame_core::{DType, Device, Tensor};
use flame_core::conv::Conv2d;
use std::sync::Arc;

fn main() -> Result<()> {
    flame_core::init();
    let device = Device::cuda(0)?;

    println!("Creating Conv2d...");
    // in=128, out=3, k=3, s=1, p=1
    let mut conv = Conv2d::new(128, 3, 3, 1, 1, device.cuda_device().clone())?;

    println!("Simulating load_weights...");
    // Replace weight with new tensor (on CPU then moved to device, or just device)
    // vae.rs loads from safetensors (CPU) then converts to BF16 (Device).
    // Let's create a new tensor on device.
    let new_weight = Tensor::randn(
        flame_core::Shape::from_dims(&[3, 128, 3, 3]),
        0.0,
        1.0,
        device.cuda_device().clone(),
    )?.to_dtype(DType::BF16)?;
    
    conv.weight = new_weight;
    // Note: we are NOT updating weight_hwio_bf16, so it is stale!

    println!("Creating input...");
    // [1, 128, 512, 512]
    let input = Tensor::randn(
        flame_core::Shape::from_dims(&[1, 128, 512, 512]),
        0.0,
        1.0,
        device.cuda_device().clone(),
    )?.to_dtype(DType::BF16)?;

    println!("Running forward...");
    let output = conv.forward(&input)?;

    println!("Output shape: {:?}", output.shape().dims());

    if output.shape().dims() == &[1, 3, 512, 512] {
        println!("SUCCESS: Output shape matches expected 512x512");
    } else {
        println!("FAILURE: Output shape mismatch! Expected 512x512, got {:?}", output.shape().dims());
    }

    Ok(())
}
