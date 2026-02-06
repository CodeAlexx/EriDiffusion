use flame_core::{device::Device, DType, Result, Tensor};
use safetensors::SafeTensors;
use std::sync::Arc;
use memmap2::MmapOptions;
use std::fs::File;
use eridiffusion::ops::RMSNorm;

fn main() -> Result<()> {
    env_logger::init();
    let device = Device::cuda(0)?;
    
    let file = File::open("rms_norm_test.safetensors").unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&mmap).unwrap();
    
    let load = |name: &str| -> Result<Tensor> {
        let view = tensors.tensor(name).unwrap();
        let shape = flame_core::Shape::from_dims(view.shape());
        let data = view.data();
        let mut values = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            values.push(half::bf16::from_bits(bits).to_f32());
        }
        Tensor::from_vec(values, shape, device.cuda_device_arc())?.to_dtype(DType::BF16)
    };
    
    let x = load("x")?;
    let weight = load("weight")?;
    let y_ref = load("y_ref")?;
    
    let dim = x.shape().dims()[2];
    
    // Manual RMSNorm
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.square()?.mean_dim(&[2], true)?;
    let inv_rms = variance.add_scalar(1e-6)?.rsqrt()?;
    let y_norm = x_f32.mul(&inv_rms)?.to_dtype(DType::BF16)?;
    
    // Apply weight
    let weight_expanded = weight.reshape(&[1, 1, dim])?;
    let y = y_norm.mul(&weight_expanded)?;
    
    let diff = y.sub(&y_ref)?.abs()?;
    let diff_vec = diff.to_vec()?;
    let max_diff = diff_vec.iter().fold(0.0f32, |a, &b| a.max(b));
    let sum_diff: f32 = diff_vec.iter().sum();
    let mean_diff = sum_diff / diff_vec.len() as f32;
    
    println!("Max difference: {}", max_diff);
    println!("Mean difference: {}", mean_diff);
    
    if max_diff > 1e-2 {
        println!("!! SIGNIFICANT DIFFERENCE !!");
    } else {
        println!("RMSNorm parity OK");
    }
    
    Ok(())
}
