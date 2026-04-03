use flame_core::{Tensor, Shape, Device, DType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda(0)?;
    let shape = Shape::from_dims(&[1, 64, 64, 512]);
    let t = Tensor::zeros(shape, device.cuda_device_arc())?;
    println!("Original shape: {:?}", t.shape());

    let p = t.permute(&[0, 3, 1, 2])?;
    println!("Permuted [0, 3, 1, 2] shape: {:?}", p.shape());

    Ok(())
}
