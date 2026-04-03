use flame_core::{Tensor, Shape, Device};
use safetensors::SafeTensors;
use memmap2::MmapOptions;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/alex/SwarmUI/Models/diffusion_models/sd35-vae.safetensors";
    println!("Loading {}", path);
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    if let Ok(tensor) = tensors.tensor("decoder.conv_in.weight") {
        println!("decoder.conv_in.weight: {:?}", tensor.shape());
    } else {
        println!("decoder.conv_in.weight not found");
    }

    for name in tensors.names() {
        if name.starts_with("decoder.up_blocks") || name.starts_with("decoder.mid_block") {
            if name.contains("norm") || name.contains("weight") {
                 let tensor = tensors.tensor(name)?;
                 println!("{}: {:?}", name, tensor.shape());
            }
        }
    }

    Ok(())
}
