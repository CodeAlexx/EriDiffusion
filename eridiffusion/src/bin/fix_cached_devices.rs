//! Fix cached latents with wrong DeviceIds by loading and re-saving them
//! This is much faster than re-encoding through VAE

use anyhow::{Result, Context};
use candle_core::{Device, Tensor};
use std::path::{Path, PathBuf};
use std::fs;
use safetensors::{load_file, save_file};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Cached Latent Device Fixer ===");
    
    // Get the single cached device
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);
    
    // Find all .latent_cache directories
    let cache_dirs = find_cache_dirs("/home/alex/diffusers-rs/datasets")?;
    
    if cache_dirs.is_empty() {
        println!("No cache directories found!");
        return Ok(());
    }
    
    println!("Found {} cache directories", cache_dirs.len());
    
    for cache_dir in cache_dirs {
        println!("\nProcessing cache: {}", cache_dir.display());
        fix_cache_directory(&cache_dir, &device)?;
    }
    
    println!("\n=== All caches fixed! ===");
    Ok(())
}

fn find_cache_dirs(root: &str) -> Result<Vec<PathBuf>> {
    let mut cache_dirs = Vec::new();
    
    fn visit_dirs(dir: &Path, cache_dirs: &mut Vec<PathBuf>) -> Result<()> {
        if dir.file_name() == Some(std::ffi::OsStr::new(".latent_cache")) {
            cache_dirs.push(dir.to_path_buf());
            return Ok(());
        }
        
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dirs(&path, cache_dirs)?;
                }
            }
        }
        Ok(())
    }
    
    visit_dirs(Path::new(root), &mut cache_dirs)?;
    Ok(cache_dirs)
}

fn fix_cache_directory(cache_dir: &Path, device: &Device) -> Result<()> {
    let entries = fs::read_dir(cache_dir)?;
    let mut fixed_count = 0;
    let mut total_count = 0;
    
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension() == Some(std::ffi::OsStr::new("safetensors")) {
            total_count += 1;
            
            // Load the tensors
            let tensors = load_file(&path, &Device::Cpu)?;
            
            // Check if any tensor needs fixing
            let mut needs_fix = false;
            for (name, tensor) in &tensors {
                // Check device - if it's not on our target device, we need to fix it
                match tensor.device() {
                    Device::Cuda(cuda_device) => {
                        // The tensor is on CUDA, but might be wrong device ID
                        needs_fix = true;
                        println!("  {} has tensor '{}' on {:?}", 
                            path.file_name().unwrap().to_string_lossy(),
                            name,
                            tensor.device()
                        );
                    }
                    Device::Cpu => {
                        // CPU tensors are fine, we'll move them to CUDA when loading
                    }
                    _ => {}
                }
            }
            
            if needs_fix {
                // Move all tensors to CPU first (safetensors format stores on CPU)
                let mut cpu_tensors = HashMap::new();
                for (name, tensor) in tensors {
                    let cpu_tensor = tensor.to_device(&Device::Cpu)?;
                    cpu_tensors.insert(name, cpu_tensor);
                }
                
                // Save back to file
                save_file(&cpu_tensors, &path)?;
                fixed_count += 1;
                println!("  ✓ Fixed {}", path.file_name().unwrap().to_string_lossy());
            }
        }
    }
    
    println!("Fixed {}/{} files in {}", 
        fixed_count, 
        total_count,
        cache_dir.file_name().unwrap().to_string_lossy()
    );
    
    Ok(())
}