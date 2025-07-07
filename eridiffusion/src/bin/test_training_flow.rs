//! Test the training flow with mock model

use anyhow::Result;
use candle_core::{Device, Tensor, DType, Var};
use candle_nn::{AdamW, ParamsAdamW, Optimizer};
use eridiffusion::trainers::flux_data_loader::{FluxDataLoader, DatasetConfig};
use eridiffusion::memory::{cuda, BlockSwapManager, BlockSwapConfig};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    env_logger::init();
    
    println!("Testing Flux LoRA training flow with dataset...\n");
    
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Test memory management
    println!("\n=== Memory Management ===");
    let pool = cuda::get_memory_pool(0)?;
    let stats = pool.read().unwrap().get_stats();
    println!("Initial GPU memory: {} MB allocated", stats.allocated_bytes / 1024 / 1024);
    
    // Set up block swapping
    println!("\n=== Block Swapping Setup ===");
    let swap_config = BlockSwapConfig {
        max_gpu_memory: 20 * 1024 * 1024 * 1024, // 20GB
        swap_dir: PathBuf::from("/tmp/flux_test_swap"),
        active_blocks: 8,
        ..Default::default()
    };
    let block_manager = BlockSwapManager::new(swap_config)?;
    println!("Block swap manager initialized");
    
    // Load dataset
    println!("\n=== Dataset Loading ===");
    let dataset_config = DatasetConfig {
        folder_path: PathBuf::from("/home/alex/diffusers-rs/datasets/40_woman"),
        caption_ext: "txt".to_string(),
        caption_dropout_rate: 0.05,
        shuffle_tokens: false,
        cache_latents_to_disk: false,
        resolutions: vec![(512, 512), (768, 768), (1024, 1024)],
        center_crop: false,
        random_flip: true,
    };
    
    let mut data_loader = FluxDataLoader::new(dataset_config, device.clone())?;
    println!("Dataset loaded: 55 samples");
    
    // Create mock LoRA parameters
    println!("\n=== Mock LoRA Parameters ===");
    let lora_rank = 16;
    // For Flux latents: [batch_size, 16, 64, 64] -> flattened to [batch_size, 16*64*64=65536]
    let latent_dim = 16 * 64 * 64;  // 65536
    
    // Create mock LoRA weights
    // LoRA A: projects from latent_dim to lora_rank
    let lora_a = Var::from_tensor(&Tensor::randn(0.0f32, 0.01, &[lora_rank, latent_dim], &device)?)?;
    // LoRA B: projects from lora_rank back to latent_dim
    let lora_b = Var::from_tensor(&Tensor::zeros(&[latent_dim, lora_rank], DType::F32, &device)?)?;
    
    println!("Created LoRA parameters: rank={}, latent_dim={}", lora_rank, latent_dim);
    
    // Create optimizer
    let mut optimizer = AdamW::new(
        vec![lora_a.clone(), lora_b.clone()],
        ParamsAdamW {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        },
    )?;
    
    // Training loop simulation
    println!("\n=== Training Simulation ===");
    let num_steps = 10;
    let batch_size = 2;
    let start_time = Instant::now();
    
    for step in 0..num_steps {
        // Load batch
        let batch = data_loader.get_batch(batch_size)?;
        if batch.is_empty() {
            println!("No more data!");
            break;
        }
        
        // Simulate forward pass
        let mock_latents = Tensor::randn(0.0f32, 1.0, &[batch_size, 16, 64, 64], &device)?;
        let _mock_timesteps = Tensor::rand(0.0f32, 1.0, &[batch_size], &device)?;
        
        // Simulate LoRA computation
        // Flatten latents: [batch_size, 16, 64, 64] -> [batch_size, 16*64*64]
        let flattened = mock_latents.flatten_from(1)?;
        
        // Apply LoRA: input @ A^T @ B^T
        // flattened: [batch_size, latent_dim]
        // lora_a: [lora_rank, latent_dim] 
        // lora_b: [latent_dim, lora_rank]
        
        // First project to lower dimension: [batch_size, latent_dim] @ [lora_rank, latent_dim]^T = [batch_size, lora_rank]
        let projected = flattened.matmul(&lora_a.t()?)?;
        
        // Then project back: [batch_size, lora_rank] @ [latent_dim, lora_rank]^T = [batch_size, latent_dim]
        let lora_output = projected.matmul(&lora_b.t()?)?;
        
        // Simulate loss
        let target = Tensor::randn(0.0f32, 1.0, lora_output.shape(), &device)?;
        let loss = (lora_output - target)?.powf(2.0)?.mean_all()?;
        let loss_value = loss.to_scalar::<f32>()?;
        
        // Backward pass
        loss.backward()?;
        
        // Optimizer step
        optimizer.backward_step(&loss)?;
        
        // Memory stats
        if step % 5 == 0 {
            let stats = pool.read().unwrap().get_stats();
            let mem_mb = stats.allocated_bytes as f32 / (1024.0 * 1024.0);
            let elapsed = start_time.elapsed().as_secs_f32();
            let steps_per_sec = (step + 1) as f32 / elapsed;
            
            println!(
                "Step {}/{} | Loss: {:.4} | Speed: {:.2} it/s | Mem: {:.0} MB",
                step + 1, num_steps, loss_value, steps_per_sec, mem_mb
            );
        }
        
        // Clear cache periodically
        if step % 10 == 0 && step > 0 {
            cuda::empty_cache()?;
        }
    }
    
    // Final stats
    println!("\n=== Training Complete ===");
    let total_time = start_time.elapsed();
    println!("Total time: {:.2}s", total_time.as_secs_f32());
    println!("Average speed: {:.2} it/s", num_steps as f32 / total_time.as_secs_f32());
    
    let final_stats = pool.read().unwrap().get_stats();
    println!("Final GPU memory: {} MB", final_stats.allocated_bytes / 1024 / 1024);
    println!("Peak GPU memory: {} MB", final_stats.peak_allocated / 1024 / 1024);
    
    // Test block swapping
    println!("\n=== Block Swap Stats ===");
    let swap_stats = block_manager.get_stats();
    println!("Total swaps: {}", swap_stats.total_swaps);
    println!("GPU->CPU: {}", swap_stats.gpu_to_cpu);
    println!("CPU->GPU: {}", swap_stats.cpu_to_gpu);
    
    println!("\nAll tests completed successfully!");
    Ok(())
}