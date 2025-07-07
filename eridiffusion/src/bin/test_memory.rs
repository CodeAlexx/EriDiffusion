//! Test memory management improvements

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use eridiffusion::memory::{
    BlockSwapManager, BlockSwapConfig, BlockType,
    cuda, MemoryPoolConfig, DiffusionConfig, PrecisionMode, AttentionStrategy
};
use std::path::PathBuf;

fn main() -> Result<()> {
    env_logger::init();
    
    println!("Testing memory management improvements...");
    
    // Test 1: Memory pool
    test_memory_pool()?;
    
    // Test 2: Block swapping
    test_block_swapping()?;
    
    println!("\nAll tests passed!");
    Ok(())
}

fn test_memory_pool() -> Result<()> {
    println!("\n=== Testing Memory Pool ===");
    
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Get memory pool
    let pool = cuda::get_memory_pool(0)?;
    
    // Check memory stats
    let stats = pool.read().unwrap().get_stats();
    println!("Initial memory stats:");
    println!("  Allocated: {} MB", stats.allocated_bytes / 1024 / 1024);
    println!("  Reserved: {} MB", stats.reserved_bytes / 1024 / 1024);
    
    // Test allocation
    let size = 100 * 1024 * 1024; // 100MB
    println!("\nAllocating 100MB...");
    
    // Create a tensor to test allocation
    let tensor = Tensor::zeros(&[1024, 1024, 25], DType::F32, &device)?;
    
    // Check memory after allocation
    let stats = pool.read().unwrap().get_stats();
    println!("After allocation:");
    println!("  Allocated: {} MB", stats.allocated_bytes / 1024 / 1024);
    println!("  Reserved: {} MB", stats.reserved_bytes / 1024 / 1024);
    
    // Test empty cache
    println!("\nEmptying cache...");
    cuda::empty_cache()?;
    
    let stats = pool.read().unwrap().get_stats();
    println!("After empty_cache:");
    println!("  Allocated: {} MB", stats.allocated_bytes / 1024 / 1024);
    println!("  Reserved: {} MB", stats.reserved_bytes / 1024 / 1024);
    
    Ok(())
}

fn test_block_swapping() -> Result<()> {
    println!("\n=== Testing Block Swapping ===");
    
    let device = Device::cuda_if_available(0)?;
    
    // Create block swap config
    let config = BlockSwapConfig {
        max_gpu_memory: 1 * 1024 * 1024 * 1024, // 1GB for testing
        swap_dir: PathBuf::from("/tmp/test_block_swap"),
        active_blocks: 4,
        ..Default::default()
    };
    
    println!("Creating block swap manager...");
    let manager = BlockSwapManager::new(config)?;
    
    // Create test tensors
    println!("Creating test tensors...");
    let tensor1 = Tensor::randn(0.0f32, 1.0, &[512, 512], &device)?;
    let tensor2 = Tensor::randn(0.0f32, 1.0, &[512, 512], &device)?;
    let tensor3 = Tensor::randn(0.0f32, 1.0, &[512, 512], &device)?;
    
    // Register tensors as blocks
    println!("Registering tensors as swappable blocks...");
    manager.register_tensor("block1".to_string(), &tensor1, BlockType::Attention)?;
    manager.register_tensor("block2".to_string(), &tensor2, BlockType::MLP)?;
    manager.register_tensor("block3".to_string(), &tensor3, BlockType::LayerNorm)?;
    
    // Access blocks (should trigger swapping if needed)
    println!("Accessing blocks...");
    let _t1 = manager.access_block("block1")?;
    let _t2 = manager.access_block("block2")?;
    let _t3 = manager.access_block("block3")?;
    
    // Get stats
    let stats = manager.get_stats();
    println!("Block swap stats:");
    println!("  Total swaps: {}", stats.total_swaps);
    println!("  GPU->CPU: {}", stats.gpu_to_cpu);
    println!("  CPU->GPU: {}", stats.cpu_to_gpu);
    println!("  CPU->Disk: {}", stats.cpu_to_disk);
    println!("  Disk->CPU: {}", stats.disk_to_cpu);
    
    // Test Flux blocks
    println!("\nBuilding Flux blocks...");
    let flux_blocks = BlockSwapManager::build_flux_blocks(19, 38, 3072);
    println!("Created {} Flux blocks", flux_blocks.len());
    
    // Show some block info
    for (i, block) in flux_blocks.iter().take(5).enumerate() {
        println!("  Block {}: {} ({}MB)", i, block.id, block.size_bytes / 1024 / 1024);
    }
    
    Ok(())
}