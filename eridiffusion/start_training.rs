#!/usr/bin/env rust-script
//! Simple SD3.5 LoKr training starter
//! 
//! This creates a basic training setup to test our implementation

use std::path::PathBuf;
use std::fs;

fn main() {
    println!("🚀 SD3.5 LoKr Training Setup");
    println!("============================\n");
    
    // Check for dataset
    let dataset_path = PathBuf::from("/home/alex/eridiffusion/datasets/40_woman");
    if !dataset_path.exists() {
        println!("❌ Dataset not found at: {}", dataset_path.display());
        println!("   Please ensure the 40_woman dataset is available.");
        return;
    }
    
    // Count images in dataset
    let image_count = count_images(&dataset_path);
    println!("✅ Found dataset with {} images", image_count);
    
    // Create output directories
    let output_dir = PathBuf::from("output/sd35_lokr_rust");
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    println!("✅ Created output directory: {}", output_dir.display());
    
    // Create cache directory
    let cache_dir = PathBuf::from(".cache/latents");
    fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    println!("✅ Created cache directory: {}", cache_dir.display());
    
    // Training parameters
    println!("\n📋 Training Configuration:");
    println!("   - Model: SD3.5 Medium");
    println!("   - Network: LoKr (rank=16, alpha=16)");
    println!("   - Dataset: 40_woman ({} images x 20 repeats)", image_count);
    println!("   - Batch Size: 1");
    println!("   - Learning Rate: 1e-4");
    println!("   - Steps: ~2000");
    println!("   - Resolution: 1024x1024");
    
    // Create training config file
    let config = r#"{
  "model": "sd35",
  "dataset": {
    "path": "/home/alex/40_woman",
    "resolution": 1024,
    "repeats": 20,
    "cache_latents": true
  },
  "network": {
    "type": "lokr",
    "rank": 16,
    "alpha": 16,
    "factor": 4,
    "use_tucker": true,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]
  },
  "training": {
    "batch_size": 1,
    "learning_rate": 0.0001,
    "steps": 2000,
    "warmup_steps": 100,
    "save_every": 250,
    "sample_every": 250
  },
  "output_dir": "output/sd35_lokr_rust"
}"#;
    
    let config_path = PathBuf::from("configs/sd35_lokr_rust.json");
    fs::write(&config_path, config).expect("Failed to write config");
    println!("\n✅ Created training config: {}", config_path.display());
    
    // Show next steps
    println!("\n🎯 Next Steps:");
    println!("1. Ensure you have SD3.5 model weights downloaded");
    println!("2. Build the training binary:");
    println!("   cargo build --release --example train_sd35_lokr");
    println!("3. Run training:");
    println!("   ./target/release/examples/train_sd35_lokr \\");
    println!("     --dataset-path /home/alex/40_woman \\");
    println!("     --model-path /path/to/sd35/model \\");
    println!("     --vae-path /path/to/vae \\");
    println!("     --clip-l-path /path/to/clip-l \\");
    println!("     --clip-g-path /path/to/clip-g \\");
    println!("     --t5-path /path/to/t5-xxl");
    
    println!("\n💡 For now, let's create a demo training loop...\n");
    
    // Create a simple demo
    demo_training_loop();
}

fn count_images(path: &PathBuf) -> usize {
    let valid_extensions = ["jpg", "jpeg", "png", "webp"];
    let mut count = 0;
    
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if valid_extensions.contains(&ext.to_str().unwrap_or("").to_lowercase().as_str()) {
                    count += 1;
                }
            }
        }
    }
    
    count
}

fn demo_training_loop() {
    println!("🔄 Starting demo training loop...\n");
    
    // Simulate training steps
    let total_steps = 2000;
    let save_every = 250;
    let sample_every = 250;
    
    for step in 1..=20 {
        let actual_step = step * 100;
        let loss = 0.1 * (1.0 / (step as f32).sqrt());
        
        println!("Step {}/{}: loss = {:.4}", actual_step, total_steps, loss);
        
        if actual_step % save_every == 0 {
            println!("  💾 Saving checkpoint at step {}", actual_step);
        }
        
        if actual_step % sample_every == 0 {
            println!("  🎨 Generating samples at step {}", actual_step);
        }
        
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    
    println!("\n✅ Demo training completed!");
    println!("\n📊 Summary:");
    println!("   - Total steps: 2000");
    println!("   - Final loss: ~0.022");
    println!("   - Checkpoints saved: 8");
    println!("   - Sample batches: 8");
}

