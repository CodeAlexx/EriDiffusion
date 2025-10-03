//! Example of how to use the validation system in a training loop

use cudarc::driver::CudaDevice;
use eridiffusion::trainers::{
    validation_advanced::{
        create_validation_runner, ValidationConfig, ValidationMetrics, ValidationRunner,
    },
    validation_formatter::{create_validation_formatter, ValidationFormatter},
};
use std::path::PathBuf;
use std::sync::Arc;

/// Example training loop with validation
fn training_loop_with_validation() -> anyhow::Result<()> {
    // Setup device
    let device = Arc::new(CudaDevice::new(0)?);

    // Create validation configuration
    let val_config = ValidationConfig {
        dataset_path: PathBuf::from("./validation_dataset"),
        batch_size: 4,
        every_n_steps: 500,
        num_samples: Some(40), // Use only 40 validation samples
        cache_latents: true,
        save_samples: true,
        num_generation_samples: 4,
        eval_timesteps: None, // Random timesteps
        early_stopping: Some(eridiffusion::trainers::validation_advanced::EarlyStoppingConfig {
            monitor: "val_loss".to_string(),
            min_delta: 0.0001,
            patience: 10,
            mode: eridiffusion::trainers::validation_advanced::EarlyStoppingMode::Min,
        }),
        compute_advanced_metrics: true,
    };

    // Create validation runner
    let output_dir = PathBuf::from("./validation_output");
    let mut val_runner = create_validation_runner(val_config.clone(), output_dir, device)?;

    // Create formatter for pretty output
    let formatter = create_validation_formatter();

    // Simulated training loop
    let mut best_val_loss: Option<f32> = None;

    for step in (0..=10000).step_by(500) {
        // ... training code here ...

        // Run validation every 500 steps
        if step > 0 && step % val_config.every_n_steps == 0 {
            // Print header
            formatter.print_header(step)?;

            // Simulate validation batches
            let total_batches = 10;
            for batch in 1..=total_batches {
                formatter.print_batch_progress(batch, total_batches)?;
                // ... actual validation computation ...
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            // Simulate sample generation
            for sample in 1..=4 {
                formatter.print_sample_progress(sample, 4)?;
                // ... actual sample generation ...
                std::thread::sleep(std::time::Duration::from_millis(200));
            }

            // Create mock metrics (in real usage, these come from validation)
            let metrics = create_mock_metrics(step);

            // Print results
            formatter.print_results(&metrics, "./validation_output", best_val_loss)?;

            // Update best loss
            if best_val_loss.is_none() || metrics.loss < best_val_loss.unwrap() {
                best_val_loss = Some(metrics.loss);
            }

            // Check early stopping
            if step > 5000 && metrics.loss > best_val_loss.unwrap() {
                formatter.print_early_stopping_warning(2, 10)?;
            }

            // Print completion
            formatter.print_complete(2.5)?;

            println!(); // Extra line for spacing
        }
    }

    Ok(())
}

/// Create mock metrics for demonstration
fn create_mock_metrics(step: usize) -> ValidationMetrics {
    use chrono::Utc;
    use std::collections::HashMap;

    // Simulate improving loss over time
    let base_loss = 0.1 - (step as f32 / 100000.0);
    let loss = base_loss + (rand::random::<f32>() * 0.01 - 0.005);

    // Create timestep losses
    let mut loss_per_timestep = Vec::new();
    for t in (0..1000).step_by(50) {
        let t_loss = loss + (t as f32 / 10000.0);
        loss_per_timestep.push((t, t_loss));
    }

    ValidationMetrics {
        step,
        timestamp: Utc::now(),
        loss,
        loss_per_timestep,
        num_samples: 40,
        fid_score: Some(30.0 - (step as f32 / 500.0)),
        inception_score: Some((7.5 + (step as f32 / 2000.0), 0.3)),
        clip_score: Some(0.25 + (step as f32 / 40000.0)),
        custom_metrics: HashMap::new(),
        generation_time_ms: Some(1500),
    }
}

/// Demonstrate the exact output format requested
fn demonstrate_exact_format() {
    println!("================================================================");
    println!("Running validation at step 5000");
    println!("================================================================");
    println!("Validating batch 10/10");
    println!("Generating validation samples...");
    println!("Generating sample 4/4");
    println!("Saved samples to ./validation_output/samples_step_00005000");
    println!("Validation Loss: 0.045623 (n=40), Perplexity: 1.05, FID: 25.40");
    println!();
    println!("Loss by timestep range:");
    println!("  0-200: 0.023456");
    println!("  200-400: 0.034567");
    println!("  400-600: 0.045678");
    println!("  600-800: 0.056789");
    println!("  800-1000: 0.067890");
    println!();
    println!("✅ Validation improved: 0.048234 -> 0.045623 (Δ=0.002611)");
    println!("📊 Saved validation plots to ./validation_output");
}

fn main() -> anyhow::Result<()> {
    println!("=== Validation System Example ===\n");

    // Show the exact format requested
    println!("Exact format demonstration:");
    demonstrate_exact_format();

    println!("\n\n=== Running simulated training loop ===\n");

    // Run the example training loop
    // Note: This would fail without proper setup, so we'll just show the structure
    // training_loop_with_validation()?;

    println!("\nExample complete!");
    Ok(())
}
