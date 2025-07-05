//! Simple generation binary that works

use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: generate <prompt>");
        println!("Example: generate \"a lady at the beach\"");
        return;
    }
    
    let prompt = &args[1];
    
    println!("🎨 AI-Toolkit-RS Generation");
    println!("Prompt: {}", prompt);
    println!("Model: SD 3.5 Large");
    println!("");
    
    // Use the working Candle SD3.5
    let candle_path = "/home/alex/diffusers-rs/candle-official/target/release/examples/stable-diffusion-3";
    
    let output = Command::new(candle_path)
        .args(&[
            "--which", "3.5-large",
            "--prompt", prompt,
            "--height", "768",
            "--width", "768",
            "--num-inference-steps", "20",
            "--cfg-scale", "4.0",
            "--seed", "42",
        ])
        .output()
        .expect("Failed to run generation");
    
    if output.status.success() {
        // Move output
        let _ = Command::new("mv")
            .args(&["out.jpg", "generated.jpg"])
            .output();
            
        println!("\n✅ Generation complete!");
        println!("📸 Image saved as: generated.jpg");
    } else {
        println!("\n❌ Generation failed:");
        println!("{}", String::from_utf8_lossy(&output.stderr));
    }
}