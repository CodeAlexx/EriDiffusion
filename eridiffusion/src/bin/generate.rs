#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use flame_core::Result;
use std::process::Command;

/// Simple generation binary - deprecated, use model-specific CLIs instead

fn main() -> flame_core::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: generate <prompt>");
        println!("Example: generate \"a lady at the beach\"");
        println!("\nNOTE: This binary is deprecated. Please use:");
        println!("  - sdxl generate --prompt \"...\" for SDXL");
        println!("  - sd35 generate --prompt \"...\" for SD 3.5");
        println!("  - flux generate --prompt \"...\" for Flux");
        return Ok(());
    }

    let prompt = &args[1];

    println!("🎨 EriDiffusion Generation");
    println!("Prompt: {}", prompt);
    println!("Model: SD 3.5 Large");
    println!("");

    // This binary is deprecated - use model-specific CLIs instead
    eprintln!("This binary is deprecated. Redirecting to sd35 CLI...");

    let output = Command::new("sd35")
        .args(&["generate", "--prompt", prompt, "--variant", "large", "--output", "generated.jpg"])
        .output()?;

    if output.status.success() {
        println!("\n✅ Generation complete!");
        println!("📸 Image saved as: generated.jpg");
    } else {
        println!("\n❌ Generation failed:");
        println!("{}", String::from_utf8_lossy(&output.stderr));
    }

    Ok(())
}
