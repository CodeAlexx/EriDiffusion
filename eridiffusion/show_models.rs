//! Simple script to demonstrate all supported models

fn main() {
    println!("🎨 AI-Toolkit Supported Models");
    println!("==============================\n");
    
    let models = vec![
        // Image Generation Models
        ("SD 1.5", "sd15", "512x512", "50 steps", "DDIM", "Classic Stable Diffusion"),
        ("SDXL", "sdxl", "1024x1024", "40 steps", "DDIM", "High-resolution SD"),
        ("SD 3", "sd3", "1024x1024", "28 steps", "Flow", "Multimodal diffusion"),
        ("SD 3.5", "sd35", "1024x1024", "28 steps", "Flow", "Latest SD version"),
        ("Flux", "flux", "1024x1024", "20 steps", "Flow", "State-of-the-art flow model"),
        ("Flux-Schnell", "flux-schnell", "1024x1024", "4 steps", "Flow", "Fast 4-step generation"),
        ("Flux-Dev", "flux-dev", "1024x1024", "50 steps", "Flow", "Development version"),
        ("PixArt-α", "pixart-alpha", "1024x1024", "25 steps", "DDIM", "Efficient DiT model"),
        ("PixArt-Σ", "pixart-sigma", "1024x1024", "25 steps", "DDIM", "Improved PixArt"),
        ("AuraFlow", "auraflow", "1024x1024", "30 steps", "Flow", "Flow-based generation"),
        ("HiDream", "hidream", "768x768", "40 steps", "DDIM", "High-quality dreamy images"),
        ("KonText", "kontext", "1024x1024", "25 steps", "Flow", "Contextual control for Flux"),
        ("OmniGen 2", "omnigen2", "1024x1024", "30 steps", "DDIM", "Multi-modal generation"),
        ("Flex 1", "flex1", "768x768", "35 steps", "DDIM", "Flexible generation v1"),
        ("Flex 2", "flex2", "1024x1024", "30 steps", "DDIM", "Flexible generation v2"),
        ("Chroma", "chroma", "768x768", "40 steps", "DDIM", "Color-focused model"),
        ("Lumina", "lumina", "1024x1024", "30 steps", "Flow", "Luminous generations"),
        
        // Video Generation Models
        ("Wan 2.1", "wan21", "1024x576", "25 steps", "Flow", "Video model (uses Flux VAE)"),
        ("LTX", "ltx", "768x512", "25 steps", "Flow", "Latent Text-to-Video"),
        ("Hunyuan Video", "hunyuan-video", "1280x720", "30 steps", "Flow", "HD video generation"),
    ];
    
    println!("📊 {} Total Models Supported\n", models.len());
    
    println!("🖼️  Image Generation Models:");
    println!("{:-<80}", "");
    println!("{:<15} {:<15} {:<12} {:<10} {:<8} {:<30}", 
        "Model", "Code", "Resolution", "Steps", "Type", "Description");
    println!("{:-<80}", "");
    
    for (i, (name, code, res, steps, sched, desc)) in models[..17].iter().enumerate() {
        println!("{:<15} {:<15} {:<12} {:<10} {:<8} {:<30}", 
            name, code, res, steps, sched, desc);
    }
    
    println!("\n🎬 Video Generation Models:");
    println!("{:-<80}", "");
    println!("{:<15} {:<15} {:<12} {:<10} {:<8} {:<30}", 
        "Model", "Code", "Resolution", "Steps", "Type", "Description");
    println!("{:-<80}", "");
    
    for (name, code, res, steps, sched, desc) in models[17..].iter() {
        println!("{:<15} {:<15} {:<12} {:<10} {:<8} {:<30}", 
            name, code, res, steps, sched, desc);
    }
    
    println!("\n🔧 Model Features:");
    println!("- All models support LoRA/LoKr/DoRA adapters");
    println!("- Video models generate temporal sequences");
    println!("- Flow models use flow matching instead of DDPM");
    println!("- Schnell variant runs in just 4 steps");
    println!("- KonText provides contextual control for Flux");
    
    println!("\n💾 Memory Requirements:");
    println!("- SD 1.5: ~1GB");
    println!("- SDXL: ~6GB");
    println!("- SD3/3.5: ~8GB");
    println!("- Flux: ~12GB");
    println!("- Hunyuan Video: ~16GB");
    
    println!("\n🚀 Example Usage:");
    println!("cargo run --example generate_all_models -- --models sd35,flux");
    println!("cargo run --example real_generation -- --model-dir ./models/sd35 --model sd35");
}