use eridiffusion::{
    models::sdxl::{SDXLConfig, SDXLModel},
    pipelines::sdxl::SDXLPipeline,
    schedulers::{DDIMScheduler, SchedulerConfig},
    Result,
};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("🎨 Generating SDXL image: 'a white swan on mars'");

    // Model paths
    let base_model_path =
        "/home/alex/SwarmUI/Models/diffusion_models/sd_xl_base_1.0_0.9vae.safetensors";
    let vae_path = "/home/alex/SwarmUI/Models/vae/sdxl_vae.safetensors";
    let clip_l_path = "/home/alex/SwarmUI/Models/clip/clip_l.safetensors";
    let clip_g_path = "/home/alex/SwarmUI/Models/clip/clip_g.safetensors";

    // Load SDXL model
    println!("Loading SDXL model...");
    let config = SDXLConfig {
        model_path: base_model_path.to_string(),
        vae_path: Some(vae_path.to_string()),
        clip_l_path: clip_l_path.to_string(),
        clip_g_path: clip_g_path.to_string(),
        device: "cuda:0".to_string(),
        dtype: "fp16".to_string(),
    };

    let model = SDXLModel::from_config(&config)?;

    // Create scheduler
    let scheduler_config = SchedulerConfig {
        num_train_timesteps: 1000,
        beta_start: 0.00085,
        beta_end: 0.012,
        beta_schedule: "scaled_linear".to_string(),
        prediction_type: "epsilon".to_string(),
    };
    let scheduler = DDIMScheduler::new(scheduler_config);

    // Create pipeline
    let mut pipeline = SDXLPipeline::new(model, scheduler);

    // Generate image
    let prompt = "a white swan on mars, red martian landscape, detailed feathers, photorealistic, high quality";
    let negative_prompt = "ugly, blurry, low quality, distorted, deformed";

    println!("Generating image...");
    let image = pipeline.generate(
        prompt,
        Some(negative_prompt),
        1024, // width
        1024, // height
        30,   // num_inference_steps
        7.5,  // guidance_scale
        None, // seed
    )?;

    // Save image
    let output_dir = Path::new("/home/alex/diffusers-rs/output");
    std::fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join("white_swan_on_mars.png");
    image.save(&output_path)?;

    println!("✅ Image saved to: {}", output_path.display());

    Ok(())
}
