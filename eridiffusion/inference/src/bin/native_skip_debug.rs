use anyhow::Result;
use clap::Parser;
use std::time::Instant;

use eridiffusion_core::Device as ModelDevice;
use eridiffusion_inference::sdxl::{
    config::{SdxlPaths, SdxlRunConfig},
    native::SdxlNativePipeline,
    weights::load_resources,
};
use eridiffusion_training::sdxl::RuntimeMode;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "a quick native debug run")]
    prompt: String,
    #[arg(long, default_value = "")]
    negative_prompt: String,
    #[arg(long, default_value_t = 1)]
    steps: usize,
    #[arg(long, default_value_t = 1.0)]
    guidance_scale: f32,
    #[arg(long, default_value_t = 64)]
    height: usize,
    #[arg(long, default_value_t = 64)]
    width: usize,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(
        long,
        default_value = "/home/alex/SwarmUI/Models/Stable-Diffusion/OfficialStableDiffusion/sd_xl_base_1.0.safetensors"
    )]
    unet: String,
    #[arg(
        long,
        default_value = "/home/alex/SwarmUI/Models/VAE/OfficialStableDiffusion/sdxl_vae.safetensors"
    )]
    vae: String,
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/clip/clip_l.safetensors")]
    clip_l: String,
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/clip/clip_g.safetensors")]
    clip_g: String,
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/clip/tokenizer.json")]
    tokenizer: String,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();

    let paths = SdxlPaths {
        unet: args.unet.clone().into(),
        vae: args.vae.clone().into(),
        clip_l: args.clip_l.clone().into(),
        clip_g: args.clip_g.clone().into(),
        tokenizer: args.tokenizer.clone().into(),
    };

    run_native(args, paths, start)
}

fn run_native(args: Args, paths: SdxlPaths, start: Instant) -> Result<()> {
    let resources = load_resources(&paths, 77, args.device)?;
    let runtime_mode = RuntimeMode::Resident;
    let pipeline =
        SdxlNativePipeline::new(resources, runtime_mode, ModelDevice::Cuda(args.device))?;

    let run_cfg = SdxlRunConfig {
        steps: args.steps,
        guidance_scale: args.guidance_scale,
        height: args.height,
        width: args.width,
        seed: Some(0),
    };
    println!("[native-debug] starting sampler with cfg={:?}", run_cfg);
    let _ = pipeline.sample(
        &args.prompt,
        &args.negative_prompt,
        run_cfg.steps,
        run_cfg.guidance_scale,
        run_cfg.height,
        run_cfg.width,
        run_cfg.seed,
    );
    println!("[native-debug] finished after {:?}", start.elapsed());
    Ok(())
}
