use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use env_logger::Env;
use eridiffusion_core::Device as ModelDevice;
use eridiffusion_inference::sdxl::{
    config::{SdxlPaths, SdxlRunConfig},
    sampler::{save_image, SdxlSampler},
    weights::{build_pipeline, load_resources},
    SdxlNativePipeline,
};
use eridiffusion_training::sdxl::RuntimeMode;

#[derive(Parser, Debug)]
struct Args {
    #[arg(
        long,
        default_value = "a white swan on mars, red martian landscape, detailed feathers, photorealistic"
    )]
    prompt: String,
    #[arg(long, default_value = "")]
    negative_prompt: String,
    #[arg(long, default_value_t = 30)]
    steps: usize,
    #[arg(long, default_value_t = 7.5)]
    guidance_scale: f32,
    #[arg(long, default_value_t = 1024)]
    height: usize,
    #[arg(long, default_value_t = 1024)]
    width: usize,
    #[arg(long)]
    seed: Option<u64>,
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
    #[arg(long, default_value = "output/sdxl.png")]
    output: String,
    #[arg(long, value_enum, default_value = "native")]
    backend: BackendArg,
}

#[derive(Clone, Debug, ValueEnum)]
enum BackendArg {
    Native,
    Legacy,
}

fn main() -> Result<()> {
    let _ = env_logger::Builder::from_env(Env::default().default_filter_or("info")).try_init();

    let args = Args::parse();

    let paths = SdxlPaths {
        unet: args.unet.into(),
        vae: args.vae.into(),
        clip_l: args.clip_l.into(),
        clip_g: args.clip_g.into(),
        tokenizer: args.tokenizer.into(),
    };

    let run_cfg = SdxlRunConfig {
        steps: args.steps,
        guidance_scale: args.guidance_scale,
        height: args.height,
        width: args.width,
        seed: args.seed,
    };

    let runtime_mode = RuntimeMode::Resident;
    let images = match args.backend {
        BackendArg::Native => {
            let resources =
                load_resources(&paths, 77, args.device).context("failed to load SDXL resources")?;
            let pipeline =
                SdxlNativePipeline::new(resources, runtime_mode, ModelDevice::Cuda(args.device))
                    .context("failed to initialize native SDXL pipeline")?;
            let mut sampler = SdxlSampler::new_native(&pipeline);
            sampler
                .run(&args.prompt, &args.negative_prompt, &run_cfg)
                .context("native SDXL sampling failed")?
        }
        BackendArg::Legacy => {
            let mut pipeline = build_pipeline(&paths, run_cfg.steps, args.device, runtime_mode)
                .context("failed to initialize legacy SDXL pipeline")?;
            let mut sampler = SdxlSampler::new_legacy(&mut pipeline);
            sampler
                .run(&args.prompt, &args.negative_prompt, &run_cfg)
                .context("legacy SDXL sampling failed")?
        }
    };

    save_image(&images, std::path::Path::new(&args.output)).context("failed to save image")?;

    println!("Image saved to {}", args.output);
    Ok(())
}
