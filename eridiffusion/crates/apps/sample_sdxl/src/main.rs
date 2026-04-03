use anyhow::{Result, Context, bail};
use clap::Parser;
use tracing::info;
use flame_core::{Device, Tensor, Shape, DType};
use eridiffusion_common_text::{HfTokenizer, clip_l::ClipL, openclip_g::OpenClipG};
use eridiffusion_models_sdxl::SdxlUnet;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)] run: String,
    #[arg(long)] prompt: String,
    #[arg(long, default_value_t=30)] steps: usize,
    #[arg(long, default_value_t=7.5)] guidance: f32,
    #[arg(long, default_value_t=1024)] height: usize,
    #[arg(long, default_value_t=1024)] width: usize,
    #[arg(long, default_value_t=42)] seed: u64,
    #[arg(long, default_value="out.png")] out: String,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct RunSpec { model_id: String, arch: String, model_paths: String }
#[derive(serde::Deserialize, Debug, Clone)]
struct DataSpec { seq: usize }
#[derive(serde::Deserialize, Debug, Clone)]
struct TrainConfigV1 { run: RunSpec, data: DataSpec }

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // 0) Resolve config + paths
    let f = std::fs::File::open(&args.run)?;
    let cfg_value: serde_yaml::Value = serde_yaml::from_reader(f)?;
    let cfg: TrainConfigV1 = serde_yaml::from_value(cfg_value.clone())?;
    let mp  = eridiffusion_common_config::load(&cfg.run.model_paths)?;
    let mp_entry = mp.models.get(&cfg.run.model_id).ok_or_else(|| anyhow::anyhow!("unknown model_id"))?;
    let arch = if cfg.run.arch == "auto" { "sdxl".to_string() } else { cfg.run.arch.clone() };
    info!("resolved model_id → arch: {} → {}", cfg.run.model_id, arch);
    info!("effective paths: {:?}", mp_entry);

    // 1) Device/registry
    let device = Device::cuda(0)?;
    let mut reg = eridiffusion_common_weights::ParamRegistry::new();

    // 2) Load UNet weights strictly: prefix histogram
    let unet_path = mp_entry.weights.as_ref().context("missing weights in modelPath entry")?;
    let mut ld_unet = eridiffusion_common_weights::SafeLoader::open(unet_path)?;
    let keys = ld_unet.list_keys()?;
    let enc = keys.iter().filter(|k| k.starts_with("encoder.")).count();
    info!("prefix histogram: encoder.*={} / total={}", enc, keys.len());
    if enc * 2 > keys.len() { bail!("Refusing text-encoder checkpoint for image backbone"); }

    // 3) Build UNet
    let unet = SdxlUnet::from_loader(&cfg_value, &mut ld_unet, &mut reg, &device)?;

    // 4) Text encoders (real)
    let seq = cfg.data.seq;
    // Tokenizer path: pick first declared text encoder as source of tokenizer.json if not explicit
    let tok_path = mp.text_encoders.values().next().map(|te| te.path.clone()).unwrap_or_else(|| "".into());
    let tok = HfTokenizer::from_path(&tok_path, seq)?;
    let (ids_c, lens_c, _pad) = tok.encode_batch(&[args.prompt.clone()])?;
    let (ids_u, lens_u, _pad2) = tok.encode_batch(&["".to_string()])?;
    // Basic sanity: same sequence policy
    let lc0 = lens_c.to_dtype(DType::F32)?.item()?;
    let lu0 = lens_u.to_dtype(DType::F32)?.item()?;
    if (lc0 - lu0).abs() > 1e-6 { tracing::warn!("cond/uncond lengths differ: {:.1} vs {:.1}", lc0, lu0); }

    // Build encoders from modelPath fields (reuse weights_txt/img as placeholders for TE paths)
    let te1_path = mp_entry.weights_txt.clone().context("missing te1 path in modelPath entry (weights_txt)")?;
    let te2_path = mp_entry.weights_img.clone().unwrap_or_else(|| te1_path.clone());
    let te1 = ClipL::from_weights_auto(&te1_path, &device, seq)?;
    let te2 = OpenClipG::from_weights_auto(&te2_path, &device, seq)?;
    let ctx1_c = te1.forward(&ids_c)?;
    let ctx2_c = te2.forward(&ids_c)?;
    let ctx1_u = te1.forward(&ids_u)?;
    let ctx2_u = te2.forward(&ids_u)?;
    let m1 = ctx1_c.to_dtype(DType::F32)?.to_vec()?;
    let m2 = ctx2_c.to_dtype(DType::F32)?.to_vec()?;
    let (mean1, std1) = mean_std(&m1);
    let (mean2, std2) = mean_std(&m2);
    info!("ctx1(cond) mean/std: {:.5}/{:.5}", mean1, std1);
    info!("ctx2(cond) mean/std: {:.5}/{:.5}", mean2, std2);

    // 5) Latents init + schedule
    if args.guidance > 12.0 { tracing::warn!("guidance {:.2} too high; clamping to 12.0", args.guidance); }
    let guidance = args.guidance.min(12.0);
    let (h, w) = (args.height/8, args.width/8);
    let mut x = eridiffusion_common_sampler::randn_nhwc_bf16([1,h,w,4], args.seed, &device)?;
    let sched = eridiffusion_common_sampler::Karras::new(args.steps);
    info!("sigma_max={:.5}", sched.sigma_max());
    x = x.mul_scalar(sched.sigma_max())?;

    // 6) Denoise loop (Euler-A with CFG)
    for (i, (sigma, t)) in sched.iter().enumerate() {
        let x2 = flame_core::Tensor::stack(&[x.clone(), x.clone()], 0)?;
        let t2 = eridiffusion_common_sampler::timestep_batch(&device, t, 2)?;
        let c1_2 = flame_core::Tensor::stack(&[ctx1_c.clone(), ctx1_u.clone()], 0)?;
        let c2_2 = flame_core::Tensor::stack(&[ctx2_c.clone(), ctx2_u.clone()], 0)?;
        let eps2 = unet.eps(&x2, &t2, &c1_2, &c2_2, &lens_c)?;
        let (eps_c, eps_u) = eridiffusion_common_sampler::split_rows2(&eps2)?;
        let eps_cfg = eps_u.add(&(eps_c.sub(&eps_u)?).mul_scalar(guidance)?)?;
        x = eridiffusion_common_sampler::euler_a_step(x, eps_cfg, sigma, &sched)?;
        println!(r#"{{"step":{},"sigma":{},"vram_gb":{:.2}}}"#, i, sigma, eridiffusion_common_weights::vram_used_gb()?);
    }

    // 7) Decode and save
    // We don't have explicit tokenizer/TE paths in ModelSpec; VAE spec is already supported in modelPath
    let vae_spec = eridiffusion_common_config::resolve_vae_block(mp_entry)?; // returns VaeBlock (cfg side)
    let spec = eridiffusion_common_vae::VaeSpec { 
        kind: match vae_spec.kind { eridiffusion_common_config::VaeKind::Sdxl => eridiffusion_common_vae::VaeKind::Sdxl,
                                     eridiffusion_common_config::VaeKind::Sd35 => eridiffusion_common_vae::VaeKind::Sd35,
                                     eridiffusion_common_config::VaeKind::Flux => eridiffusion_common_vae::VaeKind::Flux },
        path: vae_spec.path.clone(), latent_div: vae_spec.latent_div, latent_channels: vae_spec.latent_channels, latent_scale: vae_spec.latent_scale };
    let rgb = eridiffusion_common_vae::decode(&spec, &x, eridiffusion_common_vae::VaePolicy::GpuFirst)?;
    let vals = rgb.to_dtype(DType::F32)?.to_vec()?;
    if !vals.is_empty() {
        let (mut mn, mut mx, mut sum, mut sq) = (vals[0], vals[0], 0.0f32, 0.0f32);
        for &v in &vals { mn = mn.min(v); mx = mx.max(v); sum += v; sq += v*v; }
        let n = vals.len() as f32;
        let mean = sum / n; let std = ((sq / n) - mean*mean).max(0.0).sqrt();
        tracing::info!("image mean/std/min/max: {:.5}/{:.5}/{:.5}/{:.5}", mean, std, mn, mx);
    }
    eridiffusion_common_sampler::save_png_nhwc_bf16(&rgb, &args.out)?;
    Ok(())
}

fn mean_std(v: &[f32]) -> (f32, f32) {
    if v.is_empty() { return (0.0, 0.0); }
    let n = v.len() as f32;
    let mean = v.iter().copied().sum::<f32>() / n;
    let var = v.iter().map(|x| (x-mean)*(x-mean)).sum::<f32>()/n;
    (mean, var.max(0.0).sqrt())
}
