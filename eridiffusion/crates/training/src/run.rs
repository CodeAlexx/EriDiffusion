use std::{fs::File, path::PathBuf, time::Instant};

#[cfg(feature = "sdxl")]
use std::sync::Arc;

#[cfg(feature = "sdxl")]
use anyhow::{anyhow, ensure};
use eridiffusion_core::{Device as EriDevice, FluxVariant};
use eridiffusion_models::devtensor::{shape4, zeros_on};
use flame_core::{CudaDevice, DType};

#[cfg(feature = "sdxl")]
use eridiffusion_common_weights::strict_loader::StrictMmapLoader;
#[cfg(feature = "sdxl")]
use flame_core::Device as FlameDevice;
#[cfg(feature = "sdxl")]
use flame_core::Tensor;

use crate::{
    data::synthetic_batch,
    flux::data_loader::{FluxDataLoader, FluxPrecomputedCfg},
    flux_trainer::create_flux_trainer,
    model_registry::{self as reg, ModelKind, TrainCfg},
};

#[cfg(feature = "sdxl")]
use crate::sdxl::{
    data_loader::{SdxlBatch, SdxlDataLoader, SdxlPrecomputedCfg},
    registry::SdxlLayerRegistry,
    weights::SdxlWeightProvider,
    RuntimeMode,
};

/// Generic training loop entry used by tests and the trainer binary.
pub fn run_training(cfg: &TrainCfg) -> anyhow::Result<()> {
    match cfg.model_kind {
        ModelKind::Flux => run_flux(cfg),
        ModelKind::SDXL => run_sdxl(cfg),
        ModelKind::SD3 => run_sd35(cfg),
        _ => run_synthetic(cfg),
    }
}

fn run_synthetic(cfg: &TrainCfg) -> anyhow::Result<()> {
    let bundle =
        reg::build_model(cfg).map_err(|e| anyhow::anyhow!(format!("build_model failed: {}", e)))?;
    println!(
        "trainer: model_kind={:?} name={} steps={} bs={} accum={} lr={}",
        cfg.model_kind, bundle.name, cfg.steps, cfg.batch_size, cfg.grad_accum_steps, cfg.lr
    );
    for step in 1..=cfg.steps {
        let b = cfg.batch_size.max(1);
        let batch = synthetic_batch(b, 32, 32, 4, None)?;
        let loss = batch.latents.sum()?.item()?;
        println!("step {}/{} | loss {:.6}", step, cfg.steps, loss);
    }
    Ok(())
}

fn run_flux(cfg: &TrainCfg) -> anyhow::Result<()> {
    // Optional fast path: validate streaming registry and weights only
    if std::env::var("REGISTRY_ONLY").ok().as_deref() == Some("1") {
        println!("INFO: REGISTRY_ONLY enabled → skipping encoders/vae; probing Flux registry only");
        let model_path = cfg.flux_model_path.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Flux config missing model path. Provide either `flux_model_path:` or generic `weights:`"
            )
        })?;
        println!("INFO: Flux weights = {}", model_path);
        let c = crate::flux::train::Config {
            shard_path: model_path.clone(),
            device_ordinal: 0,
            probe_blocks: 3,
        };
        return crate::flux::train::train_loop(&c).map_err(|e| anyhow::anyhow!(e.to_string()));
    }

    // Validate required paths
    let model_path = cfg.flux_model_path.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "Flux config missing model path. Provide either `flux_model_path:` or generic `weights:`"
        )
    })?;
    println!("INFO: Flux weights = {}", model_path);
    let vae_path =
        cfg.flux_vae_path.as_ref().ok_or_else(|| anyhow::anyhow!("flux_vae_path missing"))?;
    if vae_path.to_lowercase().contains("sdxl") {
        println!("WARNING: flux_vae_path appears to be an SDXL VAE: {}", vae_path);
    }
    println!("INFO: Flux AE = {} (assumed latents=16, scale≈0.13025)", vae_path);
    let t5_path =
        cfg.flux_t5_path.as_ref().ok_or_else(|| anyhow::anyhow!("flux_t5_path missing"))?;
    let clip_path =
        cfg.flux_clip_path.as_ref().ok_or_else(|| anyhow::anyhow!("flux_clip_path missing"))?;
    let t5_tok = cfg
        .flux_t5_tokenizer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("flux_t5_tokenizer missing"))?;
    let clip_tok = cfg
        .flux_clip_tokenizer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("flux_clip_tokenizer missing"))?;
    let variant = match cfg.flux_variant.as_deref() {
        Some("schnell") => FluxVariant::Schnell,
        Some("dev") => FluxVariant::Dev,
        _ => FluxVariant::Base,
    };
    let device = EriDevice::Cuda(0);

    #[cfg(feature = "nvml")]
    println!("telemetry: NVML=ON");
    #[cfg(not(feature = "nvml"))]
    println!("telemetry: NVML=OFF (using fallback)");

    let mut train_cfg = crate::FluxTrainingConfig::default();
    train_cfg.seed = cfg.seed;
    if let Some(minv) = cfg.sigma_min {
        train_cfg.sigma_min = minv;
    }
    if let Some(maxv) = cfg.sigma_max {
        train_cfg.sigma_max = maxv;
    }
    if let Some(r) = cfg.lora_rank {
        train_cfg.lora_rank = r;
    }
    if let Some(a) = cfg.lora_alpha {
        train_cfg.lora_alpha = a;
    }
    if let Some(z) = cfg.lora_zero_init {
        train_cfg.lora_zero_init = z;
    }

    let mut current_seed = train_cfg.seed;
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let mut trainer = rt.block_on(create_flux_trainer(
        std::path::Path::new(model_path),
        std::path::Path::new(vae_path),
        std::path::Path::new(t5_path),
        std::path::Path::new(clip_path),
        std::path::Path::new(t5_tok),
        std::path::Path::new(clip_tok),
        variant,
        train_cfg,
        device.clone(),
    ))?;

    let mut flux_loader = if let Some(manifest) = cfg.flux_manifest.as_ref() {
        println!("INFO: Flux manifest = {}", manifest);
        let pre_cfg = FluxPrecomputedCfg {
            manifest_path: PathBuf::from(manifest),
            root_dir: cfg.flux_manifest_root.as_ref().map(PathBuf::from),
            batch_size: cfg.batch_size.max(1),
            shuffle: cfg.flux_manifest_shuffle.unwrap_or(true),
            max_epochs: cfg.flux_manifest_repeat,
            enforce_bf16: true,
            validate_on_load: cfg.flux_manifest_validate.unwrap_or(false),
            cache_index: cfg.flux_manifest_index.as_ref().map(PathBuf::from),
            device: device.clone(),
        };
        let loader = FluxDataLoader::from_precomputed(pre_cfg)?;
        if let Some(stats) = loader.precomputed_stats() {
            let to_gb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            println!(
                "INFO: Flux cache entries={} | latents={:.2} GB | text={:.2} GB | clip={:.2} GB | manifest_hash={}",
                stats.len,
                to_gb(stats.total_latent_bytes),
                to_gb(stats.total_t5_bytes),
                to_gb(stats.total_clip_bytes),
                stats.manifest_hash.as_deref().unwrap_or("<unknown>"),
            );
        }
        if let Some(hit) = loader.cache_index_hit() {
            println!("INFO: Flux manifest index cache_hit={hit}");
        }
        Some(loader)
    } else {
        None
    };

    let b = cfg.batch_size.max(1);
    let mut csv: Option<csv::Writer<File>> = std::env::var("TRAIN_LOG_CSV")
        .ok()
        .and_then(|p| File::create(&p).ok())
        .map(csv::Writer::from_writer);
    if let Some(w) = csv.as_mut() {
        w.write_record(["step", "loss", "grad_norm", "sec_per_it", "alloc_mb", "ckpt_path"]).ok();
        let _ = w.flush();
    }

    let out_dir = std::env::var("TRAIN_LOG_CSV")
        .ok()
        .and_then(|p| PathBuf::from(p).parent().map(|pp| pp.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("runs/flux_real"));
    let ckpt_dir = out_dir.join("checkpoints");
    std::fs::create_dir_all(&ckpt_dir).ok();
    let ckpt_every_env = std::env::var("CKPT_EVERY").ok().and_then(|s| s.parse::<usize>().ok());
    let ckpt_every = cfg.ckpt_every.or(ckpt_every_env).unwrap_or(25);

    let mut last = Instant::now();
    let ckpt_root = format!("{}/checkpoints", out_dir.display());
    let mut start_step: usize = 1;
    if let Ok(Some((loaded_step, loaded_seed))) = trainer.load_latest_checkpoint(&ckpt_root) {
        println!("resume: loaded step={}, seed={}", loaded_step, loaded_seed);
        start_step = (loaded_step as usize).saturating_add(1);
        current_seed = loaded_seed;
    }

    for step in start_step..=cfg.steps {
        let cuda_dev = match device {
            EriDevice::Cuda(ix) => CudaDevice::new(ix)?,
            _ => anyhow::bail!("run_flux requires CUDA device"),
        };

        cuda_dev.synchronize()?;

        let (loss, batch_stats) = if let Some(loader) = flux_loader.as_mut() {
            let batch = loop {
                match loader.next()? {
                    Some(b) => break b,
                    None => continue,
                }
            };
            let stats = batch.telemetry.clone();
            let loss = trainer.train_step_precomputed(&batch)?;
            (loss, stats)
        } else {
            let images = zeros_on(shape4(b as i64, 64, 64, 3), &device, DType::F32)
                .map_err(|e| anyhow::anyhow!("Failed to create image batch: {e}"))?;
            let captions: Vec<String> = (0..b).map(|_| "a photo".to_string()).collect();
            let negs: Vec<String> = (0..b).map(|_| "".to_string()).collect();
            let loss = rt.block_on(trainer.train_step(&images, &captions, &negs))?;
            (loss, None)
        };

        cuda_dev.synchronize()?;
        let dt = last.elapsed().as_secs_f32().max(1e-6);
        last = Instant::now();
        let gn = trainer.grad_norm();

        let nvml_mb = crate::telemetry_mem::gpu_alloc_mb(0);
        let bytes = trainer.lora_param_bytes();
        let lora_mb = (bytes as f32) / (1024.0 * 1024.0);
        let alloc_mb = nvml_mb.max(lora_mb);
        #[cfg(feature = "nvml")]
        if step == 1 {
            println!("NVML used ≈ {:.3} MB", nvml_mb);
        }
        if step % 10 == 0 {
            let (val, unit) = crate::telemetry_mem::format_bytes_auto(bytes);
            let unit_str = match unit {
                crate::telemetry_mem::MemUnit::KB => "KB",
                crate::telemetry_mem::MemUnit::MB => "MB",
                crate::telemetry_mem::MemUnit::GB => "GB",
            };
            println!("diag: lora_bytes ≈ {:.3} {}", val, unit_str);
        }

        let mut ckpt_path = String::new();
        if step % ckpt_every == 0 || step == cfg.steps {
            if let Ok(path) = trainer.save_checkpoint(&ckpt_root, step as u64, current_seed) {
                ckpt_path = path;
            }
        }

        let batches_per_sec = 1.0 / dt;
        let mut samples_per_sec = b as f32 / dt;
        let mut h2d_mb: Option<f32> = None;
        let mut h2d_mb_s = 0.0f32;
        if let Some(stats) = batch_stats.as_ref() {
            let total_mb = stats.total_mb();
            h2d_mb = Some(total_mb);
            h2d_mb_s = total_mb / dt;
            samples_per_sec = stats.samples as f32 / dt;
        }

        if let Some(total_mb) = h2d_mb {
            println!(
                "step {}/{} | loss {:.6} | grad_norm {:.6} | sec/it {:.3} | alloc_MB {:.1} | h2d_MB {:.2} | h2d_MBps {:.2} | batches/s {:.2} | samples/s {:.2}",
                step,
                cfg.steps,
                loss,
                gn,
                dt,
                alloc_mb,
                total_mb,
                h2d_mb_s,
                batches_per_sec,
                samples_per_sec
            );
        } else {
            println!(
                "step {}/{} | loss {:.6} | grad_norm {:.6} | sec/it {:.3} | alloc_MB {:.1}",
                step, cfg.steps, loss, gn, dt, alloc_mb
            );
        }

        if let Some(w) = csv.as_mut() {
            let _ = w.write_record(&[
                step.to_string(),
                format!("{:.6}", loss),
                format!("{:.6}", gn),
                format!("{:.6}", dt),
                format!("{:.3}", alloc_mb),
                ckpt_path,
            ]);
            let _ = w.flush();
        }
    }
    Ok(())
}

#[cfg(feature = "sdxl")]
fn run_sdxl(cfg: &TrainCfg) -> anyhow::Result<()> {
    if std::env::var("REGISTRY_ONLY").ok().as_deref() == Some("1") {
        let unet_path =
            cfg.sdxl_unet_path.as_ref().ok_or_else(|| anyhow::anyhow!("sdxl_unet_path missing"))?;
        let c = crate::sdxl::train::Config {
            shard_path: unet_path.clone(),
            device_ordinal: 0,
            probe_blocks: 3,
        };
        return crate::sdxl::train::train_loop(&c).map_err(|e| anyhow::anyhow!(e.to_string()));
    }

    if let Some(manifest) = cfg.sdxl_manifest.as_ref() {
        println!("INFO: SDXL manifest = {}", manifest);
        return run_sdxl_manifest(cfg, manifest);
    }

    println!("INFO: no SDXL manifest specified → falling back to registry probe");
    let unet_path =
        cfg.sdxl_unet_path.as_ref().ok_or_else(|| anyhow::anyhow!("sdxl_unet_path missing"))?;
    let c = crate::sdxl::train::Config {
        shard_path: unet_path.clone(),
        device_ordinal: 0,
        probe_blocks: 3,
    };
    crate::sdxl::train::train_loop(&c).map_err(|e| anyhow::anyhow!(e.to_string()))
}

#[cfg(feature = "sdxl")]
fn run_sdxl_manifest(cfg: &TrainCfg, manifest: &str) -> anyhow::Result<()> {
    let device = EriDevice::Cuda(0);
    let flame_device = FlameDevice::cuda(0)?;
    let weight_path = cfg
        .sdxl_unet_path
        .as_ref()
        .or_else(|| cfg.weights.as_ref())
        .ok_or_else(|| anyhow::anyhow!("sdxl_unet_path (or weights) missing"))?;
    let mmap = Arc::new(StrictMmapLoader::open(PathBuf::from(weight_path))?);
    let provider = Arc::new(SdxlWeightProvider::new(mmap, flame_device));
    let registry = SdxlLayerRegistry::build(provider.clone(), RuntimeMode::Resident)?;

    let pre_cfg = SdxlPrecomputedCfg {
        manifest_path: PathBuf::from(manifest),
        root_dir: cfg.sdxl_manifest_root.as_ref().map(PathBuf::from),
        batch_size: cfg.batch_size.max(1),
        shuffle: cfg.sdxl_manifest_shuffle.unwrap_or(true),
        max_epochs: cfg.sdxl_manifest_repeat,
        enforce_bf16: true,
        validate_on_load: cfg.sdxl_manifest_validate.unwrap_or(false),
        cache_index: cfg.sdxl_manifest_index.as_ref().map(PathBuf::from),
        device: device.clone(),
    };

    let mut loader = SdxlDataLoader::from_precomputed(pre_cfg)?;
    if let Some(stats) = loader.precomputed_stats() {
        let to_gb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "INFO: SDXL cache entries={} | latents={:.2} GB | clip_l={:.2} GB | clip_g={:.2} GB | pooled={:.2} GB | manifest_hash={}",
            stats.len,
            to_gb(stats.total_latent_bytes),
            to_gb(stats.total_clip_l_bytes),
            to_gb(stats.total_clip_g_bytes),
            to_gb(stats.total_pooled_bytes),
            stats.manifest_hash.as_deref().unwrap_or("<unknown>"),
        );
    }
    if let Some(hit) = loader.cache_index_hit() {
        println!("INFO: SDXL manifest index cache_hit={hit}");
    }

    let steps = cfg.steps.max(1);
    let mut consumed = 0usize;
    for step in 1..=steps {
        let Some(batch) = loader.next()? else {
            println!(
                "INFO: SDXL manifest exhausted after {} batches; stopping at step {}",
                consumed,
                step - 1
            );
            break;
        };
        consumed += 1;

        let loss_val = run_sdxl_forward(&registry, &batch)?;
        if let Some(t) = &batch.telemetry {
            println!(
                "step {}/{} | loss {:.6} | h2d_MB {:.2} | samples {}",
                step,
                steps,
                loss_val,
                t.total_mb(),
                t.samples
            );
        } else {
            println!(
                "step {}/{} | loss {:.6} | latents {:?}",
                step,
                steps,
                loss_val,
                batch.latents.shape().dims()
            );
        }

        if step >= steps {
            break;
        }
    }

    println!("INFO: SDXL manifest loop consumed {} batches", consumed);
    Ok(())
}

#[cfg(feature = "sdxl")]
fn run_sdxl_forward(registry: &SdxlLayerRegistry, batch: &SdxlBatch) -> anyhow::Result<f32> {
    let latents = to_nhwc(&batch.latents)?;
    let cond = registry.make_conditioning(&batch.pooled, &batch.timesteps, &batch.time_ids)?;
    let ctx = Tensor::cat(&[&batch.clip_l, &batch.clip_g], 2)?; // [B,T,2048]
    let sample_bf16 = latents.to_dtype(DType::BF16)?;
    let ctx_bf16 = ctx.to_dtype(DType::BF16)?;
    let sample = registry.forward_blocks(sample_bf16, &ctx_bf16, &cond, 0)?;

    let loss = sample.powf(2.0)?.mean()?;
    let loss_host = loss.to_vec1::<f32>()?;
    Ok(*loss_host.first().unwrap_or(&0.0))
}

#[cfg(feature = "sdxl")]
fn to_nhwc(latents: &Tensor) -> anyhow::Result<Tensor> {
    let dims = latents.shape().dims().to_vec();
    ensure!(dims.len() == 4, "SDXL latents must be rank-4, got {:?}", dims);
    if dims[3] == 4 {
        return Ok(latents.clone_result()?);
    }
    ensure!(dims[1] == 4, "SDXL latents must have channel dim 4 (NCHW/NHWC)");
    latents.permute(&[0, 2, 3, 1])
}

#[cfg(feature = "sdxl")]
fn sdxl_batch_to_prepared(batch: &SdxlBatch) -> anyhow::Result<PreparedBatch> {
    let dims = batch.latents.shape().dims().to_vec();
    ensure!(dims.len() == 4, "SDXL latents must be rank-4, got {:?}", dims);
    let b = dims[0] as usize;
    ensure!(dims[1] == 4, "SDXL latents must have 4 channels, got {}", dims[1]);
    let h_lat = dims[2] as u32;
    let w_lat = dims[3] as u32;
    let upscale = 8u32;
    let height_px = h_lat.saturating_mul(upscale);
    let width_px = w_lat.saturating_mul(upscale);

    let captions: Vec<String> = batch.records.iter().map(|r| r.caption.clone()).collect();

    let per_sample_metadata: Vec<JsonValue> = batch
        .records
        .iter()
        .map(|r| {
            JsonValue::Object(
                r.metadata.clone().into_iter().collect::<JsonMap<String, JsonValue>>(),
            )
        })
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("original_sizes".to_string(), json!(vec![(width_px, height_px); b]));
    metadata.insert("crop_coords".to_string(), json!(vec![(0u32, 0u32); b]));
    metadata.insert("target_sizes".to_string(), json!(vec![(width_px, height_px); b]));
    let time_ids =
        batch.time_ids.to_vec2::<f32>().context("failed to copy SDXL time_ids to host")?;
    metadata.insert("time_ids".to_string(), json!(time_ids));
    let timesteps =
        batch.timesteps.to_vec1::<f32>().context("failed to copy SDXL timesteps to host")?;
    metadata.insert("timesteps".to_string(), json!(timesteps));
    metadata.insert("sample_metadata".to_string(), JsonValue::Array(per_sample_metadata));

    Ok(PreparedBatch {
        images: batch.latents.clone(),
        latents: Some(batch.latents.clone()),
        captions,
        metadata,
    })
}

#[cfg(not(feature = "sdxl"))]
fn run_sdxl(_cfg: &TrainCfg) -> anyhow::Result<()> {
    anyhow::bail!("SDXL not enabled in this build")
}

#[cfg(feature = "sd35")]
fn run_sd35(cfg: &TrainCfg) -> anyhow::Result<()> {
    let model =
        cfg.sd35_model_path.as_ref().ok_or_else(|| anyhow::anyhow!("sd35_model_path missing"))?;
    let c = crate::sd35::train::Config {
        shard_path: model.clone(),
        device_ordinal: 0,
        probe_blocks: 3,
    };
    crate::sd35::train::train_loop(&c).map_err(|e| anyhow::anyhow!(e.to_string()))
}

#[cfg(not(feature = "sd35"))]
fn run_sd35(_cfg: &TrainCfg) -> anyhow::Result<()> {
    anyhow::bail!("SD35 not enabled in this build")
}
