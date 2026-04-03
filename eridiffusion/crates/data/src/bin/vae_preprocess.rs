// eridiffusion/crates/data/src/bin/vae_preprocess.rs
// GPU-only VAE preprocessing CLI: CPU decode/resize → GPU NHWC BF16 → VAE encode → sharded BF16 latents + manifest.jsonl

use anyhow::{bail, Context, Result};
use clap::Parser;
use eridiffusion_common_vae::{VaeKind, VaeSpec};
use eridiffusion_data::vae_preprocessor::load_vae_spec_from_model_paths;
use eridiffusion_common_weights as cw; // for write_safetensors
use eridiffusion_data::vae_preprocessor::{PreprocessConfig, VaePreprocessor, SampleInfo};
use flame_core::Tensor;
use serde::Serialize;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name="vae_preprocess", about="GPU-only VAE preprocessor: images → BF16 latents (.safetensors) + manifest.jsonl")]
struct Args {
    /// Folder of input images (recursively scans for jpg/jpeg/png/webp)
    #[arg(long)]
    input: PathBuf,
    /// Output directory (created if missing)
    #[arg(long)]
    out: PathBuf,
    /// CUDA device index (default: 0)
    #[arg(long, default_value_t = 0)]
    cuda: usize,

    // ---- VAE spec ----
    /// VAE kind: sdxl | sd35 | flux (overrides YAML if provided)
    #[arg(long, value_parser = ["sdxl","sd35","flux"], required=false)]
    vae_kind: Option<String>,
    /// Path to VAE safetensors (overrides YAML if provided)
    #[arg(long)]
    vae_path: Option<PathBuf>,
    /// VAE latent divisor (e.g., 8 for SDXL) (overrides YAML if provided)
    #[arg(long)]
    latent_div: Option<usize>,
    /// VAE latent channels (e.g., 4) (overrides YAML if provided)
    #[arg(long)]
    latent_channels: Option<usize>,
    /// VAE latent scale (e.g., 0.18215 for SDXL) (overrides YAML if provided)
    #[arg(long)]
    latent_scale: Option<f32>,

    // ---- YAML model paths (optional) ----
    /// Path to modelPaths YAML to auto-load VAE spec
    #[arg(long)]
    model_paths: Option<PathBuf>,
    /// Model id inside YAML (e.g., sdxl_base_1.0)
    #[arg(long)]
    model_id: Option<String>,

    // ---- preprocess ----
    /// Target side length (square). Default 1024.
    #[arg(long, default_value_t = 1024)]
    size: u32,
    /// Center-crop to square before resize
    #[arg(long, default_value_t = true)]
    center_crop: bool,
    /// Normalize RGB to [-1,1] (true) or keep [0,1] (false)
    #[arg(long, default_value_t = true)]
    normalize_to_neg1_1: bool,

    // ---- batching/sharding ----
    /// GPU micro-batch size for encode
    #[arg(long, default_value_t = 4)]
    microbatch: usize,
    /// Number of samples per latents shard
    #[arg(long, default_value_t = 512)]
    shard_size: usize,
}

#[derive(Serialize)]
struct ManifestRow<'a> {
    id: String,
    path: &'a str,
    orig_hw: (u32, u32),
    crop_xy: (u32, u32),
    out_hw: (u32, u32),
    latents: ItemRef<'a>, // shard/key/shape/dtype
}

#[derive(Serialize)]
struct ItemRef<'a> {
    shard: &'a str,
    key: String,
    shape: (usize, usize, usize), // [lh,lw,lc]
    dtype: &'a str,               // "bf16"
}

fn main() -> Result<()> {
    let args = Args::parse();
    create_dir_all(&args.out).context("creating output dir")?;

    // 1) Collect images
    let mut paths = vec![];
    for entry in walkdir::WalkDir::new(&args.input) {
        let entry = entry?;
        if entry.file_type().is_file() {
            let p = entry.path();
            if has_img_ext(p) {
                paths.push(p.to_path_buf());
            }
        }
    }
    if paths.is_empty() {
        bail!("no images found under {:?}", args.input);
    }
    paths.sort();

    // 2) Build VAE spec + preprocessor
    // Prefer YAML if both --model-paths and --model-id are provided.
    let mut vae: VaeSpec = if let (Some(mp), Some(mid)) = (&args.model_paths, &args.model_id) {
        load_vae_spec_from_model_paths(mp, mid)?
    } else {
        // Fall back to CLI flags; require at least path and kind.
        let vae_path = args.vae_path.clone().ok_or_else(|| anyhow::anyhow!("--vae-path required (or provide --model-paths and --model-id)"))?;
        let vae_kind = args.vae_kind.clone().ok_or_else(|| anyhow::anyhow!("--vae-kind required (or provide --model-paths and --model-id)"))?;
        let kind = match vae_kind.as_str() {
            "sdxl" => VaeKind::Sdxl,
            "sd35" => VaeKind::Sd35,
            "flux" => VaeKind::Flux,
            _ => bail!("unsupported vae_kind {}", vae_kind),
        };
        VaeSpec {
            kind,
            path: vae_path.to_string_lossy().to_string(),
            latent_div: args.latent_div.unwrap_or(8),
            latent_channels: args.latent_channels.unwrap_or(4),
            latent_scale: args.latent_scale.unwrap_or(0.18215),
        }
    };

    // Apply overrides if any CLI flags were provided alongside YAML
    if let Some(k) = &args.vae_kind {
        vae.kind = match k.as_str() { "sdxl"=>VaeKind::Sdxl, "sd35"=>VaeKind::Sd35, "flux"=>VaeKind::Flux, _=> bail!("unsupported vae_kind {}", k) };
    }
    if let Some(p) = &args.vae_path { vae.path = p.to_string_lossy().to_string(); }
    if let Some(v) = args.latent_div { vae.latent_div = v; }
    if let Some(v) = args.latent_channels { vae.latent_channels = v; }
    if let Some(v) = args.latent_scale { vae.latent_scale = v; }
    let pre = VaePreprocessor::new(args.cuda, vae.clone())?;

    let cfg = PreprocessConfig {
        target_size: (args.size, args.size),
        center_crop: args.center_crop,
        normalize_to_neg1_1: args.normalize_to_neg1_1,
        microbatch: args.microbatch.max(1),
    };

    // 3) Prepare sharding & manifest
    let mut shard_idx: usize = 0;
    let mut shard_entries: Vec<(String, Tensor)> = Vec::with_capacity(args.shard_size);
    let mut rows_pending: Vec<(String, SampleInfo)> = Vec::with_capacity(args.shard_size);
    let mut manifest = File::create(args.out.join("manifest.jsonl")).context("create manifest")?;

    // 4) Encode in micro-batches; shard writes
    let mut sample_counter: usize = 0;
    for chunk in paths.chunks(cfg.microbatch) {
        let batch = pre.encode_paths(chunk, &cfg)?;
        let b = batch.info.len();

        for i in 0..b {
            let id = format!("sample_{:06}", sample_counter);
            sample_counter += 1;

            // Slice [B,lh,lw,lc] → [lh,lw,lc] for the i-th sample
            let lat_i = batch
                .latents
                .narrow(0, i as usize, 1)?
                .squeeze(Some(0))?; // keep NHWC sans batch

            // Accumulate for shard write
            shard_entries.push((id.clone(), lat_i));
            rows_pending.push((id, batch.info[i].clone()));

            if shard_entries.len() >= args.shard_size {
                shard_idx += 1;
                let shard_name = format!("latents_{:05}.safetensors", shard_idx);
                let shard_path = args.out.join(&shard_name);
                cw::write_safetensors(&shard_path, &shard_entries)
                    .with_context(|| format!("write {}", shard_path.display()))?;

                // Emit manifest rows (shape/dtype from the first entry; all share LC/shape)
                for (id, info) in rows_pending.drain(..) {
                    let (lh, lw, lc) = latent_shape(&shard_entries[0].1)?;
                    let id_owned = id;
                    let id_key = id_owned.clone();
                    let path_str = info.path.to_string_lossy().to_string();
                    let row = ManifestRow {
                        id: id_owned,
                        path: &path_str,
                        orig_hw: info.orig_hw,
                        crop_xy: info.crop_xy,
                        out_hw: info.out_hw,
                        latents: ItemRef {
                            shard: &shard_name,
                            key: id_key,
                            shape: (lh, lw, lc),
                            dtype: "bf16",
                        },
                    };
                    serde_json::to_writer(&mut manifest, &row)?;
                    manifest.write_all(b"\n")?;
                }
                shard_entries.clear();
            }
        }
    }

    // 5) Flush final shard if needed
    if !shard_entries.is_empty() {
        shard_idx += 1;
        let shard_name = format!("latents_{:05}.safetensors", shard_idx);
        let shard_path = args.out.join(&shard_name);
        cw::write_safetensors(&shard_path, &shard_entries)
            .with_context(|| format!("write {}", shard_path.display()))?;
        for (id, info) in rows_pending.drain(..) {
            let (lh, lw, lc) = latent_shape(&shard_entries[0].1)?;
            let id_owned = id;
            let id_key = id_owned.clone();
            let path_str = info.path.to_string_lossy().to_string();
            let row = ManifestRow {
                id: id_owned,
                path: &path_str,
                orig_hw: info.orig_hw,
                crop_xy: info.crop_xy,
                out_hw: info.out_hw,
                latents: ItemRef {
                    shard: &shard_name,
                    key: id_key,
                    shape: (lh, lw, lc),
                    dtype: "bf16",
                },
            };
            serde_json::to_writer(&mut manifest, &row)?;
            manifest.write_all(b"\n")?;
        }
    }

    eprintln!(
        "✅ wrote {} manifest rows, {} shards in {}",
        sample_counter, shard_idx, args.out.display()
    );
    Ok(())
}

fn has_img_ext(p: &Path) -> bool {
    match p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
        Some(ext) if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp") => true,
        _ => false,
    }
}

fn latent_shape(t: &Tensor) -> Result<(usize, usize, usize)> {
    let d = t.shape().dims().to_vec();
    if d.len() != 3 {
        bail!("latent tensor must be [lh,lw,lc], got {:?}", d);
    }
    Ok((d[0] as usize, d[1] as usize, d[2] as usize))
}
