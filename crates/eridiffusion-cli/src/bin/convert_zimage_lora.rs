//! Convert an external Z-Image LoRA (kohya / musubi / comfy / ai-toolkit
//! format) into the EDv2 trainer's bundle format so that
//! `sample_zimage --lora <path>` (or any consumer that goes through
//! `ZImageLoraBundle::load`) can render it.
//!
//! Why this exists: EDv2's `ZImageLoraBundle::load` accepts either the
//! ai-toolkit format (`diffusion_model.layers.{i}.<suffix>.lora_{A,B}.weight`)
//! or the legacy trainer format (`layers.{i}.<suffix>.lora_{A,B}` with
//! `attention.out` aliasing `attention.to_out.0`). External LoRAs commonly
//! ship with:
//!   - kohya naming `lora_unet_layers_<i>_<module>.lora_down/up.weight`
//!   - fused-QKV LoRAs (single LoRA on `attention.qkv` with up shape
//!     `[3*dim, rank]`) that need to be split into 3 trainer-format LoRAs
//!   - per-module `<prefix>.alpha` scalar (often != rank)
//!   - mixed ranks (e.g. rank=96 for QKV, rank=32 for everything else)
//!
//! This binary normalises all of those into the trainer's legacy format.
//! Padding smaller-rank LoRAs to the target rank with zero rows/cols
//! preserves their math exactly (the zero rows/cols contribute nothing to
//! the matmul). The source `alpha/rank` scale is pre-baked into `lora_B`
//! so the trainer's default `scale=1.0` reproduces the source's effective
//! contribution per module.
//!
//! This is the EDv2 port of `flame-diffusion-archive/zimage-trainer/
//! src/bin/convert_external_lora.rs` — identity-verified there against the
//! eri2 musubi reference comfy LoRA. Logic is unchanged; only module paths
//! were adapted to EDv2 (`flame_core::CudaDevice` instead of
//! `cudarc::driver::CudaDevice`, `flame_core::Error` is now
//! `flame_core::FlameError`).
//!
//! Usage:
//!   convert_zimage_lora \
//!     --input /home/alex/.serenity/models/loras/eri2_zimage_lora_comfy.safetensors \
//!     --output /tmp/eri2_zimage_lora_trainer_format.safetensors \
//!     --target-rank 96

use clap::Parser;
use flame_core::{CudaDevice, DType, FlameError, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

const NUM_LAYERS: usize = 30;
const DIM: usize = 3840;
const MLP_HIDDEN: usize = 10240;

#[derive(Parser, Debug)]
#[command(about = "Convert kohya/musubi/comfy/ai-toolkit Z-Image LoRA → EDv2 trainer format")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Trainer bundle uses ONE rank for all 7 LoRAs per layer. We pad
    /// smaller-rank source LoRAs with zero rows/cols to this rank.
    /// Default 96 covers the eri2 reference (QKV rank 96, others rank 32).
    #[arg(long, default_value_t = 96)]
    target_rank: usize,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let device = flame_core::global_cuda_device();

    let src = flame_core::serialization::load_file(&args.input, &device)?;
    eprintln!(
        "[convert] loaded {} tensors from {}",
        src.len(),
        args.input.display()
    );
    let format = detect_format(&src);
    eprintln!("[convert] detected format: {:?}", format);

    let mut out: HashMap<String, Tensor> = HashMap::new();

    // Per-format key naming. We only handle kohya right now (the comfy
    // reference). Add formats as needed.
    let (suffix_a, suffix_b) = match format {
        ExternalFormat::KohyaSdxl | ExternalFormat::DiffusersKohya => {
            (".lora_down.weight", ".lora_up.weight")
        }
        ExternalFormat::AiToolkit => (".lora_A.weight", ".lora_B.weight"),
        ExternalFormat::TrainerSplit => {
            eprintln!("[convert] input is already in trainer split-QKV format — copying through");
            for (k, v) in &src {
                out.insert(k.clone(), v.to_dtype(DType::F32)?);
            }
            return write_out(&out, &args.output);
        }
    };

    let mut total_modules = 0usize;
    let mut split_qkv_count = 0usize;
    let mut padded_count = 0usize;

    for layer in 0..NUM_LAYERS {
        // DiffusersKohya already has split Q/K/V — handle it via the per-projection
        // path below (skip the fused split here).
        if matches!(format, ExternalFormat::DiffusersKohya) {
            for target in ["to_q", "to_k", "to_v"] {
                convert_split_qkv_dk(
                    &src, &mut out, layer, target,
                    args.target_rank, &device,
                    &mut total_modules, &mut padded_count,
                )?;
            }
            // attention.out (diffusers naming: to_out.0)
            convert_dk_module(
                &src, &mut out, layer,
                "attention.to_out.0", "attention.out",
                args.target_rank, DIM, DIM,
                &device, &mut total_modules, &mut padded_count,
            )?;
            // feed_forward.w1/w2/w3
            convert_dk_module(
                &src, &mut out, layer,
                "feed_forward.w1", "feed_forward.w1",
                args.target_rank, DIM, MLP_HIDDEN,
                &device, &mut total_modules, &mut padded_count,
            )?;
            convert_dk_module(
                &src, &mut out, layer,
                "feed_forward.w2", "feed_forward.w2",
                args.target_rank, MLP_HIDDEN, DIM,
                &device, &mut total_modules, &mut padded_count,
            )?;
            convert_dk_module(
                &src, &mut out, layer,
                "feed_forward.w3", "feed_forward.w3",
                args.target_rank, DIM, MLP_HIDDEN,
                &device, &mut total_modules, &mut padded_count,
            )?;
            continue;
        }

        // attention.qkv (fused) — split into 3 trainer LoRAs.
        let qkv_prefix = match format {
            ExternalFormat::KohyaSdxl => format!("lora_unet_layers_{layer}_attention_qkv"),
            ExternalFormat::AiToolkit => format!("layers.{layer}.attention.qkv"),
            ExternalFormat::DiffusersKohya | ExternalFormat::TrainerSplit => unreachable!(),
        };
        let down_key = format!("{qkv_prefix}{suffix_a}");
        let up_key = format!("{qkv_prefix}{suffix_b}");
        match (src.get(&down_key), src.get(&up_key)) {
            (Some(down), Some(up)) => {
                let down_dims = down.shape().dims().to_vec();
                let up_dims = up.shape().dims().to_vec();
                let rank = down_dims[0];
                if up_dims[0] != 3 * DIM || up_dims[1] != rank || down_dims[1] != DIM {
                    eprintln!(
                        "[convert] L{layer} attention.qkv unexpected shape: down={:?} up={:?}",
                        down_dims, up_dims
                    );
                } else {
                    let alpha = read_alpha(&src, &qkv_prefix, rank)?;
                    let scale_src = alpha / rank as f32;
                    // Pad to target rank. Zero pad on the rank axis = zero contribution
                    // from added components → math preserved.
                    let padded_down = pad_rows(down, args.target_rank, &device)?;
                    let padded_up_full = pad_cols(up, args.target_rank, &device)?;
                    // Pre-bake the source scale into lora_B. The trainer's
                    // bundle is init'd with alpha=rank → trainer-internal
                    // scale=1.0. Multiplying lora_B by scale_src here means
                    // trainer's `scale_src * (B @ A)` produces the correct
                    // effective contribution. Avoids needing to plumb a
                    // separate alpha through the sample binary.
                    let scaled_up_full = padded_up_full
                        .to_dtype(DType::F32)?
                        .mul_scalar(scale_src)?;
                    // Split fused up [3*DIM, target_rank] into 3 [DIM, target_rank].
                    let up_q = scaled_up_full.narrow(0, 0, DIM)?.contiguous()?;
                    let up_k = scaled_up_full.narrow(0, DIM, DIM)?.contiguous()?;
                    let up_v = scaled_up_full.narrow(0, 2 * DIM, DIM)?.contiguous()?;
                    if layer == 0 {
                        eprintln!(
                            "[convert] L0 attention.qkv: src rank={rank} alpha={alpha:.3} scale={scale_src:.5}; \
                             pre-baked scale into lora_B so trainer's default scale=1.0 reproduces it"
                        );
                    }
                    let down_f32 = padded_down.to_dtype(DType::F32)?;
                    for (target, up_split) in
                        [("to_q", up_q), ("to_k", up_k), ("to_v", up_v)]
                    {
                        let prefix = format!("layers.{layer}.attention.{target}");
                        out.insert(format!("{prefix}.lora_A"), down_f32.clone());
                        out.insert(format!("{prefix}.lora_B"), up_split);
                        total_modules += 1;
                    }
                    split_qkv_count += 1;
                    if rank < args.target_rank {
                        padded_count += 1;
                    }
                }
            }
            _ => eprintln!("[convert] L{layer} missing attention.qkv pair"),
        }

        // attention.out — direct map with rank padding.
        convert_full(
            &src,
            &mut out,
            layer,
            "attention_out",
            "attention.out",
            args.target_rank,
            DIM,
            DIM,
            format,
            suffix_a,
            suffix_b,
            &device,
            &mut total_modules,
            &mut padded_count,
        )?;

        // feed_forward.w1/w2/w3.
        convert_full(
            &src, &mut out, layer, "feed_forward_w1", "feed_forward.w1",
            args.target_rank, DIM, MLP_HIDDEN, format, suffix_a, suffix_b,
            &device, &mut total_modules, &mut padded_count,
        )?;
        convert_full(
            &src, &mut out, layer, "feed_forward_w2", "feed_forward.w2",
            args.target_rank, MLP_HIDDEN, DIM, format, suffix_a, suffix_b,
            &device, &mut total_modules, &mut padded_count,
        )?;
        convert_full(
            &src, &mut out, layer, "feed_forward_w3", "feed_forward.w3",
            args.target_rank, DIM, MLP_HIDDEN, format, suffix_a, suffix_b,
            &device, &mut total_modules, &mut padded_count,
        )?;
    }

    eprintln!(
        "[convert] wrote {total_modules} adapter slots ({split_qkv_count} fused-QKV split into 3 each, {padded_count} padded to target rank)"
    );
    write_out(&out, &args.output)
}

fn write_out(out: &HashMap<String, Tensor>, path: &PathBuf) -> anyhow::Result<()> {
    flame_core::serialization::save_tensors(
        out,
        path,
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    eprintln!("[convert] saved → {}", path.display());
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum ExternalFormat {
    /// `lora_unet_layers_<i>_attention_qkv.lora_down.weight` — fused-QKV per
    /// layer, kohya naming. Eri2 musubi reference comfy LoRA.
    KohyaSdxl,
    /// `*.lora_A.weight` / `*.lora_B.weight` — ai-toolkit style.
    AiToolkit,
    /// `diffusion_model.layers.<i>.attention.to_q.lora_down.weight` —
    /// diffusers naming (split Q/K/V, `to_out.0` for out-proj) + kohya
    /// suffixes. Used by HuggingFace nphSi/Z-Image-Lora identity LoRAs.
    DiffusersKohya,
    /// Already in trainer format.
    TrainerSplit,
}

fn detect_format(lora: &HashMap<String, Tensor>) -> ExternalFormat {
    if lora
        .keys()
        .any(|k| k.starts_with("lora_unet_") && k.ends_with(".lora_down.weight"))
    {
        return ExternalFormat::KohyaSdxl;
    }
    if lora.keys().any(|k| {
        k.starts_with("diffusion_model.layers.")
            && (k.ends_with(".lora_down.weight") || k.ends_with(".lora_up.weight"))
    }) {
        return ExternalFormat::DiffusersKohya;
    }
    if lora
        .keys()
        .any(|k| k.contains(".attention.to_q.lora_A") || k.contains(".attention.to_q.lora_B"))
    {
        return ExternalFormat::TrainerSplit;
    }
    ExternalFormat::AiToolkit
}

fn read_alpha(src: &HashMap<String, Tensor>, prefix: &str, fallback_rank: usize) -> Result<f32> {
    let alpha_key = format!("{prefix}.alpha");
    match src.get(&alpha_key) {
        Some(t) => {
            let v = t.to_dtype(DType::F32)?.to_vec()?;
            Ok(v.first().copied().unwrap_or(fallback_rank as f32))
        }
        None => Ok(fallback_rank as f32), // assumption: alpha=rank → scale=1.0
    }
}

#[allow(clippy::too_many_arguments)]
fn convert_full(
    src: &HashMap<String, Tensor>,
    out: &mut HashMap<String, Tensor>,
    layer: usize,
    src_module_underscored: &str,
    dst_suffix_dotted: &str,
    target_rank: usize,
    in_dim: usize,
    out_dim: usize,
    format: ExternalFormat,
    suffix_a: &str,
    suffix_b: &str,
    device: &Arc<CudaDevice>,
    total_modules: &mut usize,
    padded_count: &mut usize,
) -> Result<()> {
    let prefix_src = match format {
        ExternalFormat::KohyaSdxl => format!("lora_unet_layers_{layer}_{src_module_underscored}"),
        ExternalFormat::AiToolkit => {
            format!("layers.{layer}.{}", dst_suffix_dotted)
        }
        // DiffusersKohya is handled in main loop directly via convert_dk_module — never reaches here.
        ExternalFormat::DiffusersKohya | ExternalFormat::TrainerSplit => unreachable!(),
    };
    let down_key = format!("{prefix_src}{suffix_a}");
    let up_key = format!("{prefix_src}{suffix_b}");
    let (down, up) = match (src.get(&down_key), src.get(&up_key)) {
        (Some(d), Some(u)) => (d, u),
        _ => {
            eprintln!("[convert] L{layer} {dst_suffix_dotted}: missing — skipping");
            return Ok(());
        }
    };
    let down_dims = down.shape().dims().to_vec();
    let up_dims = up.shape().dims().to_vec();
    let rank = down_dims[0];
    if down_dims[1] != in_dim || up_dims[0] != out_dim || up_dims[1] != rank {
        eprintln!(
            "[convert] L{layer} {dst_suffix_dotted}: shape mismatch down={:?} up={:?} expected (rank,{in_dim}) ({out_dim},rank)",
            down_dims, up_dims
        );
        return Ok(());
    }
    let alpha = read_alpha(src, &prefix_src, rank)?;
    let scale_src = alpha / rank as f32;
    let padded_down = pad_rows(down, target_rank, device)?;
    let padded_up = pad_cols(up, target_rank, device)?;
    // Pre-bake scale into lora_B so trainer's default scale=1.0 reproduces
    // the source LoRA's effective contribution.
    let scaled_up = padded_up
        .to_dtype(DType::F32)?
        .mul_scalar(scale_src)?;
    let prefix_dst = format!("layers.{layer}.{dst_suffix_dotted}");
    out.insert(
        format!("{prefix_dst}.lora_A"),
        padded_down.to_dtype(DType::F32)?,
    );
    out.insert(format!("{prefix_dst}.lora_B"), scaled_up);
    *total_modules += 1;
    if rank < target_rank {
        *padded_count += 1;
    }
    Ok(())
}

/// DiffusersKohya: convert a split QKV projection (to_q/to_k/to_v) directly.
fn convert_split_qkv_dk(
    src: &HashMap<String, Tensor>,
    out: &mut HashMap<String, Tensor>,
    layer: usize,
    target: &str, // "to_q" | "to_k" | "to_v"
    target_rank: usize,
    device: &Arc<CudaDevice>,
    count: &mut usize,
    padded: &mut usize,
) -> Result<()> {
    let prefix_src = format!("diffusion_model.layers.{layer}.attention.{target}");
    let down_key = format!("{prefix_src}.lora_down.weight");
    let up_key = format!("{prefix_src}.lora_up.weight");
    let (down, up) = match (src.get(&down_key), src.get(&up_key)) {
        (Some(d), Some(u)) => (d, u),
        _ => return Ok(()),
    };
    let rank = down.shape().dims()[0];
    let alpha = read_alpha(src, &prefix_src, rank)?;
    let scale_src = alpha / rank as f32;
    let padded_down = pad_rows(down, target_rank, device)?;
    let padded_up = pad_cols(up, target_rank, device)?;
    let scaled_up = padded_up.to_dtype(DType::F32)?.mul_scalar(scale_src)?;
    let prefix_dst = format!("layers.{layer}.attention.{target}");
    out.insert(format!("{prefix_dst}.lora_A"), padded_down.to_dtype(DType::F32)?);
    out.insert(format!("{prefix_dst}.lora_B"), scaled_up);
    *count += 1;
    if rank < target_rank {
        *padded += 1;
    }
    Ok(())
}

/// DiffusersKohya: convert an arbitrary module (out / feed_forward) with
/// rank padding and scale pre-bake. Source uses `diffusion_model.layers.<L>.<src>`,
/// destination uses `layers.<L>.<dst>` (trainer naming).
#[allow(clippy::too_many_arguments)]
fn convert_dk_module(
    src: &HashMap<String, Tensor>,
    out: &mut HashMap<String, Tensor>,
    layer: usize,
    src_module: &str, // "attention.to_out.0" or "feed_forward.w1" etc.
    dst_suffix: &str, // "attention.out" or "feed_forward.w1" etc.
    target_rank: usize,
    in_dim: usize,
    out_dim: usize,
    device: &Arc<CudaDevice>,
    count: &mut usize,
    padded: &mut usize,
) -> Result<()> {
    let prefix_src = format!("diffusion_model.layers.{layer}.{src_module}");
    let down_key = format!("{prefix_src}.lora_down.weight");
    let up_key = format!("{prefix_src}.lora_up.weight");
    let (down, up) = match (src.get(&down_key), src.get(&up_key)) {
        (Some(d), Some(u)) => (d, u),
        _ => return Ok(()),
    };
    let down_dims = down.shape().dims().to_vec();
    let up_dims = up.shape().dims().to_vec();
    let rank = down_dims[0];
    if down_dims[1] != in_dim || up_dims[0] != out_dim || up_dims[1] != rank {
        eprintln!(
            "[convert] L{layer} {src_module}: shape mismatch down={:?} up={:?}",
            down_dims, up_dims
        );
        return Ok(());
    }
    let alpha = read_alpha(src, &prefix_src, rank)?;
    let scale_src = alpha / rank as f32;
    let padded_down = pad_rows(down, target_rank, device)?;
    let padded_up = pad_cols(up, target_rank, device)?;
    let scaled_up = padded_up.to_dtype(DType::F32)?.mul_scalar(scale_src)?;
    let prefix_dst = format!("layers.{layer}.{dst_suffix}");
    out.insert(format!("{prefix_dst}.lora_A"), padded_down.to_dtype(DType::F32)?);
    out.insert(format!("{prefix_dst}.lora_B"), scaled_up);
    *count += 1;
    if rank < target_rank {
        *padded += 1;
    }
    Ok(())
}

/// Pad a tensor [r, c] with zeros along axis 0 to reach `target_rows`.
fn pad_rows(
    t: &Tensor,
    target_rows: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let dims = t.shape().dims();
    let cur = dims[0];
    let cols = dims[1];
    if cur == target_rows {
        return t.contiguous();
    }
    if cur > target_rows {
        return Err(FlameError::InvalidInput(format!(
            "pad_rows shrinking not supported: cur={cur} target={target_rows}"
        )));
    }
    let pad = target_rows - cur;
    let zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[pad, cols]),
        t.dtype(),
        device.clone(),
    )?;
    Tensor::cat(&[t, &zeros], 0)
}

/// Pad a tensor [r, c] with zeros along axis 1 to reach `target_cols`.
fn pad_cols(
    t: &Tensor,
    target_cols: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let dims = t.shape().dims();
    let rows = dims[0];
    let cur = dims[1];
    if cur == target_cols {
        return t.contiguous();
    }
    if cur > target_cols {
        return Err(FlameError::InvalidInput(format!(
            "pad_cols shrinking not supported: cur={cur} target={target_cols}"
        )));
    }
    let pad = target_cols - cur;
    let zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[rows, pad]),
        t.dtype(),
        device.clone(),
    )?;
    Tensor::cat(&[t, &zeros], 1)
}
