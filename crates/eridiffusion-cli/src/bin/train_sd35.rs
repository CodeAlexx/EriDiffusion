//! train_sd35 — SD 3.5 Medium / Large LoRA training.
//!
//! Pipeline per step (matches upstream Python `BaseStableDiffusion3Setup`):
//!   1. Load cached `latent` ([B, 16, h, w] BF16, pre-scaled), `text_embedding`
//!      ([B, seq, 4096] BF16), `pooled` ([B, 2048] BF16).
//!   2. Per batch element: sample timestep ∈ [0, num_train_timesteps) per
//!      LOGIT_NORMAL with shift=`--timestep-shift` (preset default 1.0).
//!   3. sigma = (floor(t)+1) / num_train_timesteps;  noisy = sigma*noise + (1-sigma)*latent
//!   4. predicted = model(noisy, t_model, context, pooled)
//!   5. target    = noise - latent                (rectified flow)
//!   6. loss      = mean MSE in F32                (no v-pred preconditioning)
//!
//! Differs from `flame-diffusion/sd3-trainer`'s `pipeline::compute_sd3_loss`,
//! which applies `model_pred = -sigma*model_pred + noisy_input` before the
//! MSE — that's the kohya path. upstream Python's path is plain MSE on
//! `noise - clean` per `BaseStableDiffusion3Setup.py:319-333` (verified). We
//! follow upstream Python.
//!
//! Single seed=42 across step + sample loops (memory: feedback_default_seed_42).

use clap::Parser;
use std::path::PathBuf;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use flame_core::adam::AdamW;
use eridiffusion_core::config::{LrScheduler, TrainConfig, TrainingMethod};
use eridiffusion_core::lycoris::{LoraInitType, LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::{loss_weight, lr_schedule, noise_modifiers, timestep_bias, validation::ValidationLoop};
use eridiffusion_core::training::training_features::OptimizerKind;
use eridiffusion_core::debug as dbg;
use eridiffusion_core::encoders::clip_g::ClipGEncoder;
use eridiffusion_core::encoders::clip_l::{ClipConfig, ClipEncoder};
use eridiffusion_core::encoders::t5_xxl::T5Encoder;
use eridiffusion_core::encoders::flux_vae_decoder::LdmVAEDecoder;
use eridiffusion_core::models::{sd35::SD35Model, TrainableModel};
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const LOGIT_NORMAL_BIAS: f32 = 0.0;     // OT default `noising_bias`
const LOGIT_NORMAL_SCALE: f32 = 1.0;    // OT default `noising_weight + 1`
const TIMESTEP_SHIFT_DEFAULT: f32 = 1.0;
const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;

const CLIP_L_PAD_ID: i32 = 49407;
const CLIP_G_PAD_ID: i32 = 0;
const CLIP_MAX_LEN: usize = 77;
// HARD: T5 padded sequence length. Combined CLIP+T5 context is `[B, 154, 4096]`.
// Mirrors `prepare_sd35.rs::TXT_PAD_LEN`. OT `StableDiffusion3BaseDataLoader.py:33`
// passes `tokenizer_1.model_max_length=77` to all three tokenizers.
const TXT_PAD_LEN: usize = 77;
// HARD: 1024×1024 is the only supported training resolution.
const TRAIN_RES: u32 = 1024;

const VAE_LATENT_CHANNELS: usize = 16;
const VAE_SCALE: f32 = 1.5305;
const VAE_SHIFT: f32 = 0.0609;

#[derive(Parser)]
struct Args {
    #[arg(long)] config: PathBuf,
    #[arg(long)] cache_dir: PathBuf,
    /// SD3.5 Medium/Large transformer safetensors (combined ckpt or DiT-only).
    /// Either single file or shard directory.
    #[arg(long)] transformer: PathBuf,
    #[arg(long, default_value = "1000")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "1.0")] lora_alpha: f64,
    /// SD3 LoRA preset default lr=3e-4.
    #[arg(long, default_value = "3e-4")] lr: f32,
    /// SD3 LoRA preset default batch_size=4. SD3.5-large at 1024² won't
    /// fit batch=4 on 24 GB — drop to 1 or 2 for that model.
    #[arg(long, default_value = "1")] batch_size: usize,
    /// Discrete-flow timestep shift. OT preset has no override → defaults
    /// to 1.0 (no shift). The diffusers/inference-time schedule uses 3.0.
    #[arg(long, default_value_t = TIMESTEP_SHIFT_DEFAULT)] timestep_shift: f32,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA weights + AdamW (m, v, t) + step counter.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Periodic + final save mode. Default `full` (LoRA + AdamW + step) for
    /// resumable runs. `weights` writes legacy weights-only files.
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "output")] output_dir: PathBuf,

    // ── Periodic save + sample (every N steps) ──────────────────────────
    /// Render an inline sample every N steps (0 = off). Loads the text
    /// encoders + VAE once up front and drops them after encoding the
    /// fixed prompt to keep training-time VRAM bounded.
    #[arg(long, default_value = "0")] sample_every: usize,
    /// Save a LoRA checkpoint every N steps (0 = off, save only at end).
    /// Independent from --sample-every (you can have one without the other).
    #[arg(long, default_value = "0")] save_every: usize,
    /// Prompt for the periodic sample.
    #[arg(long, default_value = "")] sample_prompt: String,
    /// Negative / unconditional prompt for CFG.
    #[arg(long, default_value = "")] sample_neg_prompt: String,
    /// SD3.5 VAE safetensors (defaults to --transformer if it carries the VAE).
    #[arg(long)] sample_vae: Option<PathBuf>,
    #[arg(long)] sample_clip_l: Option<PathBuf>,
    #[arg(long)] sample_clip_g: Option<PathBuf>,
    #[arg(long)] sample_t5: Option<PathBuf>,
    #[arg(long)] sample_clip_l_tokenizer: Option<PathBuf>,
    #[arg(long)] sample_clip_g_tokenizer: Option<PathBuf>,
    #[arg(long)] sample_t5_tokenizer: Option<PathBuf>,
    /// HARD-locked to 1024 — the only supported training/sample resolution.
    #[arg(long, default_value_t = TRAIN_RES as usize)] sample_size: usize,
    #[arg(long, default_value = "28")] sample_steps: usize,
    #[arg(long, default_value = "4.5")] sample_cfg: f32,
    /// Inference-time schedule shift. SD3 reference uses 3.0.
    #[arg(long, default_value = "3.0")] sample_shift: f32,
    /// HARD-locked to 77 — combined CLIP+T5 context shape is `[B, 154, 4096]`.
    #[arg(long, default_value_t = TXT_PAD_LEN)] sample_t5_max_len: usize,
    #[arg(long, default_value = "42")] sample_seed: u64,

    // ── Phase 0 multi-feature rollout (default-off; Phase 1+ will consume) ──
    #[arg(long)] min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)] caption_dropout_probability: f32,
    /// Path to a single cache file produced by `prepare_sd35` from an empty-
    /// caption sample. When `--caption-dropout-probability > 0`, the trainer
    /// loads `text_embedding` + `pooled` from this file and swaps them in
    /// (correlated) with probability `p` per step. If unset and dropout > 0,
    /// the feature is disabled with a warning.
    #[arg(long)] null_text_cache: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)] noise_offset_probability: f32,
    #[arg(long, default_value_t = 0.0)] gamma_input_perturbation: f32,
    #[arg(long, default_value_t = 0.0)] huber_strength: f32,
    #[arg(long, default_value_t = 0.0)] lr_min_factor: f32,
    #[arg(long)] validation_dataset_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0)] validation_every_steps: u64,
    #[arg(long, num_args = 0..)] multi_backend_weights: Vec<f32>,
    /// Phase 2: paired with --multi-backend-weights. Klein-only wiring; other
    /// trainers accept-and-warn until per-model wiring lands.
    #[arg(long, num_args = 0..)] multi_backend_cache_dirs: Vec<std::path::PathBuf>,
    /// Phase 2: validation prompt library JSON (Klein-only wiring; other
    /// trainers accept-and-warn).
    #[arg(long)] validation_prompts_file: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 0.0)] masked_loss_weight: f32,
    /// Master switch for EMA shadow. See train_klein.rs for full doc.
    /// Default off → byte-identical to pre-flag commits.
    #[arg(long, default_value_t = false)] ema: bool,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    #[arg(long, default_value_t = 0.9999)] ema_max_decay: f32,
    /// Swap shadow → live params at sample/save time.
    #[arg(long, default_value_t = false)] ema_validation_swap: bool,
    /// Pyramid / multi-resolution noise: number of additional resolution
    /// levels to mix into the per-step training noise. `0` (default) is a
    /// no-op — byte-identical to no-multires.
    #[arg(long, default_value_t = 0)] multires_noise_iterations: usize,
    /// Per-level discount factor for `--multires-noise-iterations`.
    #[arg(long, default_value_t = 0.3)] multires_noise_discount: f32,
    /// Multi-distribution timestep bias strategy. `none` is byte-identical.
    #[arg(long, default_value = "none")] timestep_bias_strategy: String,
    #[arg(long, default_value_t = 0.0)] timestep_bias_multiplier: f32,
    #[arg(long, default_value_t = 0.0)] timestep_bias_range_min: f32,
    #[arg(long, default_value_t = 1.0)] timestep_bias_range_max: f32,
    #[arg(long)] tread_route_pattern: Option<String>,
    /// Phase 1: optimizer family CLI surface (Phase 5 wires full dispatch).
    #[arg(long, default_value = "adamw")] optimizer: String,

    // ── Phase 6 multi-feature rollout (plumb-only; multi-backend wired in Klein) ──
    #[arg(long, num_args = 0..)] multi_backend_repeats: Vec<u32>,
    #[arg(long, default_value_t = false)] caption_tag_shuffle: bool,
    #[arg(long, default_value_t = false)] cache_clear_each_epoch: bool,
    #[arg(long, default_value_t = false)] cache_invalidate: bool,
    /// Phase 5: LR scheduler family. Default `constant` + `warmup_steps=0` is
    /// byte-equivalent to prior fixed-LR behaviour.
    #[arg(long, default_value = "constant")] lr_scheduler: String,
    /// Phase 5: linear LR warmup steps. Default 0 keeps prior behaviour.
    #[arg(long, default_value_t = 0)] warmup_steps: usize,
    /// Phase 5: cosine-with-restarts cycle count.
    #[arg(long, default_value_t = 1.0)] lr_cycles: f32,

    // ── LyCORIS algo selection (Phase 2b) ──
    //
    // `--algo lora` (default) keeps the legacy `LoRALinear` path —
    // byte-identical to pre-Phase-2b training. Other values build a
    // `LycorisLinear` bundle via `SD35Model::install_lycoris_bundle`.
    // `lora_alpha` and `rank` are shared with the legacy CLI flags above
    // (no separate `--lycoris-rank`).
    /// LyCORIS algo: `lora` (default, legacy path) | `locon` | `loha`
    /// | `lokr` | `full` | `oft`. `full` and `oft` build successfully but
    /// their `forward_delta` will error inside SD3.5's
    /// `base + delta_on_input` joint-block call pattern — Phase 2c will
    /// wire `merge_into_base` for those.
    #[arg(long, default_value = "lora")] algo: String,
    /// LoKr Kronecker split factor (ignored for non-LoKr).
    #[arg(long, default_value_t = 16)] lokr_factor: i32,
    /// OFT block size (ignored for non-OFT).
    #[arg(long, default_value_t = 32)] oft_block_size: usize,
    /// OFT Cayley-Neumann series term count (ignored for non-OFT).
    #[arg(long, default_value_t = 5)] oft_neumann_terms: usize,
    /// LoCon / LoHa / LoKr conv variant — Tucker decomposition for non-1×1
    /// kernels. SD3.5's LoRA targets are linear-only so this is currently
    /// a no-op.
    #[arg(long, default_value_t = false)] use_tucker: bool,
    /// LoKr only: factorize both W1 *and* W2 (default false: only W2).
    #[arg(long, default_value_t = false)] decompose_both: bool,
    /// Enable DoRA (weight-decomposed LoRA). Applies to LoCon/LoHa/LoKr
    /// (Full inherits, OFT errors).
    ///
    /// Phase 2b limitation: SD3.5's resident base-weight map is HashMap-keyed
    /// rather than HashMap<(block,target), Tensor>, so DoRA's magnitude
    /// initializes from `||I||_2 = 1` rather than `||W_orig||_2`. Trainer
    /// should still converge — first few hundred steps adjust magnitude.
    /// Phase 2c will wire pre-load magnitude init.
    #[arg(long, default_value_t = false)] dora: bool,
    /// DoRA magnitude axis. Default `true` matches lycoris-upstream
    /// (norm over input dims, magnitude shape `[out, 1]`).
    #[arg(long, default_value_t = true)] dora_wd_on_out: bool,
    #[arg(long, default_value_t = 1e-6)] dora_eps: f32,
    /// PEFT/SimpleTuner `--lora_init_type`. Applies to LoCon (the LoRA path)
    /// only. Choices: `default | gaussian | pissa | olora | loftq`. The
    /// PISSA/OLoRA/LoftQ variants parse but error at adapter construction
    /// because flame-core does not yet expose SVD/QR.
    #[arg(long, default_value = "default")] lora_init_type: String,
    /// SimpleTuner-style `lycoris_config preset.json`. Optional; per-target
    /// `module_algo_map` overrides apply during adapter construction.
    #[arg(long)] lycoris_config: Option<PathBuf>,
    /// SimpleTuner-parity perturbed-normal LoKr init. No-op for SD3.5 in
    /// Phase 2b (base-weight-by-target lookup not plumbed).
    #[arg(long, default_value_t = 0.0)] init_lokr_norm: f32,
    /// Per-element dropout on the adapter delta (training only).
    #[arg(long, default_value_t = 0.0)] lora_dropout: f32,
    /// Per-rank Bernoulli on the down-projection intermediate.
    #[arg(long, default_value_t = 0.0)] rank_dropout: f32,
    /// Per-step Bernoulli on the entire adapter.
    #[arg(long, default_value_t = 0.0)] module_dropout: f32,
    /// Rescale rank-mask by `1/mean(mask)` to preserve expectation.
    #[arg(long, default_value_t = false)] rank_dropout_scale: bool,
}

/// LOGIT_NORMAL timestep sample → continuous t in `[min_t, max_t)`.
/// Mirrors OT `_get_timestep_discrete` (LOGIT_NORMAL branch, line 154-160 +
/// 172): scales the unit-interval logit-normal draw into
/// `[min_noising_strength, max_noising_strength) * num_train_timesteps`,
/// then applies the resolution-aware shift.
fn sample_timestep_logit_normal(
    rng: &mut rand::rngs::StdRng,
    shift: f32,
    min_strength: f32,
    max_strength: f32,
) -> f32 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(LOGIT_NORMAL_BIAS, LOGIT_NORMAL_SCALE).unwrap();
    let z = normal.sample(rng);
    let logit_normal = 1.0 / (1.0 + (-z).exp());
    let min_t = NUM_TRAIN_TIMESTEPS as f32 * min_strength.max(0.0);
    let max_t = NUM_TRAIN_TIMESTEPS as f32 * max_strength.min(1.0);
    let t = logit_normal * (max_t - min_t) + min_t;
    if (shift - 1.0).abs() < 1e-6 {
        t
    } else {
        NUM_TRAIN_TIMESTEPS as f32 * shift * t
            / ((shift - 1.0) * t + NUM_TRAIN_TIMESTEPS as f32)
    }
}

fn collect_shards(path: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_file() { return Ok(vec![path.to_path_buf()]); }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() { anyhow::bail!("no safetensors at {:?}", path); }
    Ok(shards)
}

fn load_one_or_dir(
    path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>,
) -> flame_core::Result<std::collections::HashMap<String, Tensor>> {
    if path.is_file() { return flame_core::serialization::load_file(path, device); }
    let mut all = std::collections::HashMap::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| flame_core::Error::Io(format!("read_dir: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    entries.sort();
    for p in entries {
        all.extend(flame_core::serialization::load_file(&p, device)?);
    }
    Ok(all)
}

fn tokenize_clip(tok: &tokenizers::Tokenizer, text: &str, pad_id: i32) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(text, true).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if ids.len() > CLIP_MAX_LEN { ids.truncate(CLIP_MAX_LEN); }
    while ids.len() < CLIP_MAX_LEN { ids.push(pad_id); }
    Ok(ids)
}

fn tokenize_t5(tok: &tokenizers::Tokenizer, text: &str, max_len: usize) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(text, true).map_err(|e| anyhow::anyhow!("t5 tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if ids.len() > max_len { ids.truncate(max_len); }
    while ids.len() < max_len { ids.push(0); }
    Ok(ids)
}

/// HIGH-1 fix: 3 independent Bernoulli draws per OT
/// `StableDiffusion3Model.encode_text:397-415`. Each encoder is masked
/// independently — CLIP-L, CLIP-G, T5 — into both the combined `text_embedding`
/// `[B, 154, 4096]` and the pooled `[B, 2048]` (T5 has no pooled component).
///
/// Combined tensor layout (must match `prepare_sd35.rs::TXT_PAD_LEN=77`):
///   text_embedding rows  0..77 : padded CLIP-L+CLIP-G (channels 0..768
///                                = CLIP-L, 768..2048 = CLIP-G, 2048..4096 = pad)
///   text_embedding rows 77..154: T5 hidden (full 4096 channels)
///   pooled channels  0..768  : CLIP-L pooled
///   pooled channels  768..2048: CLIP-G pooled
fn apply_per_encoder_dropout(
    text: &Tensor,
    pooled: &Tensor,
    drop_clip_l: bool,
    drop_clip_g: bool,
    drop_t5: bool,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<(Tensor, Tensor)> {
    if !drop_clip_l && !drop_clip_g && !drop_t5 {
        return Ok((text.clone(), pooled.clone()));
    }
    // text shape: [B, 154, 4096]
    let dims = text.dims().to_vec();
    let (b, seq, hidden) = (dims[0], dims[1], dims[2]);
    debug_assert_eq!(seq, 154, "combined seq must be 154 (OT-parity)");
    debug_assert_eq!(hidden, 4096, "context hidden must be 4096");

    // Split text along seq into CLIP rows (0..77) and T5 rows (77..154).
    let clip_rows = text.narrow(1, 0, CLIP_MAX_LEN)?;       // [B, 77, 4096]
    let t5_rows = text.narrow(1, CLIP_MAX_LEN, CLIP_MAX_LEN)?; // [B, 77, 4096]

    // Within CLIP rows, channels split: 0..768 (CLIP-L), 768..2048 (CLIP-G),
    // 2048..4096 (zero pad — leave alone regardless).
    let clip_rows = if drop_clip_l || drop_clip_g {
        let clip_l_part = clip_rows.narrow(2, 0, 768)?;
        let clip_g_part = clip_rows.narrow(2, 768, 1280)?;
        let pad_part = clip_rows.narrow(2, 2048, 2048)?;
        let clip_l_part = if drop_clip_l { clip_l_part.mul_scalar(0.0)? } else { clip_l_part };
        let clip_g_part = if drop_clip_g { clip_g_part.mul_scalar(0.0)? } else { clip_g_part };
        Tensor::cat(&[&clip_l_part, &clip_g_part, &pad_part], 2)?
    } else {
        clip_rows
    };

    let t5_rows = if drop_t5 { t5_rows.mul_scalar(0.0)? } else { t5_rows };
    let new_text = Tensor::cat(&[&clip_rows, &t5_rows], 1)?;

    // Pooled: 0..768 CLIP-L, 768..2048 CLIP-G.
    let pdims = pooled.dims().to_vec();
    debug_assert_eq!(pdims[1], 2048, "pooled hidden must be 2048");
    let pool_l = pooled.narrow(1, 0, 768)?;
    let pool_g = pooled.narrow(1, 768, 1280)?;
    let pool_l = if drop_clip_l { pool_l.mul_scalar(0.0)? } else { pool_l };
    let pool_g = if drop_clip_g { pool_g.mul_scalar(0.0)? } else { pool_g };
    let new_pooled = Tensor::cat(&[&pool_l, &pool_g], 1)?;

    let _ = (b, device); // silence unused
    Ok((new_text, new_pooled))
}

/// Encode one prompt → `(context [1, seq, 4096], pooled [1, 2048])`.
/// Same shape contract as `prepare_sd35` cache.
fn encode_sd3_prompt(
    text: &str,
    clip_l: &ClipEncoder, clip_g: &ClipGEncoder, t5: &mut T5Encoder,
    tok_l: &tokenizers::Tokenizer, tok_g: &tokenizers::Tokenizer, tok_t5: &tokenizers::Tokenizer,
    t5_max_len: usize,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<(Tensor, Tensor)> {
    let ids_l = tokenize_clip(tok_l, text, CLIP_L_PAD_ID)?;
    let (clip_l_h, clip_l_pool) = clip_l.encode_sd3(&ids_l)?;
    let ids_g = tokenize_clip(tok_g, text, CLIP_G_PAD_ID)?;
    let (clip_g_h, clip_g_pool) = clip_g.encode_sdxl(&ids_g)?;
    let ids_t5 = tokenize_t5(tok_t5, text, t5_max_len)?;
    let t5_h = t5.encode(&ids_t5)?;

    let clip_lg = Tensor::cat(&[&clip_l_h, &clip_g_h], 2)?;
    let pad_zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[clip_lg.dims()[0], clip_lg.dims()[1], 4096 - 2048]),
        DType::BF16, device.clone(),
    )?;
    let clip_lg_padded = Tensor::cat(&[&clip_lg.to_dtype(DType::BF16)?, &pad_zeros], 2)?;
    let context = Tensor::cat(&[&clip_lg_padded, &t5_h.to_dtype(DType::BF16)?], 1)?
        .to_dtype(DType::BF16)?;
    let pooled = Tensor::cat(&[&clip_l_pool, &clip_g_pool], 1)?.to_dtype(DType::BF16)?;
    Ok((context, pooled))
}

// SD3 inference schedule — matches inference-flame `sd3_lora_infer` /
// flame-diffusion `sd3-trainer/src/sampling.rs` exactly.
fn build_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            if *v > 0.0 && *v < 1.0 {
                *v = shift * *v / (1.0 + (shift - 1.0) * *v);
            }
        }
    }
    t
}

fn save_png(rgb: &Tensor, path: &std::path::Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let dims = rgb_f32.dims().to_vec();
    let (h, w) = (dims[2], dims[3]);
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                let v = data[idx].clamp(-1.0, 1.0);
                let u = ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                pixels[(y * w + x) * 3 + c] = u;
            }
        }
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("RgbImage::from_raw failed"))?
        .save(path)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn inline_sample(
    model: &mut SD35Model,
    context: &Tensor,
    pooled: &Tensor,
    neg_context: &Tensor,
    neg_pooled: &Tensor,
    vae_path: &std::path::Path,
    out_path: &std::path::Path,
    size: usize,
    steps: usize,
    cfg: f32,
    shift: f32,
    seed: u64,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<()> {
    let _no_grad = AutogradContext::no_grad();
    let h_lat = size / 8;
    let w_lat = size / 8;
    let numel = VAE_LATENT_CHANNELS * h_lat * w_lat;

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(numel);
    while data.len() < numel {
        let u1 = rng.gen::<f32>().max(1e-10);
        let u2 = rng.gen::<f32>();
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        data.push(mag * theta.cos());
        if data.len() < numel { data.push(mag * theta.sin()); }
    }
    let mut latent = Tensor::from_vec(
        data, Shape::from_dims(&[1, VAE_LATENT_CHANNELS, h_lat, w_lat]), device.clone(),
    )?.to_dtype(DType::BF16)?;

    let timesteps = build_schedule(steps, shift);
    for i in 0..steps {
        let t_curr = timesteps[i];
        let t_next = timesteps[i + 1];
        // MED-1 fix: keep timestep in F32. BF16 has 8-bit mantissa → loses
        // 1-LSB precision for integer values >256, which is most of the
        // [0, 999] range. Train ↔ inference timestep embedding parity breaks
        // otherwise. The MMDiT timestep_embed promotes to F32 internally
        // (sd35.rs:398), so passing F32 here is zero cost.
        let t_vec = Tensor::from_vec(
            vec![t_curr * 1000.0], Shape::from_dims(&[1]), device.clone(),
        )?.to_dtype(DType::F32)?;
        let pred_cond = model.forward_inner(&latent, &t_vec, context, pooled)?;
        let pred_uncond = model.forward_inner(&latent, &t_vec, neg_context, neg_pooled)?;
        let diff = pred_cond.sub(&pred_uncond)?;
        let pred = pred_uncond.add(&diff.mul_scalar(cfg)?)?;
        let dt = t_next - t_curr;
        latent = latent.add(&pred.mul_scalar(dt)?)?;
    }

    // VAE decode — eridiffusion-core's `LdmVAEDecoder` is the generic LDM
    // AutoencoderKL decoder used for SD3 (16ch), SDXL/SD1.5 (4ch), Z-Image,
    // etc. SD3 normalization: scale=1.5305, shift=0.0609.
    let vae = LdmVAEDecoder::from_safetensors(
        vae_path.to_str().ok_or_else(|| anyhow::anyhow!("vae path utf8"))?,
        VAE_LATENT_CHANNELS, VAE_SCALE, VAE_SHIFT, device,
    )?;
    let rgb = vae.decode(&latent)?;
    save_png(&rgb, out_path)
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    env_logger::init();
    let args = Args::parse();
    // HARD: T5 max len locked to 77 (combined seq=154). Both training and
    // sample paths must use 77 — anything else hard-fails.
    if args.sample_t5_max_len != TXT_PAD_LEN {
        anyhow::bail!(
            "--sample-t5-max-len must be {TXT_PAD_LEN} (combined context = 154 tokens, OT-parity); got {}",
            args.sample_t5_max_len
        );
    }
    // HARD: 1024 is the only supported sample/inline resolution.
    if args.sample_size as u32 != TRAIN_RES {
        anyhow::bail!(
            "--sample-size must be {TRAIN_RES} (only 1024×1024 is supported); got {}",
            args.sample_size
        );
    }
    // MED-3: OT raises NotImplementedError for Min-SNR-γ on flow-matching loss
    // (`ModelSetupDiffusionLossMixin._flow_matching_losses` accepts only
    // CONSTANT and SIGMA). User-directed exception: keep wired but warn.
    if args.min_snr_gamma.is_some() {
        log::warn!(
            "[divergence] --min-snr-gamma={} is wired into the flow-matching loss; OT FORBIDS this for SD3 (FM has no SNR). User-requested override.",
            args.min_snr_gamma.unwrap()
        );
    }
    // Phase 2: Klein-only wiring of multi-backend + validation prompts library.
    // Other trainers accept-and-warn so configs/launchers aren't broken; full
    // wiring is a follow-up after the per-model encoder + sample paths are
    // consolidated.
    if !args.multi_backend_cache_dirs.is_empty() || !args.multi_backend_weights.is_empty() {
        log::warn!("--multi-backend-* flags are Klein-only in Phase 2; ignored here");
    }
    if args.validation_prompts_file.is_some() {
        log::warn!("--validation-prompts-file is Klein-only in Phase 2; ignored here");
    }
    std::fs::create_dir_all(&args.output_dir)?;
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let mut config = TrainConfig::from_json_path(&args.config.to_string_lossy())?;
    config.training_method = TrainingMethod::Lora;
    config.lora_rank = args.rank as u64;
    config.lora_alpha = args.lora_alpha;
    config.learning_rate = args.lr as f64;

    // Phase 0 multi-feature rollout — plumb CLI args into config (default-off, unused yet).
    config.min_snr_gamma = args.min_snr_gamma;
    config.caption_dropout_probability = args.caption_dropout_probability;
    config.noise_offset_probability = args.noise_offset_probability;
    config.gamma_input_perturbation = args.gamma_input_perturbation;
    config.huber_strength = args.huber_strength;
    config.lr_min_factor = args.lr_min_factor;
    config.validation_dataset_dir = args.validation_dataset_dir.clone();
    config.validation_every_steps = args.validation_every_steps;
    config.multi_backend_weights = args.multi_backend_weights.clone();
    config.validation_prompts_file = args.validation_prompts_file.clone();
    config.masked_loss_weight = args.masked_loss_weight;
    config.ema_inv_gamma = args.ema_inv_gamma;
    config.ema_power = args.ema_power;
    config.ema_update_after_step = args.ema_update_after_step;
    config.ema_min_decay = args.ema_min_decay;
    config.tread_route_pattern = args.tread_route_pattern.clone();

    let shards = collect_shards(&args.transformer)?;
    log::info!("Loading SD3.5 transformer from {} shard(s) (rank={} alpha={})",
        shards.len(), args.rank, args.lora_alpha);
    let mut model = SD35Model::load(&shards, &config, device.clone())?;

    // Phase 2b: parse LyCORIS algo selector. `--algo lora` (or `none`/empty)
    // keeps the legacy `LoRALinear` bundle that `SD35Model::load` already
    // built, so this branch is byte-equivalent to pre-Phase-2b training.
    // `LycorisAlgo::parse("lora")` aliases to `LycorisAlgo::LoCon`; we
    // re-map `"lora"` → `None` here explicitly so plain LoRA stays on the
    // legacy path. Users wanting the new LoCon adapter pass `--algo locon`.
    let algo_str = args.algo.trim().to_ascii_lowercase();
    let algo = if algo_str == "lora" || algo_str == "none" || algo_str.is_empty() {
        LycorisAlgo::None
    } else {
        LycorisAlgo::parse(&args.algo).map_err(|e| anyhow::anyhow!("--algo: {e}"))?
    };
    let lyc_config = LycorisBundleConfig {
        algo,
        rank: args.rank,
        alpha: args.lora_alpha as f32,
        factor: args.lokr_factor,
        block_size: args.oft_block_size,
        neumann_terms: args.oft_neumann_terms,
        use_tucker: args.use_tucker,
        decompose_both: args.decompose_both,
        use_scalar: false,
        dora: args.dora,
        dora_wd_on_out: args.dora_wd_on_out,
        dora_eps: args.dora_eps,
        init_type: LoraInitType::parse(&args.lora_init_type)
            .map_err(|e| anyhow::anyhow!("--lora_init_type: {e}"))?,
        ..LycorisBundleConfig::default()
    };
    let lyc_config = lyc_config
        .with_optional_lycoris_config_file(args.lycoris_config.as_deref())?;

    if algo != LycorisAlgo::None {
        log::info!(
            "[SD3.5] LyCORIS algo='{}' rank={} alpha={} factor={} block_size={} dora={}",
            algo.as_str(),
            lyc_config.rank,
            lyc_config.alpha,
            lyc_config.factor,
            lyc_config.block_size,
            lyc_config.dora,
        );
        if matches!(algo, LycorisAlgo::Full | LycorisAlgo::Oft) {
            log::warn!(
                "[SD3.5] algo='{}' selected — bundle construction will succeed, but \
                 forward_delta will error inside SD3.5's `base + delta_on_input` joint-block \
                 pattern. Phase 2c will wire merge-into-base for these algos.",
                algo.as_str()
            );
        }
        model.install_lycoris_bundle(&lyc_config, device.clone(), SEED)
            .map_err(|e| anyhow::anyhow!("LyCORIS bundle install: {e}"))?;
    } else {
        log::info!("[SD3.5] algo='lora' (legacy LoRALinear path, byte-identical)");
    }

    // Reference-only: silence unused-warning for dropout/init flags wired in
    // Block 2 but not yet plumbed past `LycorisBundleConfig` (Phase 2c).
    let _ = (args.init_lokr_norm, args.lora_dropout, args.rank_dropout,
             args.module_dropout, args.rank_dropout_scale);

    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA parameters", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable LoRA parameters — check model.is_lora and config.training_method");
    }

    match OptimizerKind::parse(&args.optimizer) {
        Ok(OptimizerKind::AdamW) => {}
        Ok(other) => log::warn!(
            "non-AdamW optimizer selected: {} — Phase 1 falls back to AdamW (full dispatch in Phase 5)",
            other.as_str()
        ),
        Err(e) => log::warn!("--optimizer parse: {} — falling back to AdamW", e),
    }
    let effective_caption_dropout_prob = args.caption_dropout_probability;
    // HIGH-1: OT-style 3 INDEPENDENT per-encoder Bernoullis. Each step,
    // CLIP-L / CLIP-G / T5 are zeroed independently with prob `p` each,
    // matching `StableDiffusion3Model.py:397-415`. `--null-text-cache` is
    // OPTIONAL — it's only used as a small optimisation when all three
    // encoders draw on the same step.
    let null_text: Option<(Tensor, Tensor)> = if effective_caption_dropout_prob > 0.0 {
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt = s.get("text_embedding")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 'text_embedding'"))?
                        .to_dtype(DType::BF16)?;
                    let nt_d = nt.dims();
                    if nt_d.len() != 3 || nt_d[1] != 2 * TXT_PAD_LEN || nt_d[2] != 4096 {
                        anyhow::bail!(
                            "--null-text-cache text_embedding shape {:?} but expected [B, {}, 4096]",
                            nt_d, 2 * TXT_PAD_LEN
                        );
                    }
                    let np = s.get("pooled")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 'pooled'"))?
                        .to_dtype(DType::BF16)?;
                    let np_d = np.dims();
                    if np_d.len() != 2 || np_d[1] != 2048 {
                        anyhow::bail!(
                            "--null-text-cache pooled shape {:?} but expected [B, 2048]",
                            np_d
                        );
                    }
                    log::info!(
                        "[caption-dropout] WIRED OT-style — per-encoder p={:.3} (null cache provided as 3-of-3 fast path: text={:?} pooled={:?})",
                        effective_caption_dropout_prob,
                        nt.shape().dims(),
                        np.shape().dims()
                    );
                    Some((nt, np))
                }
                Err(e) => {
                    log::warn!(
                        "[caption-dropout] failed to load --null-text-cache {}: {e} — using slice-zero only",
                        p.display()
                    );
                    None
                }
            },
            None => {
                log::info!(
                    "[caption-dropout] WIRED OT-style — per-encoder p={:.3} (slice-zero, no null-text-cache)",
                    effective_caption_dropout_prob
                );
                None
            }
        }
    } else {
        None
    };
    let mut opt = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);

    // EMA shadow (Phase 3). See train_klein.rs for the same pattern.
    let ema_cfg = EmaConfig {
        inv_gamma: args.ema_inv_gamma,
        power: args.ema_power,
        update_after_step: args.ema_update_after_step,
        min_decay: args.ema_min_decay,
        max_decay: args.ema_max_decay,
    };
    let mut ema: Option<ParameterEma> = if args.ema {
        let _g = AutogradContext::no_grad();
        let e = ParameterEma::new(&params, args.ema_max_decay)
            .map_err(|e| anyhow::anyhow!("EMA construction failed: {e}"))?;
        log::info!(
            "[ema] WIRED — {} shadow tensors, swap={}",
            e.len(),
            args.ema_validation_swap
        );
        Some(e)
    } else {
        None
    };

    // Timestep bias config — defaults are byte-identical (Strategy::None).
    let timestep_bias_cfg = {
        let strategy = timestep_bias::Strategy::parse(&args.timestep_bias_strategy)
            .map_err(|e| anyhow::anyhow!("--timestep-bias-strategy: {e}"))?;
        let cfg = timestep_bias::BiasConfig {
            strategy,
            multiplier: args.timestep_bias_multiplier,
            range_min: args.timestep_bias_range_min,
            range_max: args.timestep_bias_range_max,
        };
        if strategy != timestep_bias::Strategy::None {
            log::info!(
                "[timestep-bias] strategy={} multiplier={} range=[{}, {}]",
                strategy.as_str(),
                cfg.multiplier,
                cfg.range_min,
                cfg.range_max
            );
        }
        cfg
    };

    if let Some(resume_path) = args.resume_lora.as_ref() {
        log::info!("Resuming LoRA weights only (no optimizer state) from {}", resume_path.display());
        model.load_weights(&resume_path.to_string_lossy())?;
    }

    let mut start_step: usize = 0;
    if let Some(resume_path) = args.resume_full.as_ref() {
        log::info!("Full-resume from {}", resume_path.display());
        let loaded = checkpoint::load_full(resume_path, &device)?;
        let named = model.named_parameters();
        checkpoint::apply_lora_weights(&loaded, &named)?;
        checkpoint::apply_to_optimizer(&loaded, &mut opt, &named, args.rank, args.lora_alpha as f32)?;
        start_step = loaded.header.step as usize;
        if start_step >= args.steps {
            log::warn!("Resumed step ({start_step}) >= --steps ({}) — nothing to do.", args.steps);
            return Ok(());
        }
        log::info!("Continuing from step {start_step}/{}", args.steps);
    }

    let save_mode_full = match args.save_mode.as_str() {
        "full" => true,
        "weights" => false,
        other => anyhow::bail!("--save-mode must be `full` or `weights`, got `{other}`"),
    };

    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!("Found {} cached samples", cache_files.len());

    // Phase 2: validation harness — held-out cache + cadence. None at default.
    let validation_loop: Option<ValidationLoop> = if let (Some(dir), n) =
        (args.validation_dataset_dir.as_ref(), args.validation_every_steps)
    {
        if n > 0 {
            let v = ValidationLoop::new(dir, n)?;
            log::info!(
                "[validation] {} held-out samples, every {} steps",
                v.len(),
                n
            );
            Some(v)
        } else {
            None
        }
    } else {
        None
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    // ── Periodic-sample setup ────────────────────────────────────────────
    let periodic = args.sample_every > 0;
    let (sample_cap, sample_uncond, sample_pooled, sample_neg_pooled, sample_vae_path) = if periodic {
        let clip_l_p = args.sample_clip_l.as_ref().ok_or_else(|| anyhow::anyhow!(
            "--sample-every > 0 requires --sample-clip-l"))?;
        let clip_g_p = args.sample_clip_g.as_ref().ok_or_else(|| anyhow::anyhow!(
            "--sample-every > 0 requires --sample-clip-g"))?;
        let t5_p = args.sample_t5.as_ref().ok_or_else(|| anyhow::anyhow!(
            "--sample-every > 0 requires --sample-t5"))?;
        let tok_l_p = args.sample_clip_l_tokenizer.as_ref().ok_or_else(|| anyhow::anyhow!(
            "--sample-every > 0 requires --sample-clip-l-tokenizer"))?;
        let tok_g_p = args.sample_clip_g_tokenizer.as_ref().ok_or_else(|| anyhow::anyhow!(
            "--sample-every > 0 requires --sample-clip-g-tokenizer"))?;
        let tok_t5_p = args.sample_t5_tokenizer.as_ref().ok_or_else(|| anyhow::anyhow!(
            "--sample-every > 0 requires --sample-t5-tokenizer"))?;
        let vae_p = args.sample_vae.as_ref().cloned().unwrap_or_else(|| args.transformer.clone());
        log::info!("[sample-setup] loading text encoders to encode prompt once...");
        let clip_l_w = load_one_or_dir(clip_l_p, &device)?;
        let clip_l = ClipEncoder::new(clip_l_w, ClipConfig::default(), device.clone());
        let clip_g_w = load_one_or_dir(clip_g_p, &device)?;
        let clip_g = ClipGEncoder::new(clip_g_w, device.clone());
        let mut t5 = T5Encoder::load(t5_p.to_str().ok_or_else(|| anyhow::anyhow!("t5 path utf8"))?, &device)?;
        let tok_l = tokenizers::Tokenizer::from_file(tok_l_p)
            .map_err(|e| anyhow::anyhow!("clip_l tok: {e}"))?;
        let tok_g = tokenizers::Tokenizer::from_file(tok_g_p)
            .map_err(|e| anyhow::anyhow!("clip_g tok: {e}"))?;
        let tok_t5 = tokenizers::Tokenizer::from_file(tok_t5_p)
            .map_err(|e| anyhow::anyhow!("t5 tok: {e}"))?;
        let (cap, pool) = encode_sd3_prompt(
            &args.sample_prompt, &clip_l, &clip_g, &mut t5,
            &tok_l, &tok_g, &tok_t5, args.sample_t5_max_len, &device,
        )?;
        let (unc, npool) = encode_sd3_prompt(
            &args.sample_neg_prompt, &clip_l, &clip_g, &mut t5,
            &tok_l, &tok_g, &tok_t5, args.sample_t5_max_len, &device,
        )?;
        log::info!("[sample-setup] cap={:?} pooled={:?}", cap.dims(), pool.dims());
        drop(clip_l); drop(clip_g); drop(t5);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        log::info!("[sample-setup] text encoders dropped; periodic sample enabled (every {} steps)",
            args.sample_every);
        (Some(cap), Some(unc), Some(pool), Some(npool), Some(vae_p))
    } else {
        (None, None, None, None, None)
    };

    // Step-0 baseline sample (LoRA at zero init = base output).
    if periodic {
        let cap = sample_cap.as_ref().unwrap();
        let unc = sample_uncond.as_ref().unwrap();
        let pool = sample_pooled.as_ref().unwrap();
        let npool = sample_neg_pooled.as_ref().unwrap();
        let vae_p = sample_vae_path.as_ref().unwrap();
        let out_path = args.output_dir.join("sample_step0_base.png");
        log::info!("[sample step=0] BASELINE → {}", out_path.display());
        if let Err(e) = inline_sample(
            &mut model, cap, pool, unc, npool, vae_p, &out_path,
            args.sample_size, args.sample_steps, args.sample_cfg,
            args.sample_shift, args.sample_seed, &device,
        ) {
            log::warn!("[sample step=0] failed: {e}");
        }
    }

    let board = BoardWriter::open(
        &args.output_dir,
        BoardWriter::new_session_id(),
        if start_step > 0 { Some(start_step as u64) } else { None },
    ).map_err(|e| log::warn!("board.db open failed: {e}")).ok();
    if let Some(b) = &board {
        log::info!("SerenityBoard: writing scalars to {}", b.db_path.display());
    }
    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;

    let sched: LrScheduler = lr_schedule::parse_cli_scheduler(&args.lr_scheduler);
    for step in start_step..args.steps {
        let bs = args.batch_size.max(1);

        let mut latents = Vec::with_capacity(bs);
        let mut texts = Vec::with_capacity(bs);
        let mut pooleds = Vec::with_capacity(bs);
        for b in 0..bs {
            let cache_idx = (step * bs + b) % cache_files.len();
            let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;
            let l = sample.get("latent")
                .ok_or_else(|| anyhow::anyhow!("cached sample {cache_idx} missing 'latent'"))?
                .to_dtype(DType::BF16)?;
            let t = sample.get("text_embedding")
                .ok_or_else(|| anyhow::anyhow!("cached sample {cache_idx} missing 'text_embedding'"))?
                .to_dtype(DType::BF16)?;
            // HARD: combined CLIP+T5 sequence must be exactly 154 tokens
            // (TXT_PAD_LEN*2). Reject pre-2026-05-08 caches built with the
            // old --t5-max-len=256 default (seq=333), which would otherwise
            // silently feed off-distribution context into the MMDiT.
            let t_dims = t.dims();
            if t_dims.len() != 3 || t_dims[1] != 2 * TXT_PAD_LEN || t_dims[2] != 4096 {
                anyhow::bail!(
                    "cache {} has text_embedding shape {:?} but expected [B, {}, 4096] (TXT_PAD_LEN={}). Re-run prepare_sd35 with current code.",
                    cache_files[cache_idx].display(), t_dims, 2 * TXT_PAD_LEN, TXT_PAD_LEN
                );
            }
            let p = sample.get("pooled")
                .ok_or_else(|| anyhow::anyhow!("cached sample {cache_idx} missing 'pooled'"))?
                .to_dtype(DType::BF16)?;
            let p_dims = p.dims();
            if p_dims.len() != 2 || p_dims[1] != 2048 {
                anyhow::bail!(
                    "cache {} has pooled shape {:?} but expected [B, 2048].",
                    cache_files[cache_idx].display(), p_dims
                );
            }
            // HIGH-1 fix: OT-style 3 INDEPENDENT per-encoder Bernoullis
            // (`StableDiffusion3Model.py:397-415`). The cached combined tensors
            // are sliced and partially zeroed in place; no `--null-text-cache`
            // dependency for the encoder-mask path. The legacy single-coin
            // null-cache swap is preserved as a fallback when --null-text-cache
            // is provided AND --caption-dropout-probability > 0: we keep the
            // existing behaviour for users who had it wired previously, and
            // apply the new per-encoder draws on TOP for OT parity.
            let (t, p) = if effective_caption_dropout_prob > 0.0 {
                use rand::Rng;
                let p_drop = effective_caption_dropout_prob;
                let drop_l = rng.r#gen::<f32>() < p_drop;
                let drop_g = rng.r#gen::<f32>() < p_drop;
                let drop_t5 = rng.r#gen::<f32>() < p_drop;
                // If null-text-cache is provided AND all three encoders draw,
                // use the cached null pair (cheaper than zeroing all slices).
                if drop_l && drop_g && drop_t5 {
                    if let Some((ref nt, ref np)) = null_text {
                        (nt.clone(), np.clone())
                    } else {
                        apply_per_encoder_dropout(&t, &p, true, true, true, &device)?
                    }
                } else {
                    apply_per_encoder_dropout(&t, &p, drop_l, drop_g, drop_t5, &device)?
                }
            } else {
                (t, p)
            };
            latents.push(l); texts.push(t); pooleds.push(p);
        }
        let latent = if bs == 1 { latents.into_iter().next().unwrap() }
            else { Tensor::cat(&latents.iter().collect::<Vec<_>>(), 0)? };
        let text = if bs == 1 { texts.into_iter().next().unwrap() }
            else { Tensor::cat(&texts.iter().collect::<Vec<_>>(), 0)? };
        let pooled = if bs == 1 { pooleds.into_iter().next().unwrap() }
            else { Tensor::cat(&pooleds.iter().collect::<Vec<_>>(), 0)? };

        // Per-batch-element timesteps + sigmas.
        let mut t_per_b: Vec<f32> = Vec::with_capacity(bs);
        let mut sigma_per_b: Vec<f32> = Vec::with_capacity(bs);
        let mut t_model_per_b: Vec<f32> = Vec::with_capacity(bs);
        for _ in 0..bs {
            let raw_t = sample_timestep_logit_normal(
                &mut rng,
                args.timestep_shift,
                config.min_noising_strength as f32,
                config.max_noising_strength as f32,
            );
            // Default-off: Strategy::None → returns raw_t unchanged.
            let t_continuous = timestep_bias::apply_bias(
                raw_t,
                NUM_TRAIN_TIMESTEPS as f32,
                &timestep_bias_cfg,
            );
            let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
            let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
            t_per_b.push(t_continuous);
            sigma_per_b.push(sigma);
            // Model expects timesteps in `[0, 1000)` (sinusoidal embedding
            // freqs are computed as `t * freq` with no scaling). upstream Python
            // passes the raw `timestep` int from `_get_timestep_discrete`,
            // i.e. `sigma_idx` itself (NOT `sigma_idx+1` — that off-by-one
            // is internal to `_add_noise_discrete`'s sigma table only).
            // Inference path equivalent: `vec![t_curr * 1000.0]` where
            // `t_curr ∈ [0, 1)` → same scale as `sigma_idx ∈ [0, 1000)`.
            t_model_per_b.push(sigma_idx as f32);
        }

        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        // Pyramid / multi-resolution noise (additive). Default-off when
        // iterations == 0 → byte-identical.
        let noise = noise_modifiers::maybe_apply_multires_noise(
            &noise,
            args.multires_noise_iterations,
            args.multires_noise_discount,
            &mut rng,
        )?;
        // Phase 1: noise modifiers (default-off). Offset noise is part of the
        // clean noise; input perturbation feeds model input only.
        let clean_noise = noise_modifiers::maybe_apply_offset_noise(
            &noise,
            config.offset_noise_weight as f32,
            args.noise_offset_probability,
            &mut rng,
        )?;
        let perturbed_noise = noise_modifiers::maybe_apply_input_perturbation(
            &clean_noise,
            args.gamma_input_perturbation,
            &mut rng,
        )?;
        let noisy = if bs == 1 {
            perturbed_noise.mul_scalar(sigma_per_b[0])?
                .add(&latent.mul_scalar(1.0 - sigma_per_b[0])?)?
        } else {
            let mut pieces = Vec::with_capacity(bs);
            for b in 0..bs {
                let n_b = perturbed_noise.narrow(0, b, 1)?;
                let l_b = latent.narrow(0, b, 1)?;
                let s = sigma_per_b[b];
                pieces.push(n_b.mul_scalar(s)?.add(&l_b.mul_scalar(1.0 - s)?)?);
            }
            Tensor::cat(&pieces.iter().collect::<Vec<_>>(), 0)?
        };
        let target = clean_noise.sub(&latent)?;
        // Keep timestep in F32. BF16 has 8-bit mantissa → loses 1-LSB precision
        // for integer values >256, which is most of the [0, 999] range. Train
        // ↔ inference timestep embedding parity breaks otherwise. The model's
        // sin/cos embedding promotes to F32 internally either way.
        let timestep = Tensor::from_vec(
            t_model_per_b.clone(), Shape::from_dims(&[bs]), device.clone(),
        )?.to_dtype(flame_core::DType::F32)?;

        let t_continuous = t_per_b[0];
        let sigma = sigma_per_b[0];
        let sigma_idx = (t_per_b[0].floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);

        if step == 0 {
            log::info!("step 0 | batch={} latent={:?} text={:?} pooled={:?} sigma[0]={:.4} (idx={})",
                bs, latent.dims(), text.dims(), pooled.dims(), sigma, sigma_idx);
        }

        // SD3.5 forward — `forward_inner` expects (noisy, timestep, context, pooled).
        let pred = model.forward_inner(&noisy, &timestep, &text, &pooled)?;
        if pred.dims() != target.dims() {
            anyhow::bail!("predicted velocity shape {:?} != target {:?}",
                pred.dims(), target.dims());
        }

        // F32 mean MSE — matches upstream Python (loss_weight_fn=CONSTANT, mse_strength=1.0).
        // Phase 1: combined loss + per-step weighting. Default-off invariant.
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        let raw_loss = loss_weight::combined_loss(
            &pred_f32,
            &target_f32,
            config.mse_strength as f32,
            config.mae_strength as f32,
            args.huber_strength,
        )?;
        let loss = loss_weight::apply_loss_weight(
            &raw_loss,
            sigma,
            config.loss_weight_fn,
            args.min_snr_gamma,
            true,
        )?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        let dbg_on = dbg::enabled("OT_DEBUG_STATS");
        if dbg_on {
            let p_st = dbg::stats(&pred);
            let t_st = dbg::stats(&target);
            eprintln!(
                "[OT_DEBUG step={:5}] t={:.2} loss(pre-scale)={:.4} | pred[mean={:+.3e} std={:.3e} max|·|={:.3e}] target[mean={:+.3e} std={:.3e} max|·|={:.3e}]",
                step, t_continuous, loss_val,
                p_st.mean, p_st.std, p_st.abs_max,
                t_st.mean, t_st.std, t_st.abs_max,
            );
        }

        let grads = loss.backward()?;

        // Grad-flow diagnostic.  Runs at step 1 — NOT step 0 — because every
        // LoRA-style algo (LoRA, LoCon, LoHa, LoKr) initializes one factor
        // at zero so `delta = factor_a @ factor_b = 0` at step 0.  Backward
        // through `delta * weight` then forces half the leaves to zero
        // gradient by mathematical construction.  Step 1 (after the first
        // optimizer step has driven the zero leaves off zero) is when the
        // assertion can distinguish "real bug" from "expected zero-init
        // pattern".  See flame-core/docs/TRAINER_DIAGNOSTICS.md.
        if step == 1 {
            let named = model.named_parameters();
            let named_refs: Vec<(&str, &flame_core::parameter::Parameter)> = named
                .iter()
                .map(|(n, p)| (n.as_str(), p))
                .collect();
            let report = flame_core::diagnostics::assert_grad_flow(&grads, &named_refs)?;
            if report.is_clean() {
                log::info!("[grad-flow] step 2 clean ({} params)", report.ok_count);
            } else {
                log::warn!("{}", report.summary());
            }
        }

        // Fusion Sprint Phase 5: device-resident global L2 norm — one D2H per step.
        let grad_refs: Vec<&flame_core::Tensor> = params
            .iter()
            .filter_map(|p| grads.get(p.id()))
            .collect();
        let total_norm = flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?
            .item()? as f32;
        if dbg_on {
            eprintln!("[OT_DEBUG step={:5}] grad_norm_pre_clip={:.4e}", step, total_norm);
        }
        let scale = if total_norm > CLIP_GRAD_NORM { CLIP_GRAD_NORM / total_norm } else { 1.0 };
        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                let g_scaled = if scale < 1.0 { g.mul_scalar(scale)? } else { g.clone() };
                param.set_grad(g_scaled)?;
            }
        }

        // Phase 5: dispatch LR per scheduler. Default Constant + warmup_steps=0
        // is byte-equivalent to prior fixed-LR behaviour.
        let cur_lr = lr_schedule::dispatch_lr(
            &sched,
            args.lr,
            step,
            args.steps,
            args.warmup_steps,
            args.lr_min_factor,
            args.lr_cycles,
        );
        {
            let _g = AutogradContext::no_grad();
            opt.set_lr(cur_lr);
            opt.step(&params)?;
            opt.zero_grad(&params);
            if let Some(ref mut e) = ema {
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1))?;
            }
        }
        AutogradContext::clear();

        let _ = total_loss;
        eridiffusion_core::training::progress::log_step(
            step, args.steps, cache_files.len(), args.batch_size.max(1),
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // Phase 2: validation eval pass (no_grad) every `validation_every_steps`.
        // step+1 because `step` here is 0-based; ValidationLoop::should_run
        // expects the 1-based completed-step number.
        if let Some(ref vloop) = validation_loop {
            if vloop.should_run(step + 1) {
                let mut sum = 0.0_f32;
                let mut count = 0_usize;
                for vfile in &vloop.cache_files {
                    let _g = AutogradContext::no_grad();
                    let sample = match flame_core::serialization::load_file(vfile, &device) {
                        Ok(s) => s,
                        Err(e) => {
                            log::warn!("[validation] load {} failed: {e}", vfile.display());
                            continue;
                        }
                    };
                    let v_lat = match sample.get("latent") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => {
                            log::warn!("[validation] {} missing latent", vfile.display());
                            continue;
                        }
                    };
                    let v_txt = match sample.get("text_embedding") {
                        Some(t) => {
                            let d = t.dims();
                            if d.len() != 3 || d[1] != 2 * TXT_PAD_LEN || d[2] != 4096 {
                                log::warn!(
                                    "[validation] {} has text_embedding shape {:?} but expected [B, {}, 4096]; skipping",
                                    vfile.display(), d, 2 * TXT_PAD_LEN
                                );
                                continue;
                            }
                            t.to_dtype(DType::BF16)?
                        }
                        None => {
                            log::warn!("[validation] {} missing text_embedding", vfile.display());
                            continue;
                        }
                    };
                    let v_pool = match sample.get("pooled") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => {
                            log::warn!("[validation] {} missing pooled", vfile.display());
                            continue;
                        }
                    };
                    // Sample timestep + noise identically to training. Validation
                    // uses its OWN run-side RNG so it does not perturb the
                    // training-side seeded sequence (byte invariance). SD3.5
                    // schedule respects --timestep-shift exactly like the
                    // training step.
                    // MED-5 fix: avoid the `SEED ^ (step+1)` collision class
                    // (when `step + 1 == SEED`, the seed becomes 0). Mix with a
                    // 64-bit golden-ratio prime instead.
                    let mut vrng = rand::rngs::StdRng::seed_from_u64(
                        SEED.wrapping_mul(0x9E3779B97F4A7C15)
                            .wrapping_add(step as u64 + 1),
                    );
                    let t_continuous = sample_timestep_logit_normal(
                        &mut vrng,
                        args.timestep_shift,
                        config.min_noising_strength as f32,
                        config.max_noising_strength as f32,
                    );
                    let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
                    let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
                    let v_noise = Tensor::randn(v_lat.shape().clone(), 0.0, 1.0, device.clone())?
                        .to_dtype(DType::BF16)?;
                    let v_noisy = v_noise.mul_scalar(sigma)?
                        .add(&v_lat.mul_scalar(1.0 - sigma)?)?;
                    let v_target = v_noise.sub(&v_lat)?;
                    // Match training: F32 timestep with raw sigma_idx scale
                    // (NOT sigma_idx+1; see comment at training-step assembly).
                    let v_t_model = sigma_idx as f32;
                    let v_timestep = Tensor::from_vec(
                        vec![v_t_model],
                        Shape::from_dims(&[v_lat.shape().dims()[0]]),
                        device.clone(),
                    )?.to_dtype(DType::F32)?;
                    let v_pred = match model.forward_inner(&v_noisy, &v_timestep, &v_txt, &v_pool) {
                        Ok(p) => p,
                        Err(e) => {
                            log::warn!("[validation] forward failed: {e}");
                            continue;
                        }
                    };
                    let v_loss = v_pred.to_dtype(DType::F32)?
                        .sub(&v_target.to_dtype(DType::F32)?)?
                        .square()?
                        .mean()?;
                    let v_loss_val = v_loss.to_vec()?[0];
                    if v_loss_val.is_finite() {
                        sum += v_loss_val;
                        count += 1;
                    }
                    AutogradContext::clear();
                }
                if count > 0 {
                    let val_avg = sum / count as f32;
                    log::info!(
                        "[validation step={}] loss/val = {:.4} ({} samples)",
                        step + 1,
                        val_avg,
                        count
                    );
                    if let Some(b) = &board {
                        b.log_scalar("loss/val", (step + 1) as u64, val_avg as f64);
                    }
                }
            }
        }

        let step_num = step + 1;

        // EMA swap: when --ema --ema-validation-swap, save and sample see
        // EMA-averaged weights. Backup is restored at the end of this block.
        let _save_now = args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps;
        let _sample_now = periodic && step_num % args.sample_every == 0 && step_num < args.steps;
        let ema_backup = if (_save_now || _sample_now) && args.ema_validation_swap {
            if let Some(ref e) = ema {
                let _g = AutogradContext::no_grad();
                Some(e.swap_with_live(&params)
                    .map_err(|err| anyhow::anyhow!("EMA swap_with_live (mid) failed: {err}"))?)
            } else {
                None
            }
        } else {
            None
        };

        // Save-only checkpoint.
        if args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps {
            let mid_ckpt = args.output_dir.join(format!("sd35_lora_step{step_num}.safetensors"));
            if save_mode_full {
                let header = CkptHeader::from_adamw(
                    "train_sd35", step_num as u64, &opt,
                    args.rank, args.lora_alpha as f32, SEED, String::new(),
                );
                let named = model.named_parameters();
                if let Err(e) = checkpoint::save_full(&mid_ckpt, &named, &opt, &header) {
                    log::warn!("[save step {step_num}] full save failed: {e}");
                }
            } else if let Err(e) = model.save_weights(&mid_ckpt.to_string_lossy()) {
                log::warn!("[save step {step_num}] failed: {e}");
            } else {
                log::info!("[save step {step_num}] {}", mid_ckpt.display());
            }
        }

        // Inline sample (independent of save).
        if periodic && step_num % args.sample_every == 0 && step_num < args.steps {
            let cap = sample_cap.as_ref().unwrap();
            let unc = sample_uncond.as_ref().unwrap();
            let pool = sample_pooled.as_ref().unwrap();
            let npool = sample_neg_pooled.as_ref().unwrap();
            let vae_p = sample_vae_path.as_ref().unwrap();
            let out_path = args.output_dir.join(format!("sample_step{step_num}.png"));
            log::info!("[sample step={step_num}] → {}", out_path.display());
            if let Err(e) = inline_sample(
                &mut model, cap, pool, unc, npool, vae_p, &out_path,
                args.sample_size, args.sample_steps, args.sample_cfg,
                args.sample_shift, args.sample_seed, &device,
            ) {
                log::warn!("[sample step={step_num}] failed: {e}");
            }
        }
        if let (Some(backup), Some(ref e)) = (ema_backup, &ema) {
            let _g = AutogradContext::no_grad();
            e.restore_swapped(&params, backup)
                .map_err(|err| anyhow::anyhow!("EMA restore_swapped (mid) failed: {err}"))?;
        }
    }

    // Final EMA swap before final save+sample. No restore — process exits.
    if args.ema_validation_swap {
        if let Some(ref e) = ema {
            let _g = AutogradContext::no_grad();
            let _ = e.swap_with_live(&params)
                .map_err(|err| anyhow::anyhow!("EMA swap_with_live (final) failed: {err}"))?;
            log::info!("[ema] swapped EMA shadow into live params for final save + sample");
        }
    }

    let trained = args.steps - start_step;
    let avg = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    log::info!("Training complete: {trained} new steps (total={}), avg loss={:.4}", args.steps, avg);
    if let Some(b) = &board { b.set_status("completed"); }

    let ckpt = args.output_dir.join(format!("sd35_lora_{}steps.safetensors", args.steps));
    if save_mode_full {
        let header = CkptHeader::from_adamw(
            "train_sd35", args.steps as u64, &opt,
            args.rank, args.lora_alpha as f32, SEED, String::new(),
        );
        let named = model.named_parameters();
        if let Err(e) = checkpoint::save_full(&ckpt, &named, &opt, &header) {
            log::warn!("save_full failed: {e}");
        } else {
            log::info!("Saved checkpoint to {}", ckpt.display());
        }
    } else if let Err(e) = model.save_weights(&ckpt.to_string_lossy()) {
        log::warn!("save_weights failed: {e}");
    } else {
        log::info!("Saved checkpoint to {}", ckpt.display());
    }

    if periodic {
        let cap = sample_cap.as_ref().unwrap();
        let unc = sample_uncond.as_ref().unwrap();
        let pool = sample_pooled.as_ref().unwrap();
        let npool = sample_neg_pooled.as_ref().unwrap();
        let vae_p = sample_vae_path.as_ref().unwrap();
        let out_path = args.output_dir.join(format!("sample_step{}.png", args.steps));
        log::info!("[sample FINAL step={}] → {}", args.steps, out_path.display());
        if let Err(e) = inline_sample(
            &mut model, cap, pool, unc, npool, vae_p, &out_path,
            args.sample_size, args.sample_steps, args.sample_cfg,
            args.sample_shift, args.sample_seed, &device,
        ) {
            log::warn!("[sample final] failed: {e}");
        }
    }

    Ok(())
}
