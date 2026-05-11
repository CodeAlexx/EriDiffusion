//! sample_u1 — SenseNova-U1-8B-MoT text-to-image sampler.
//!
//! Port of `inference-flame/src/bin/sensenova_u1_gen.rs` (non-think path)
//! adapted for the EDv2 trainer ecosystem. Optionally loads a LoRA bundle
//! saved by `train_u1 --lora-save-to` (upstream PEFT format).
//!
//! Pipeline:
//!   1. Build BPE tokenizer from `vocab.json` + `merges.txt` + `added_tokens.json`.
//!   2. Tokenize cond/uncond queries using U1's chat template.
//!   3. forward_und for both → KvCache.
//!   4. Init Gaussian noise image, scaled by `compute_noise_scale(grid_h, grid_w)`.
//!   5. Build timestep grid via `apply_time_schedule`.
//!   6. For each step: patchify → extract_feature_gen → +timestep/+noise_scale →
//!      forward_gen (cond + uncond) → fm_head_forward → CFG combine →
//!      Euler step on z → unpatchify.
//!   7. Denorm `x * 0.5 + 0.5`, save PNG.

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

use eridiffusion_core::models::sensenova_u1::{
    self, patchify, unpatchify, KvCache, SenseNovaU1, TimeOrScale,
};
use eridiffusion_core::models::sensenova_u1_lora as u1lora;

const SYSTEM_MESSAGE_FOR_GEN: &str = concat!(
    "You are an image generation and editing assistant that accurately understands and executes ",
    "user intent.\n\nYou support two modes:\n\n",
    "1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. ",
    "Put all reasoning inside the block using plain text. DO NOT include any image tags. ",
    "Keep it reasonable and directly useful for producing the final image.\n\n",
    "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n",
    "Task Types:\n\nA. Text-to-Image Generation:\n",
    "- Generate a high-quality image based on the user's description.\n",
    "- Ensure visual clarity, semantic consistency, and completeness.\n",
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n",
    "B. Image Editing:\n",
    "- Use the provided image(s) as input or reference for modification or transformation.\n",
    "- The result can be an edited image or a new image based on the reference(s).\n",
    "- Preserve all unspecified attributes unless explicitly changed.\n\n",
    "General Rules:\n",
    "- For any visible text in the image, follow the language specified for the rendered text in ",
    "the user's description, not the language of the prompt. If no language is specified, use the ",
    "user's input language."
);

#[derive(Parser)]
#[command(about = "SenseNova-U1-8B-MoT text-to-image sampler (non-think mode).")]
struct Args {
    /// Directory containing `model.safetensors.index.json` + shards +
    /// `vocab.json` + `merges.txt` + `added_tokens.json`.
    #[arg(long)]
    model_path: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value = "output.png")]
    output: PathBuf,

    /// Image height, must be divisible by `patch_size * merge_size = 32`.
    #[arg(long, default_value = "1024")]
    height: usize,

    /// Image width.
    #[arg(long, default_value = "1024")]
    width: usize,

    #[arg(long, default_value = "28")]
    steps: usize,

    #[arg(long, default_value = "4.0")]
    cfg_scale: f32,

    /// Time-schedule shift. Higher = more low-noise resolution detail.
    #[arg(long, default_value = "0.5")]
    timestep_shift: f32,

    #[arg(long, default_value = "42")]
    seed: u64,

    /// Optional LoRA adapter file (upstream PEFT safetensors). When set,
    /// adapters are attached after model load and applied at every q/k/v/o
    /// + mlp + fm_head linear that matches.
    #[arg(long)]
    lora: Option<PathBuf>,
}

fn build_tokenizer(weights_dir: &std::path::Path) -> Result<Tokenizer> {
    let vocab = weights_dir.join("vocab.json");
    let merges = weights_dir.join("merges.txt");
    let added = weights_dir.join("added_tokens.json");

    let bpe = BPE::from_file(
        vocab.to_str().context("vocab path not utf-8")?,
        merges.to_str().context("merges path not utf-8")?,
    )
    .build()
    .map_err(|e| anyhow!("BPE::build failed: {e}"))?;

    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tok.with_decoder(Some(ByteLevel::default()));

    let raw = std::fs::read_to_string(&added)
        .with_context(|| format!("read {}", added.display()))?;
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).context("added_tokens.json")?;
    let mut entries: Vec<(String, u64)> = map
        .into_iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k, id)))
        .collect();
    entries.sort_by_key(|(_, id)| *id);

    let base_size = tok.get_vocab_size(false) as u64;
    if let Some((_, first_id)) = entries.first() {
        if *first_id != base_size {
            return Err(anyhow!(
                "added_tokens.json starts at id {first_id} but base vocab size is {base_size}"
            ));
        }
    }

    let added_tokens: Vec<AddedToken> = entries
        .into_iter()
        .map(|(content, _)| AddedToken::from(content, true))
        .collect();
    tok.add_special_tokens(&added_tokens);

    let im_start = tok
        .token_to_id("<|im_start|>")
        .ok_or_else(|| anyhow!("<|im_start|> not in tokenizer"))?;
    if im_start != 151644 {
        return Err(anyhow!("<|im_start|> mapped to {im_start}, expected 151644"));
    }
    Ok(tok)
}

/// Build the T2I chat-template prompt.
fn build_t2i_query(system: &str, user: &str, append: &str) -> String {
    let mut q = String::new();
    if !system.is_empty() {
        q.push_str("<|im_start|>system\n");
        q.push_str(system);
        q.push_str("<|im_end|>\n");
    }
    q.push_str("<|im_start|>user\n");
    q.push_str(user);
    q.push_str("<|im_end|>\n");
    q.push_str("<|im_start|>assistant\n");
    q.push_str(append);
    q
}

fn encode_query(tok: &Tokenizer, query: &str) -> Result<Vec<i32>> {
    let enc = tok.encode(query, false).map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

fn make_noise_image(
    seed: u64,
    shape: &[usize],
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    use rand::{Rng, SeedableRng};
    let numel: usize = shape.iter().product();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(numel);
    for _ in 0..numel {
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen();
        data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    let t = Tensor::from_vec(data, Shape::from_dims(shape), device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}

fn save_png(image: &Tensor, path: &std::path::Path) -> Result<()> {
    let img_f32 = image.to_dtype(DType::F32)?;
    let dims = img_f32.shape().dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        return Err(anyhow!("save_png expects [1, 3, H, W], got {dims:?}"));
    }
    let (h, w) = (dims[2], dims[3]);
    let data = img_f32.to_vec()?;
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = data[c * h * w + y * w + x];
                let u = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                pixels[(y * w + x) * 3 + c] = u;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow!("RgbImage::from_raw failed"))?
        .save(path)?;
    Ok(())
}

fn forward_gen_for(
    model: &mut SenseNovaU1,
    image_embeds: &Tensor,
    text_len: usize,
    token_h: usize,
    token_w: usize,
    cache: &KvCache,
) -> Result<Tensor> {
    Ok(model.forward_gen(image_embeds, text_len, token_h, token_w, cache, None)?)
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // ---- 0) Device + tokenizer + model ----------------------------------
    let device = flame_core::CudaDevice::new(0)
        .map_err(|e| anyhow!("CudaDevice::new(0): {e}"))?;
    log::info!("[sample_u1] loading tokenizer from {}", args.model_path.display());
    let tok = build_tokenizer(&args.model_path)?;

    log::info!("[sample_u1] loading model");
    let t_load = std::time::Instant::now();
    let mut model = SenseNovaU1::load(&args.model_path, &device)?;
    log::info!("[sample_u1] model loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    // ---- 0b) Optional LoRA attach ---------------------------------------
    if let Some(lora_path) = args.lora.as_ref() {
        log::info!("[sample_u1] loading LoRA from {}", lora_path.display());
        let adapters = u1lora::load_adapters(lora_path, device.clone())?;
        log::info!("[sample_u1] attaching {} LoRA adapters", adapters.len());
        model.attach_lora_adapters(adapters);
    }

    // ---- 1) Geometry checks --------------------------------------------
    let cfg = model.config().clone();
    let patch = cfg.patch_size;
    let merge = cfg.merge_size();
    let token_p = patch * merge;
    if args.width % token_p != 0 || args.height % token_p != 0 {
        anyhow::bail!(
            "--width {} / --height {} must be divisible by patch_size*merge_size = {token_p}",
            args.width, args.height,
        );
    }
    let grid_h = args.height / patch;
    let grid_w = args.width / patch;
    let token_h = grid_h / merge;
    let token_w = grid_w / merge;
    let l_tokens = token_h * token_w;
    let b: usize = 1;

    // ---- 2) Tokenize cond/uncond ---------------------------------------
    // Non-think mode: append <think>\n\n</think>\n\n<img> so the model
    // immediately produces an empty think block then the <img> sentinel.
    let cond_query = build_t2i_query(
        SYSTEM_MESSAGE_FOR_GEN,
        &args.prompt,
        "<think>\n\n</think>\n\n<img>",
    );
    let uncond_query = build_t2i_query("", "", "<img>");
    let cond_ids = encode_query(&tok, &cond_query)?;
    let uncond_ids = encode_query(&tok, &uncond_query)?;
    log::info!(
        "[sample_u1] cond tokens={}  uncond tokens={}",
        cond_ids.len(), uncond_ids.len(),
    );

    // ---- 3) Prefix forwards --------------------------------------------
    let t_prefix = std::time::Instant::now();
    let (cond_cache, _cond_last) = model.forward_und(&cond_ids)?;
    let (uncond_cache, _) = model.forward_und(&uncond_ids)?;
    log::info!("[sample_u1] prefix forward: {:.2}s", t_prefix.elapsed().as_secs_f32());

    // ---- 4) Init noise --------------------------------------------------
    let noise_scale = model.compute_noise_scale(grid_h, grid_w);
    log::info!(
        "[sample_u1] grid={}x{}, tokens={}x{} (L={}), noise_scale={:.4}",
        grid_h, grid_w, token_h, token_w, l_tokens, noise_scale,
    );
    let mut img = make_noise_image(
        args.seed,
        &[b, 3, args.height, args.width],
        &device,
    )?;
    img = img.mul_scalar(noise_scale)?;

    // ---- 5) Build timestep grid ----------------------------------------
    let mut t_uniform: Vec<f32> = (0..=args.steps)
        .map(|i| i as f32 / args.steps as f32)
        .collect();
    t_uniform = model.apply_time_schedule(&t_uniform, l_tokens, args.timestep_shift);

    // ---- 6) Step loop --------------------------------------------------
    let s_norm = noise_scale / cfg.noise_scale_max_value;
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    for step in 0..args.steps {
        let t = t_uniform[step];
        let t_next = t_uniform[step + 1];
        let t_step = std::time::Instant::now();

        let z = patchify(&img, patch * merge, false)?;
        let pixel_values = patchify(&img, patch, true)?;
        let pixel_flat = pixel_values.reshape(&[b * grid_h * grid_w, 3 * patch * patch])?;

        let mut image_embeds = model.extract_feature_gen(&pixel_flat, grid_h, grid_w)?;
        let t_vec = vec![t; b * l_tokens];
        let t_tensor = Tensor::from_vec(t_vec, Shape::from_dims(&[b * l_tokens]), device.clone())?
            .to_dtype(DType::BF16)?;
        let t_emb = model
            .time_or_scale_embed(&t_tensor, TimeOrScale::Timestep)?
            .reshape(&[b, l_tokens, cfg.hidden_size])?;
        let mut additive = t_emb;
        if cfg.add_noise_scale_embedding {
            let s_tensor = Tensor::from_vec(
                vec![s_norm; b * l_tokens],
                Shape::from_dims(&[b * l_tokens]),
                device.clone(),
            )?
            .to_dtype(DType::BF16)?;
            let s_emb = model
                .time_or_scale_embed(&s_tensor, TimeOrScale::NoiseScale)?
                .reshape(&[b, l_tokens, cfg.hidden_size])?;
            additive = additive.add(&s_emb)?;
        }
        image_embeds = image_embeds.add(&additive)?;

        let h_cond = forward_gen_for(
            &mut model, &image_embeds, cond_cache.next_t_index, token_h, token_w, &cond_cache,
        )?;
        let h_uncond = forward_gen_for(
            &mut model, &image_embeds, uncond_cache.next_t_index, token_h, token_w, &uncond_cache,
        )?;
        let x_cond = model.fm_head_forward(&h_cond)?;
        let x_uncond = model.fm_head_forward(&h_uncond)?;
        let denom = (1.0_f32 - t).max(cfg.t_eps);
        let inv_denom = 1.0 / denom;
        let v_cond = x_cond.sub(&z)?.mul_scalar(inv_denom)?;
        let v_uncond = x_uncond.sub(&z)?.mul_scalar(inv_denom)?;
        let v_diff = v_cond.sub(&v_uncond)?;
        let v = v_uncond.add(&v_diff.mul_scalar(args.cfg_scale)?)?;
        let z_next = z.add(&v.mul_scalar(t_next - t)?)?;
        img = unpatchify(&z_next, patch * merge, args.height, args.width)?;

        log::info!(
            "[sample_u1] step {:>3}/{}  t={:.4}→{:.4}  {:.2}s",
            step + 1, args.steps, t, t_next, t_step.elapsed().as_secs_f32(),
        );
    }
    drop(_no_grad);

    // ---- 7) Denorm + save ----------------------------------------------
    let final_img = img.mul_scalar(0.5)?.add_scalar(0.5)?;
    save_png(&final_img, &args.output)?;
    log::info!("[sample_u1] saved → {}", args.output.display());

    // Reference unused-import guard.
    let _ = sensenova_u1::patchify;
    Ok(())
}
