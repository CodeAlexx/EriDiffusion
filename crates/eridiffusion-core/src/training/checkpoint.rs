//! Full training-state checkpoint: LoRA weights + optimizer state + step
//! counter + metadata, all in a single safetensors file.
//!
//! Why single-file: atomic save/replace, single command-line path on resume,
//! no `.optstate` orphans.
//!
//! Optimizer state lives under reserved key prefixes:
//!   `__opt__/adamw/m/<canonical_name>`
//!   `__opt__/adamw/v/<canonical_name>`
//!
//! Header (step, optimizer kind, hyperparams, rank, alpha, rng) is stored in
//! the safetensors-standard `__metadata__` map under the JSON-encoded key
//! `__eridiffusion_ckpt__`.
//!
//! A weights-only legacy file (no metadata) loads via `--resume-lora` —
//! `load_full` refuses such files with a clear error message.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{adam::AdamW, parameter::Parameter, serialization, Tensor};
use serde::{Deserialize, Serialize};

use crate::{EriDiffusionError, Result};

pub const CKPT_HEADER_KEY: &str = "__eridiffusion_ckpt__";
pub const OPT_PREFIX: &str = "__opt__/";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CkptHeader {
    pub format_version: u32,
    pub trainer: String,
    pub step: u64,
    pub optimizer: String,
    pub adam_t: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub rank: usize,
    pub alpha: f32,
    pub rng_state: u64,
    pub config_hash: String,
}

impl CkptHeader {
    pub fn from_adamw(
        trainer: &str,
        step: u64,
        optimizer: &AdamW,
        rank: usize,
        alpha: f32,
        rng_state: u64,
        config_hash: String,
    ) -> Self {
        Self {
            format_version: 1,
            trainer: trainer.to_string(),
            step,
            optimizer: "adamw".to_string(),
            adam_t: optimizer.t(),
            lr: optimizer.lr(),
            beta1: optimizer.beta1(),
            beta2: optimizer.beta2(),
            eps: optimizer.eps(),
            weight_decay: optimizer.weight_decay(),
            rank,
            alpha,
            rng_state,
            config_hash,
        }
    }
}

/// Save a full checkpoint.
///
/// `named_params` MUST use the same canonical naming the trainer's
/// weights-only save uses (e.g. for Z-Image:
/// `diffusion_model.layers.{i}.attention.to_q.lora_A.weight`). On resume,
/// the optimizer state is matched by these names against live Parameters.
pub fn save_full(
    path: &Path,
    named_params: &[(String, Parameter)],
    optimizer: &AdamW,
    header: &CkptHeader,
) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut state_count = 0usize;

    for (name, param) in named_params {
        if tensors.contains_key(name) {
            return Err(EriDiffusionError::Training(format!(
                "ckpt save: duplicate canonical name {name}"
            )));
        }
        tensors.insert(name.clone(), param.tensor()?);
        if let Some((m, v)) = optimizer.state_for(param) {
            tensors.insert(format!("{OPT_PREFIX}adamw/m/{name}"), m);
            tensors.insert(format!("{OPT_PREFIX}adamw/v/{name}"), v);
            state_count += 1;
        }
    }

    let header_json = serde_json::to_string(header)?;
    let mut metadata: HashMap<String, String> = HashMap::new();
    metadata.insert(CKPT_HEADER_KEY.to_string(), header_json);

    crate::training::save_direct::save_tensors_with_metadata_direct(&tensors, &metadata, path)
        .map_err(|e| EriDiffusionError::Training(format!("ckpt save: {e}")))?;

    log::info!(
        "[ckpt save] {} | step={} | {} params | {}/{} with optimizer state",
        path.display(),
        header.step,
        named_params.len(),
        state_count,
        named_params.len(),
    );
    Ok(())
}

pub struct LoadedCkpt {
    pub header: CkptHeader,
    pub lora_tensors: HashMap<String, Tensor>,
    pub opt_m: HashMap<String, Tensor>,
    pub opt_v: HashMap<String, Tensor>,
}

pub fn load_full(path: &Path, device: &Arc<CudaDevice>) -> Result<LoadedCkpt> {
    let (tensors, metadata) = serialization::load_tensors_with_metadata(path, device.clone())
        .map_err(|e| EriDiffusionError::Training(format!("ckpt load: {e}")))?;

    let header_json = metadata.get(CKPT_HEADER_KEY).ok_or_else(|| {
        EriDiffusionError::Training(format!(
            "{} has no `{}` metadata — this looks like a weights-only safetensors. \
             Use --resume-lora for weights-only resume (optimizer + step counter restart fresh).",
            path.display(),
            CKPT_HEADER_KEY,
        ))
    })?;
    let header: CkptHeader = serde_json::from_str(header_json)?;

    let m_prefix = format!("{OPT_PREFIX}adamw/m/");
    let v_prefix = format!("{OPT_PREFIX}adamw/v/");
    let mut lora = HashMap::new();
    let mut opt_m = HashMap::new();
    let mut opt_v = HashMap::new();
    for (key, t) in tensors {
        if let Some(name) = key.strip_prefix(&m_prefix) {
            opt_m.insert(name.to_string(), t);
        } else if let Some(name) = key.strip_prefix(&v_prefix) {
            opt_v.insert(name.to_string(), t);
        } else {
            lora.insert(key, t);
        }
    }
    Ok(LoadedCkpt { header, lora_tensors: lora, opt_m, opt_v })
}

/// Apply a loaded checkpoint to live state.
///  - Validates optimizer kind + rank/alpha match (refuses on mismatch).
///  - Sets AdamW step counter (`adam_t`).
///  - Pairs each `(name, &Parameter)` with saved m/v by name; missing
///    pairs warn but don't fail (live param will start with fresh state,
///    matching first-step initialization).
///  - LR/weight_decay are NOT restored from the ckpt — caller's CLI flags
///    win, but a mismatch is logged so the user knows.
pub fn apply_to_optimizer(
    loaded: &LoadedCkpt,
    optimizer: &mut AdamW,
    named_params: &[(String, Parameter)],
    expected_rank: usize,
    expected_alpha: f32,
) -> Result<()> {
    if loaded.header.optimizer != "adamw" {
        return Err(EriDiffusionError::Training(format!(
            "ckpt optimizer is `{}` but trainer uses adamw",
            loaded.header.optimizer
        )));
    }
    if loaded.header.rank != expected_rank {
        return Err(EriDiffusionError::Training(format!(
            "ckpt rank={} but trainer rank={} — LoRA shapes are incompatible, refusing",
            loaded.header.rank, expected_rank
        )));
    }
    if (loaded.header.alpha - expected_alpha).abs() > 1e-6 {
        return Err(EriDiffusionError::Training(format!(
            "ckpt alpha={} but trainer alpha={} — scale would diverge silently, refusing",
            loaded.header.alpha, expected_alpha
        )));
    }

    let live_lr = optimizer.lr();
    if (live_lr - loaded.header.lr).abs() > 1e-9 {
        log::warn!(
            "[resume] LR differs: ckpt={} live={} (live wins; ckpt LR is informational)",
            loaded.header.lr, live_lr
        );
    }
    let live_wd = optimizer.weight_decay();
    if (live_wd - loaded.header.weight_decay).abs() > 1e-9 {
        log::warn!(
            "[resume] weight_decay differs: ckpt={} live={} (live wins)",
            loaded.header.weight_decay, live_wd
        );
    }

    optimizer.set_t(loaded.header.adam_t);

    let mut applied = 0usize;
    let mut missing = 0usize;
    for (name, param) in named_params {
        match (loaded.opt_m.get(name), loaded.opt_v.get(name)) {
            (Some(m), Some(v)) => {
                optimizer.set_state(param, m.clone(), v.clone());
                applied += 1;
            }
            _ => {
                missing += 1;
            }
        }
    }
    log::info!(
        "[resume] AdamW state applied: {applied}/{} params | {missing} missing | adam_t={} | step={}",
        named_params.len(),
        loaded.header.adam_t,
        loaded.header.step,
    );
    Ok(())
}

/// Apply LoRA weights from a checkpoint into a name→Parameter mapping.
/// Names not found in the checkpoint are left unchanged (logged as warn).
pub fn apply_lora_weights(
    loaded: &LoadedCkpt,
    named_params: &[(String, Parameter)],
) -> Result<()> {
    use flame_core::DType;
    let mut applied = 0usize;
    let mut missing = 0usize;
    for (name, param) in named_params {
        match loaded.lora_tensors.get(name) {
            Some(t) => {
                let live = param.tensor()?;
                let target_dtype = live.dtype();
                let cast = if t.dtype() == target_dtype {
                    t.clone()
                } else {
                    t.to_dtype(target_dtype)?
                };
                // Match dtype-cast + requires_grad of the original Parameter.
                let _ = target_dtype;
                let with_grad = if cast.dtype() == DType::F32 || cast.dtype() == DType::BF16 {
                    cast.requires_grad_(true)
                } else {
                    cast
                };
                param.set_data(with_grad)?;
                applied += 1;
            }
            None => {
                log::warn!("[resume] no saved tensor for `{name}`, leaving live weights");
                missing += 1;
            }
        }
    }
    log::info!("[resume] LoRA weights applied: {applied}/{} ({missing} missing)", named_params.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// Diffusers/PEFT → Kohya (ComfyUI-compatible) LoRA key converter
// ---------------------------------------------------------------------------
//
// Port of SimpleTuner PR #2704 `convert_diffusers_to_comfyui_sd_lora`
// (simpletuner/helpers/training/lora_format.py). ComfyUI's stock SDXL/SD1.x
// LoRA loader expects Kohya-style keys: `lora_unet_<underscored_path>.lora_down.weight`,
// `.lora_up.weight`, and a per-module `.alpha` scalar.
//
// Original Python triggers on diffusers component prefixes (`unet.`,
// `text_encoder.`, `text_encoder_2.`) and accepts both PEFT
// (`.lora_A.weight` / `.lora_B.weight`) and old-style
// (`.lora.down.weight` / `.lora.up.weight`) suffixes.
//
// EDv2-specific note: the SDXL trainer's `named_parameters()` already emits
// LDM-style internal paths (`input_blocks.X.1.transformer_blocks.Y.attnZ.to_q`),
// not diffusers paths (`unet.down_blocks.X.attentions.Y...`). To keep the
// porter honest about the source code we're porting, the converter accepts
// an explicit `component_prefix` argument: callers wrap their map with
// `unet.` (or another component) before invoking, so the inner logic mirrors
// the SimpleTuner Python verbatim. `convert_sdxl_unet_to_kohya` is the
// SDXL-trainer-specific helper that does the wrapping.

/// Map a diffusers component prefix to its Kohya prefix.
/// Mirrors SimpleTuner `_kohya_component_prefix` (sdxl=true).
fn kohya_component_prefix(component_prefix: &str, sdxl: bool) -> Option<&'static str> {
    let component = component_prefix.strip_suffix('.').unwrap_or(component_prefix);
    match component {
        "unet" => Some("lora_unet"),
        "text_encoder" => Some(if sdxl { "lora_te1" } else { "lora_te" }),
        "text_encoder_2" if sdxl => Some("lora_te2"),
        _ => None,
    }
}

/// Build the Kohya module key for a diffusers module key.
/// Mirrors SimpleTuner `_kohya_module_key`.
fn kohya_module_key(module_key: &str, component_prefix: &str, sdxl: bool) -> Option<String> {
    let kohya_prefix = kohya_component_prefix(component_prefix, sdxl)?;
    let module_path = module_key.strip_prefix(component_prefix)?;
    let module_path = module_path.replace(".processor.", ".");
    Some(format!("{kohya_prefix}_{}", module_path.replace('.', "_")))
}

/// Suffix rewrite map: PEFT/old → Kohya.
fn suffix_map_lookup(key: &str) -> Option<(&'static str, &'static str)> {
    // (matched, kohya_replacement)
    const MAP: &[(&str, &str)] = &[
        (".lora.down.weight", ".lora_down.weight"),
        (".lora.up.weight", ".lora_up.weight"),
        (".lora_A.weight", ".lora_down.weight"),
        (".lora_B.weight", ".lora_up.weight"),
    ];
    for (s, repl) in MAP {
        if key.ends_with(s) {
            return Some((s, repl));
        }
    }
    None
}

/// Verbatim port of SimpleTuner `convert_diffusers_to_comfyui_sd_lora`.
///
/// Inputs whose key doesn't match a known component prefix or LoRA suffix
/// pass through unchanged.
///
/// `lora_alpha` is broadcast to every unique kohya module that emitted a
/// `.lora_down.weight`. If `None`, no `.alpha` entries are synthesized
/// (mirrors Python's None branch from `_resolve_alpha_for_module`).
pub fn convert_diffusers_to_comfyui_sd_lora(
    state_dict: &HashMap<String, Tensor>,
    lora_alpha: Option<f32>,
    sdxl: bool,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    use flame_core::{DType, Shape};
    let prefixes = ["unet.", "text_encoder.", "text_encoder_2."];
    let mut converted: HashMap<String, Tensor> = HashMap::new();
    let mut alpha_keys: Vec<String> = Vec::new();
    let mut alpha_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (key, weight) in state_dict {
        let component_prefix = prefixes.iter().find(|p| key.starts_with(*p)).copied();
        let component_prefix = match component_prefix {
            Some(p) => p,
            None => {
                converted.insert(key.clone(), weight.clone());
                continue;
            }
        };
        let matched = suffix_map_lookup(key);
        let (matched_suffix, kohya_suffix) = match matched {
            Some(m) => m,
            None => {
                converted.insert(key.clone(), weight.clone());
                continue;
            }
        };
        let module_key = &key[..key.len() - matched_suffix.len()];
        let kohya_key = match kohya_module_key(module_key, component_prefix, sdxl) {
            Some(k) => k,
            None => {
                converted.insert(key.clone(), weight.clone());
                continue;
            }
        };
        let new_key = format!("{kohya_key}{kohya_suffix}");
        converted.insert(new_key, weight.clone());
        if kohya_suffix == ".lora_down.weight" && alpha_seen.insert(kohya_key.clone()) {
            alpha_keys.push(kohya_key);
        }
    }

    if let Some(alpha) = lora_alpha {
        for k in alpha_keys {
            let t = Tensor::from_vec(vec![alpha], Shape::from_dims(&[]), device.clone())
                .map_err(|e| EriDiffusionError::Training(format!("alpha tensor: {e}")))?
                .to_dtype(DType::F32)
                .map_err(|e| EriDiffusionError::Training(format!("alpha cast: {e}")))?;
            converted.insert(format!("{k}.alpha"), t);
        }
    }
    Ok(converted)
}

/// SDXL-trainer-specific wrapper: EDv2's SDXL trainer emits LDM-style keys
/// (`input_blocks.X.1.transformer_blocks.Y.attnZ.to_q.lora_A.weight`) with
/// no `unet.` prefix. To run the verbatim SimpleTuner port we synthesize the
/// `unet.` prefix on weight keys, drop the existing `.alpha` companions
/// (the converter regenerates them), and run the converter.
///
/// Optimizer-state keys (anything starting with `__opt__/`) and the
/// `__eridiffusion_ckpt__` metadata are NOT touched — those belong to the
/// resume path, not ComfyUI. The caller passes only the LoRA-weight subset.
pub fn convert_sdxl_unet_to_kohya(
    lora_state: &HashMap<String, Tensor>,
    lora_alpha: f32,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let mut wrapped: HashMap<String, Tensor> = HashMap::with_capacity(lora_state.len());
    for (k, v) in lora_state {
        // Skip per-module `.alpha` companions written by save_weights —
        // the converter regenerates them post-suffix-rewrite.
        if k.ends_with(".alpha") {
            continue;
        }
        wrapped.insert(format!("unet.{k}"), v.clone());
    }
    convert_diffusers_to_comfyui_sd_lora(&wrapped, Some(lora_alpha), /*sdxl=*/ true, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::{DType, Shape};

    /// Round-trip: build a tiny LoRA-style parameter set + AdamW state,
    /// save_full → load_full, assert every tensor and the header recover
    /// byte-exact.
    #[test]
    fn full_ckpt_round_trips() -> Result<()> {
        let device = flame_core::global_cuda_device();
        let tmp = std::env::temp_dir().join("eridiffusion_ckpt_roundtrip.safetensors");
        let _ = std::fs::remove_file(&tmp);

        // Build 4 fake LoRA-style F32 parameters.
        let mut named: Vec<(String, Parameter)> = Vec::new();
        for i in 0..2 {
            let a = Tensor::from_vec(
                (0..32).map(|j| (i * 32 + j) as f32 * 0.0123).collect(),
                Shape::from_dims(&[4, 8]), device.clone()
            )?.requires_grad_(true);
            let b = Tensor::from_vec(
                (0..32).map(|j| (i * 32 + j) as f32 * -0.045).collect(),
                Shape::from_dims(&[8, 4]), device.clone()
            )?.requires_grad_(true);
            named.push((format!("layers.{i}.lora_A.weight"), Parameter::new(a)));
            named.push((format!("layers.{i}.lora_B.weight"), Parameter::new(b)));
        }

        // Build AdamW + simulate state by injecting m/v directly.
        let mut opt = flame_core::adam::AdamW::new(3e-4, 0.9, 0.999, 1e-8, 0.01);
        for (i, (_, p)) in named.iter().enumerate() {
            let dims = p.tensor()?.shape().dims().to_vec();
            let m = Tensor::from_vec(
                (0..dims.iter().product::<usize>()).map(|j| (i + j) as f32 * 0.001).collect(),
                Shape::from_dims(&dims), device.clone()
            )?;
            let v = Tensor::from_vec(
                (0..dims.iter().product::<usize>()).map(|j| ((i + j) as f32 * 0.001).abs()).collect(),
                Shape::from_dims(&dims), device.clone()
            )?;
            opt.set_state(p, m, v);
        }
        opt.set_t(123);

        let header = CkptHeader::from_adamw("test", 456, &opt, 4, 1.0, 42, "deadbeef".into());
        save_full(&tmp, &named, &opt, &header)?;

        // Snapshot expected values BEFORE clearing.
        let saved_a0: Vec<f32> = named[0].1.tensor()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let saved_b0: Vec<f32> = named[1].1.tensor()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let (saved_m0, saved_v0) = opt.state_for(&named[0].1).expect("m/v for first param");
        let saved_m0_vec: Vec<f32> = saved_m0.to_vec1::<f32>()?;
        let saved_v0_vec: Vec<f32> = saved_v0.to_vec1::<f32>()?;

        // Reload into fresh state.
        let loaded = load_full(&tmp, &device)?;
        assert_eq!(loaded.header.step, 456);
        assert_eq!(loaded.header.adam_t, 123);
        assert_eq!(loaded.header.optimizer, "adamw");
        assert_eq!(loaded.header.rank, 4);
        assert!((loaded.header.alpha - 1.0).abs() < 1e-9);
        assert_eq!(loaded.header.config_hash, "deadbeef");
        assert_eq!(loaded.lora_tensors.len(), 4);
        assert_eq!(loaded.opt_m.len(), 4);
        assert_eq!(loaded.opt_v.len(), 4);

        let loaded_a0: Vec<f32> = loaded.lora_tensors["layers.0.lora_A.weight"].to_vec1::<f32>()?;
        let loaded_b0: Vec<f32> = loaded.lora_tensors["layers.0.lora_B.weight"].to_vec1::<f32>()?;
        let loaded_m0: Vec<f32> = loaded.opt_m["layers.0.lora_A.weight"].to_vec1::<f32>()?;
        let loaded_v0: Vec<f32> = loaded.opt_v["layers.0.lora_A.weight"].to_vec1::<f32>()?;

        // Round-trip is F32→F32, must be byte-exact.
        assert_eq!(loaded_a0, saved_a0);
        assert_eq!(loaded_b0, saved_b0);
        assert_eq!(loaded_m0, saved_m0_vec);
        assert_eq!(loaded_v0, saved_v0_vec);

        // apply_to_optimizer pairs by name + restores t.
        let mut fresh_opt = flame_core::adam::AdamW::new(3e-4, 0.9, 0.999, 1e-8, 0.01);
        let mut fresh_named: Vec<(String, Parameter)> = Vec::new();
        for i in 0..2 {
            let a = Tensor::zeros_dtype(Shape::from_dims(&[4, 8]), DType::F32, device.clone())?
                .requires_grad_(true);
            let b = Tensor::zeros_dtype(Shape::from_dims(&[8, 4]), DType::F32, device.clone())?
                .requires_grad_(true);
            fresh_named.push((format!("layers.{i}.lora_A.weight"), Parameter::new(a)));
            fresh_named.push((format!("layers.{i}.lora_B.weight"), Parameter::new(b)));
        }
        apply_lora_weights(&loaded, &fresh_named)?;
        apply_to_optimizer(&loaded, &mut fresh_opt, &fresh_named, 4, 1.0)?;

        assert_eq!(fresh_opt.t(), 123);
        let restored_a0: Vec<f32> = fresh_named[0].1.tensor()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        assert_eq!(restored_a0, saved_a0);
        let (restored_m0, _) = fresh_opt.state_for(&fresh_named[0].1).expect("m/v after apply");
        assert_eq!(restored_m0.to_vec1::<f32>()?, saved_m0_vec);

        // Wrong rank should refuse.
        let mut other = flame_core::adam::AdamW::new(3e-4, 0.9, 0.999, 1e-8, 0.01);
        let err = apply_to_optimizer(&loaded, &mut other, &fresh_named, /*expected_rank=*/8, 1.0);
        assert!(err.is_err(), "rank mismatch must refuse");

        // Wrong alpha should refuse.
        let err = apply_to_optimizer(&loaded, &mut other, &fresh_named, 4, /*expected_alpha=*/2.0);
        assert!(err.is_err(), "alpha mismatch must refuse");

        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    /// Loading a weights-only safetensors (no __metadata__/__eridiffusion_ckpt__)
    /// must produce a clear error pointing the user at --resume-lora.
    #[test]
    fn weights_only_load_refuses_with_clear_error() -> Result<()> {
        let device = flame_core::global_cuda_device();
        let tmp = std::env::temp_dir().join("eridiffusion_weights_only.safetensors");
        let _ = std::fs::remove_file(&tmp);

        let mut tensors: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], Shape::from_dims(&[3]), device.clone())?;
        tensors.insert("foo".into(), t);
        flame_core::serialization::save_tensors(
            &tensors, &tmp, flame_core::serialization::SerializationFormat::SafeTensors,
        )?;

        let res = load_full(&tmp, &device);
        let err = match res {
            Ok(_) => panic!("weights-only must refuse"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("--resume-lora"),
            "error must mention --resume-lora; got: {msg}"
        );

        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    /// Cover SimpleTuner's documented test fixture verbatim: the diffusers
    /// PEFT key for an SDXL unet attention down weight should rewrite to
    /// `lora_unet_..._to_k.lora_down.weight`.
    #[test]
    fn kohya_converter_simpletuner_fixture() -> Result<()> {
        let device = flame_core::global_cuda_device();
        let mut sd: HashMap<String, Tensor> = HashMap::new();
        // SimpleTuner test: unet attention down weight (PEFT key form).
        let down = Tensor::from_vec(vec![0.5f32; 32], Shape::from_dims(&[8, 4]), device.clone())?;
        let up = Tensor::from_vec(vec![0.25f32; 32], Shape::from_dims(&[4, 8]), device.clone())?;
        sd.insert(
            "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight".into(),
            down.clone(),
        );
        sd.insert(
            "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight".into(),
            up.clone(),
        );

        // PEFT-style keys for text encoders (sdxl=true → te1/te2).
        let te_d = Tensor::from_vec(vec![1.0f32; 4], Shape::from_dims(&[4]), device.clone())?;
        let te_u = Tensor::from_vec(vec![1.0f32; 4], Shape::from_dims(&[4]), device.clone())?;
        sd.insert("text_encoder.foo.bar.lora_A.weight".into(), te_d.clone());
        sd.insert("text_encoder.foo.bar.lora_B.weight".into(), te_u.clone());
        sd.insert("text_encoder_2.baz.qux.lora_A.weight".into(), te_d.clone());
        sd.insert("text_encoder_2.baz.qux.lora_B.weight".into(), te_u.clone());

        // .processor. substitution.
        sd.insert(
            "unet.x.processor.to_k.lora_A.weight".into(),
            down.clone(),
        );

        // Pass-through: not a known component prefix.
        let pass = Tensor::from_vec(vec![7.0f32], Shape::from_dims(&[1]), device.clone())?;
        sd.insert("not_a_component.thing.weight".into(), pass.clone());

        // Pass-through: known component prefix but no LoRA suffix.
        sd.insert("unet.something.bias".into(), pass.clone());

        let out = convert_diffusers_to_comfyui_sd_lora(&sd, Some(8.0), true, &device)?;

        // Fixture from SimpleTuner tests/test_lora_metadata.py.
        let expect_unet_down = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight";
        let expect_unet_up = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight";
        let expect_unet_alpha = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.alpha";
        assert!(out.contains_key(expect_unet_down), "missing {expect_unet_down}");
        assert!(out.contains_key(expect_unet_up), "missing {expect_unet_up}");
        assert!(out.contains_key(expect_unet_alpha), "missing alpha for unet attn key");

        // Text encoder rewrites (sdxl → lora_te1 / lora_te2).
        assert!(out.contains_key("lora_te1_foo_bar.lora_down.weight"));
        assert!(out.contains_key("lora_te1_foo_bar.lora_up.weight"));
        assert!(out.contains_key("lora_te1_foo_bar.alpha"));
        assert!(out.contains_key("lora_te2_baz_qux.lora_down.weight"));
        assert!(out.contains_key("lora_te2_baz_qux.lora_up.weight"));
        assert!(out.contains_key("lora_te2_baz_qux.alpha"));

        // .processor. → . substitution: no underscore-stranded `processor`.
        assert!(out.contains_key("lora_unet_x_to_k.lora_down.weight"));
        assert!(!out.keys().any(|k| k.contains("processor")));

        // Pass-through preserved unchanged.
        assert!(out.contains_key("not_a_component.thing.weight"));
        assert!(out.contains_key("unet.something.bias"));

        // Alpha is the requested value, F32, scalar.
        let alpha_t = &out[expect_unet_alpha];
        assert_eq!(alpha_t.shape().dims(), &[] as &[usize]);
        assert_eq!(alpha_t.dtype(), DType::F32);
        let alpha_v: Vec<f32> = alpha_t.to_vec1::<f32>()?;
        assert_eq!(alpha_v.len(), 1);
        assert!((alpha_v[0] - 8.0).abs() < 1e-6);

        // No alpha when None passed.
        let out_no_alpha = convert_diffusers_to_comfyui_sd_lora(&sd, None, true, &device)?;
        assert!(!out_no_alpha.keys().any(|k| k.ends_with(".alpha")));

        Ok(())
    }

    /// EDv2 SDXL trainer wrapper: real saved key like
    ///   input_blocks.4.1.transformer_blocks.0.attn1.to_q.lora_A.weight
    ///   input_blocks.4.1.transformer_blocks.0.attn1.to_q.alpha
    /// must convert to
    ///   lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight
    ///   lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_up.weight
    ///   lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.alpha
    /// and the original `.alpha` companion (already-LDM scalar) must be dropped.
    #[test]
    fn kohya_converter_sdxl_trainer_keys() -> Result<()> {
        let device = flame_core::global_cuda_device();
        let mut sd: HashMap<String, Tensor> = HashMap::new();

        let down = Tensor::from_vec(vec![0.1f32; 16], Shape::from_dims(&[4, 4]), device.clone())?;
        let up = Tensor::from_vec(vec![0.2f32; 16], Shape::from_dims(&[4, 4]), device.clone())?;
        let pre_alpha = Tensor::from_vec(vec![1.0f32], Shape::from_dims(&[]), device.clone())?;

        sd.insert(
            "input_blocks.4.1.transformer_blocks.0.attn1.to_q.lora_A.weight".into(),
            down.clone(),
        );
        sd.insert(
            "input_blocks.4.1.transformer_blocks.0.attn1.to_q.lora_B.weight".into(),
            up.clone(),
        );
        // SDXL trainer also writes `<prefix>.alpha` — converter must drop it.
        sd.insert(
            "input_blocks.4.1.transformer_blocks.0.attn1.to_q.alpha".into(),
            pre_alpha,
        );
        // Multiple modules to verify alpha is per-unique-kohya-key.
        sd.insert(
            "middle_block.1.transformer_blocks.0.ff.net.2.lora_A.weight".into(),
            down.clone(),
        );
        sd.insert(
            "middle_block.1.transformer_blocks.0.ff.net.2.lora_B.weight".into(),
            up.clone(),
        );

        let out = convert_sdxl_unet_to_kohya(&sd, 16.0, &device)?;

        let attn_down = "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight";
        let attn_up = "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_up.weight";
        let attn_alpha = "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.alpha";
        let ff_down = "lora_unet_middle_block_1_transformer_blocks_0_ff_net_2.lora_down.weight";
        let ff_up = "lora_unet_middle_block_1_transformer_blocks_0_ff_net_2.lora_up.weight";
        let ff_alpha = "lora_unet_middle_block_1_transformer_blocks_0_ff_net_2.alpha";

        for k in [attn_down, attn_up, attn_alpha, ff_down, ff_up, ff_alpha] {
            assert!(out.contains_key(k), "missing {k}; got keys: {:?}", out.keys().collect::<Vec<_>>());
        }
        // The pre-rewrite `.alpha` companion (LDM-style) must be gone.
        assert!(!out.contains_key("input_blocks.4.1.transformer_blocks.0.attn1.to_q.alpha"));
        // Old PEFT keys must be gone.
        assert!(!out.keys().any(|k| k.ends_with(".lora_A.weight") || k.ends_with(".lora_B.weight")));

        // Alpha value matches what the trainer passes (lora_alpha).
        let v: Vec<f32> = out[attn_alpha].to_vec1::<f32>()?;
        assert!((v[0] - 16.0).abs() < 1e-6);

        Ok(())
    }
}
