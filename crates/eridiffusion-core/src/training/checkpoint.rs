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

    serialization::save_tensors_with_metadata(&tensors, &metadata, path)
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
}
