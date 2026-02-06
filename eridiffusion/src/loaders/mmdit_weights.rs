use crate::loaders::{PrefixedWeightLoader, WeightLoader};
use crate::models::mmdit_blocks::{
    DismantledBlock, FinalLayer, MMDiT, MMDiTConfig, PatchEmbed, QkNormKind, SelfAttention,
    TimestepEmbedder, VectorEmbedder, MLP,
};
#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
use crate::models::mmdit_cpu::{
    BlockSnapshot, Conv2dSnapshot, FinalLayerSnapshot, JointBlockSnapshot, MlpSnapshot,
    MmditCpuSnapshot, PatchEmbedSnapshot, QkNormSnapshot, SelfAttentionSnapshot,
    TimestepEmbedderSnapshot, VectorEmbedderSnapshot,
};
use crate::ops::{LayerNorm as OpsLayerNorm, Linear, RMSNorm as OpsRmsNorm};
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use bytemuck::try_cast_slice;
#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
use flame_core::cpu::{
    linear::LinearSnapshot,
    norm::{LayerNormSnapshot, RmsNormSnapshot},
    snapshot::{Bf16CpuSnapshot, F32CpuSnapshot},
};
use flame_core::{DType, Error, Result, Shape, Tensor};
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use half::bf16;
use log::{info, warn};
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use memmap2::MmapOptions;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use safetensors::{tensor::TensorView, SafeTensors};
use serde::Serialize;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use std::collections::HashSet;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use std::fs::File;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use std::path::Path;

/// Abstraction over hierarchical tensor sources (supports eager and streaming loaders).
pub trait TensorAccess: Clone {
    fn get_tensor(&self, key: &str) -> Result<Tensor>;
    fn has_tensor(&self, key: &str) -> bool;
    fn full_key(&self, key: &str) -> String;
    fn with_prefix(&self, prefix: &str) -> Self;
}

impl<'a> Clone for PrefixedWeightLoader<'a> {
    fn clone(&self) -> Self {
        Self { loader: self.loader, prefix: self.prefix.clone() }
    }
}

impl<'a> TensorAccess for PrefixedWeightLoader<'a> {
    fn get_tensor(&self, key: &str) -> Result<Tensor> {
        Ok(self.get(key)?.clone())
    }

    fn has_tensor(&self, key: &str) -> bool {
        let full = self.full_key(key);
        self.loader.keys().any(|k| k == &full)
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}.{}", self.prefix, key)
    }

    fn with_prefix(&self, prefix: &str) -> Self {
        Self { loader: self.loader, prefix: format!("{}.{}", self.prefix, prefix) }
    }
}

#[derive(Debug, Clone)]
pub struct MmditStructure {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
}

#[derive(Default, Debug, Clone, Serialize)]
pub struct MmditLoadReport {
    pub missing_required: Vec<String>,
    pub missing_optional: Vec<String>,
    pub dtype_mismatches: Vec<String>,
    pub shape_mismatches: Vec<String>,
}

impl MmditLoadReport {
    fn record_missing_required(&mut self, key: String) {
        if !self.missing_required.contains(&key) {
            self.missing_required.push(key);
        }
    }

    fn record_missing_optional(&mut self, key: String) {
        if !self.missing_optional.contains(&key) {
            self.missing_optional.push(key);
        }
    }

    fn record_dtype_mismatch(&mut self, key: String, expected: DType, got: DType) {
        self.dtype_mismatches.push(format!("{key}: expected dtype {expected:?}, got {got:?}"));
    }

    fn record_storage_mismatch(&mut self, key: String, expected: DType, got: DType) {
        self.dtype_mismatches
            .push(format!("{key}: expected storage dtype {expected:?}, got {got:?}"));
    }

    fn record_shape_mismatch(&mut self, key: String, expected: &[usize], got: &[usize]) {
        self.shape_mismatches.push(format!("{key}: expected shape {expected:?}, got {got:?}"));
    }

    pub fn has_issues(&self) -> bool {
        !self.missing_required.is_empty()
            || !self.missing_optional.is_empty()
            || !self.dtype_mismatches.is_empty()
            || !self.shape_mismatches.is_empty()
    }

    pub fn log_summary(&self, scope: &str) {
        if !self.has_issues() {
            info!("{scope}: MMDiT weights loaded cleanly");
            return;
        }

        fn emit(scope: &str, label: &str, entries: &[String]) {
            const MAX_PREVIEW: usize = 10;
            if entries.is_empty() {
                return;
            }
            if entries.len() <= MAX_PREVIEW {
                warn!("{scope}: {label}: {:?}", entries);
            } else {
                let preview: Vec<_> =
                    entries.iter().take(MAX_PREVIEW).map(|s| s.as_str()).collect();
                warn!("{scope}: {label}: {} entries (preview {:?})", entries.len(), preview);
            }
        }

        emit(scope, "missing required tensors", &self.missing_required);
        emit(scope, "missing optional tensors", &self.missing_optional);
        emit(scope, "dtype mismatches", &self.dtype_mismatches);
        emit(scope, "shape mismatches", &self.shape_mismatches);
    }
}

#[derive(Copy, Clone)]
pub(crate) enum LoadMode {
    Copy,
    Inspect,
}

impl LoadMode {
    fn should_copy(self) -> bool {
        matches!(self, LoadMode::Copy)
    }
}

/// Convenience helper that materializes weights eagerly. Prefer
/// `load_mmdit_weights_with_report` if you need diagnostics.
#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum MmditWeights {
    Gpu,
    Cpu(MmditCpuSnapshot),
}

pub fn load_mmdit_weights(model: &mut MMDiT, loader: &WeightLoader) -> Result<()> {
    let report = load_mmdit_weights_with_report(model, loader)?;
    report.log_summary("mmdit.load");
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
pub fn load_mmdit_weights_cpu(
    model: &mut MMDiT,
    loader: &WeightLoader,
) -> Result<(MmditCpuSnapshot, MmditLoadReport)> {
    let (snapshot, mut report) = capture_mmdit_cpu_snapshot(loader)?;
    model.apply_cpu_snapshot(&snapshot)?;
    report.log_summary("mmdit.load");
    Ok((snapshot, report))
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
pub fn capture_mmdit_cpu_snapshot(
    loader: &WeightLoader,
) -> Result<(MmditCpuSnapshot, MmditLoadReport)> {
    let (snapshot, report) = build_mmdit_cpu_snapshot(loader)?;
    Ok((snapshot, report))
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
pub fn capture_mmdit_cpu_snapshot_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<(MmditCpuSnapshot, MmditLoadReport)> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::InvalidOperation(format!("failed to open safetensors file: {e}")))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .map_err(|e| Error::InvalidOperation(format!("failed to mmap safetensors file: {e}")))?;
    let tensors = SafeTensors::deserialize(&*mmap)
        .map_err(|e| Error::InvalidOperation(format!("failed to deserialize safetensors: {e}")))?;
    let mut report = MmditLoadReport::default();
    let snapshot = build_mmdit_cpu_snapshot_from_views(&tensors, &mut report)?;
    Ok((snapshot, report))
}

/// Lazily materialize tensors only when needed. The returned loader captures
/// metadata and produces tensors on demand via `PrefixedWeightLoader`.
pub fn load_mmdit_weights_lazy<'a>(
    model: &mut MMDiT,
    loader: &'a WeightLoader,
) -> Result<PrefixedWeightLoader<'a>> {
    Ok(loader.pp("model.diffusion_model"))
}

pub fn infer_mmdit_structure(loader: &WeightLoader) -> Result<MmditStructure> {
    let root = loader.pp("model.diffusion_model");
    let context_block = root.pp("joint_blocks.0.context_block");
    let attn_loader = context_block.pp("attn");
    let attn_weight = attn_loader.get("qkv.weight")?;

    let dims = attn_weight.shape().dims();
    if dims.len() != 2 {
        return Err(Error::InvalidOperation(format!("unexpected qkv weight shape {:?}", dims)));
    }
    let hidden_size = dims[1];
    if hidden_size == 0 {
        return Err(Error::InvalidOperation(
            "cannot infer hidden size from empty qkv weight".into(),
        ));
    }

    if hidden_size % 64 != 0 {
        return Err(Error::InvalidOperation(format!(
            "hidden size {hidden_size} not divisible by 64; cannot infer head count"
        )));
    }
    let num_heads = hidden_size / 64;

    let depth = loader
        .keys()
        .filter_map(|k| {
            k.strip_prefix("model.diffusion_model.joint_blocks.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|idx| idx.parse::<usize>().ok())
        })
        .max()
        .map(|max_idx| max_idx + 1)
        .unwrap_or(0);

    if depth == 0 {
        return Err(Error::InvalidOperation("unable to infer depth from joint block keys".into()));
    }

    Ok(MmditStructure { hidden_size, num_heads, depth })
}

pub fn load_mmdit_weights_with_report(
    model: &mut MMDiT,
    loader: &WeightLoader,
) -> Result<MmditLoadReport> {
    load_mmdit_weights_mode(model, loader, LoadMode::Copy)
}

pub fn dry_run_mmdit_weights(model: &mut MMDiT, loader: &WeightLoader) -> Result<MmditLoadReport> {
    load_mmdit_weights_mode(model, loader, LoadMode::Inspect)
}

fn load_mmdit_weights_mode(
    model: &mut MMDiT,
    loader: &WeightLoader,
    mode: LoadMode,
) -> Result<MmditLoadReport> {
    let mut report = MmditLoadReport::default();
    let root = loader.pp("model.diffusion_model");

    load_patch_embed(model.patch_embed_mut(), &root, mode, &mut report)?;
    load_timestep_embedder(model.timestep_embedder_mut(), &root, mode, &mut report)?;
    if let Some(vector_embedder) = model.vector_embedder_mut() {
        load_vector_embedder(vector_embedder, &root, mode, &mut report)?;
    } else if loader.keys().any(|k| k.starts_with("model.diffusion_model.y_embedder")) {
        report.record_missing_optional("model.diffusion_model.y_embedder".into());
    }
    load_context_embedder(model.context_embedder_mut(), &root, mode, &mut report)?;
    load_final_layer(model.final_layer_mut(), &root, mode, &mut report)?;

    for (idx, block) in model.blocks_mut().iter_mut().enumerate() {
        let block_loader = root.with_prefix(&format!("joint_blocks.{}", idx));
        let context_loader = block_loader.with_prefix("context_block");
        load_block(block.context_block_mut(), &context_loader, mode, &mut report)?;
        let x_loader = block_loader.with_prefix("x_block");
        load_block(block.x_block_mut(), &x_loader, mode, &mut report)?;
    }

    if !loader.keys().any(|k| k == "model.diffusion_model.pos_embed.freqs") {
        report.record_missing_optional("model.diffusion_model.pos_embed.freqs".into());
    }

    Ok(report)
}

pub(crate) fn load_patch_embed<L: TensorAccess>(
    patch: &mut PatchEmbed,
    root: &L,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    {
        let proj = patch.proj_mut();
        let weight_shape = proj.weight.shape().dims().to_vec();
        let weight_dtype = proj.weight.dtype();
        let weight_storage = proj.weight.storage_dtype();
        process_required(
            root,
            "x_embedder.proj.weight",
            &weight_shape,
            weight_dtype,
            weight_storage,
            mode,
            report,
            |tensor| proj.copy_weight_from(tensor),
        )?;

        if proj.bias.is_some() {
            let bias_shape = proj.bias.as_ref().unwrap().shape().dims().to_vec();
            let bias_dtype = proj.bias.as_ref().unwrap().dtype();
            let bias_storage = proj.bias.as_ref().unwrap().storage_dtype();
            process_required(
                root,
                "x_embedder.proj.bias",
                &bias_shape,
                bias_dtype,
                bias_storage,
                mode,
                report,
                |tensor| proj.copy_bias_from(tensor),
            )?;
        }
    }

    Ok(())
}

pub(crate) fn load_timestep_embedder<L: TensorAccess>(
    embedder: &mut TimestepEmbedder,
    root: &L,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    let mlp = root.with_prefix("t_embedder.mlp");
    let (fc1, fc2) = embedder.linear_layers_mut();

    load_linear(fc1, &mlp, "0", mode, report)?;
    load_linear(fc2, &mlp, "2", mode, report)?;

    Ok(())
}

pub(crate) fn load_vector_embedder<L: TensorAccess>(
    embedder: &mut VectorEmbedder,
    root: &L,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    let mlp = root.with_prefix("y_embedder.mlp");
    let (fc1, fc2) = embedder.linear_layers_mut();
    load_linear(fc1, &mlp, "0", mode, report)?;
    load_linear(fc2, &mlp, "2", mode, report)?;
    Ok(())
}

pub(crate) fn load_context_embedder<L: TensorAccess>(
    linear: &mut Linear,
    root: &L,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    load_linear(linear, root, "context_embedder", mode, report)
}

pub(crate) fn load_final_layer<L: TensorAccess>(
    final_layer: &mut FinalLayer,
    root: &L,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    let loader = root.with_prefix("final_layer");

    {
        if loader_has_key(&loader, "norm_final.weight")
            || loader_has_key(&loader, "norm_final.bias")
        {
            let norm = final_layer.norm_mut();
            load_layer_norm(norm, &loader, "norm_final", mode, report)?;
        } else {
            report.record_missing_optional(full_key(&loader, "norm_final.*"));
        }
    }

    {
        let modulation = final_layer.modulation_mut();
        load_linear(modulation, &loader, "adaLN_modulation.1", mode, report)?;
    }

    {
        let proj = final_layer.proj_mut();
        load_linear(proj, &loader, "linear", mode, report)?;
    }

    Ok(())
}

pub(crate) fn load_block<L: TensorAccess>(
    block: &mut DismantledBlock,
    loader: &L,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    if loader_has_key(loader, "norm1.weight") || loader_has_key(loader, "norm1.bias") {
        let norm1 = block.norm1_mut();
        load_layer_norm(norm1, loader, "norm1", mode, report)?;
    } else {
        report.record_missing_optional(full_key(loader, "norm1.*"));
    }

    if let Some(norm2) = block.norm2_mut() {
        if loader_has_key(loader, "norm2.weight") || loader_has_key(loader, "norm2.bias") {
            load_layer_norm(norm2, loader, "norm2", mode, report)?;
        } else {
            report.record_missing_optional(full_key(loader, "norm2.*"));
        }
    } else if loader_has_key(loader, "norm2.weight") || loader_has_key(loader, "norm2.bias") {
        report.record_missing_optional(full_key(loader, "norm2.*"));
    }

    {
        let modulation = block.modulation_mut();
        load_linear(modulation, loader, "adaLN_modulation.1", mode, report)?;
    }

    load_self_attention(block.attn_mut(), loader, "attn", mode, report)?;

    if let Some(attn2) = block.attn2_mut() {
        load_self_attention(attn2, loader, "attn2", mode, report)?;
    }

    if let Some(mlp) = block.mlp_mut() {
        let mlp_loader = loader.with_prefix("mlp");
        load_mlp(mlp, mlp_loader, mode, report)?;
    }

    Ok(())
}

fn load_self_attention<L: TensorAccess>(
    attn: &mut SelfAttention,
    loader: &L,
    prefix: &str,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    let attn_loader = loader.with_prefix(prefix);
    let hidden = attn.hidden_size();
    let in_features = attn.in_features();
    let (weight_dtype, weight_storage, bias_dtype, bias_storage, has_bias) = {
        let q_proj = attn.q_proj_mut();
        let w_dtype = q_proj.weight.dtype();
        let w_storage = q_proj.weight.storage_dtype();
        let (b_dtype, b_storage, has_bias) = match q_proj.bias.as_ref() {
            Some(bias) => (bias.dtype(), bias.storage_dtype(), true),
            None => (q_proj.weight.dtype(), q_proj.weight.storage_dtype(), false),
        };
        (w_dtype, w_storage, b_dtype, b_storage, has_bias)
    };

    process_required(
        &attn_loader,
        "qkv.weight",
        &[hidden * 3, in_features],
        weight_dtype,
        weight_storage,
        mode,
        report,
        |tensor| attn.copy_qkv_weight_from(tensor),
    )?;

    if has_bias {
        process_required(
            &attn_loader,
            "qkv.bias",
            &[hidden * 3],
            bias_dtype,
            bias_storage,
            mode,
            report,
            |tensor| attn.copy_qkv_bias_from(tensor),
        )?;
    } else if loader_has_key(&attn_loader, "qkv.bias") {
        report.record_missing_optional(full_key(&attn_loader, "qkv.bias"));
    }

    if let Some(proj) = attn.proj_mut() {
        if loader_has_key(&attn_loader, "proj.weight") || loader_has_key(&attn_loader, "proj.bias")
        {
            load_linear(proj, &attn_loader, "proj", mode, report)?;
        } else {
            report.record_missing_optional(full_key(&attn_loader, "proj.*"));
        }
    }

    match attn.qk_norm_mut().kind() {
        QkNormKind::Layer => {
            let (norm_q, norm_k) = attn.qk_norm_mut().layer_norms_mut();
            if let (Some(norm_q), Some(norm_k)) = (norm_q, norm_k) {
                load_layer_norm(norm_q, &attn_loader, "ln_q", mode, report)?;
                load_layer_norm(norm_k, &attn_loader, "ln_k", mode, report)?;
            }
        }
        QkNormKind::Rms => {
            let (rms_q, rms_k) = attn.qk_norm_mut().rms_norms_mut();
            if let (Some(rms_q), Some(rms_k)) = (rms_q, rms_k) {
                load_rms_norm(rms_q, &attn_loader, "ln_q", mode, report)?;
                load_rms_norm(rms_k, &attn_loader, "ln_k", mode, report)?;
            }
        }
        QkNormKind::Disabled => {}
    }

    Ok(())
}

fn load_mlp<L: TensorAccess>(
    mlp: &mut MLP,
    loader: L,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    let (fc1, fc2) = mlp.fc_layers_mut();
    load_linear(fc1, &loader, "fc1", mode, report)?;
    load_linear(fc2, &loader, "fc2", mode, report)?;
    Ok(())
}

fn load_linear<L: TensorAccess>(
    linear: &mut Linear,
    loader: &L,
    key_prefix: &str,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    let weight_shape = linear.weight.shape().dims().to_vec();
    let weight_dtype = linear.weight.dtype();
    let weight_storage = linear.weight.storage_dtype();
    process_required(
        loader,
        &format!("{key_prefix}.weight"),
        &weight_shape,
        weight_dtype,
        weight_storage,
        mode,
        report,
        |tensor| linear.copy_weight_from(tensor),
    )?;

    if linear.bias.is_some() {
        let bias = linear.bias.as_ref().unwrap();
        let bias_shape = bias.shape().dims().to_vec();
        let bias_dtype = bias.dtype();
        let bias_storage = bias.storage_dtype();
        process_required(
            loader,
            &format!("{key_prefix}.bias"),
            &bias_shape,
            bias_dtype,
            bias_storage,
            mode,
            report,
            |tensor| linear.copy_bias_from(tensor),
        )?;
    }

    Ok(())
}

fn load_layer_norm<L: TensorAccess>(
    norm: &mut OpsLayerNorm,
    loader: &L,
    key_prefix: &str,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    if let Some(weight) = norm.weight.as_ref() {
        let shape = weight.shape().dims().to_vec();
        let dtype = weight.dtype();
        let storage = weight.storage_dtype();
        process_required(
            loader,
            &format!("{key_prefix}.weight"),
            &shape,
            dtype,
            storage,
            mode,
            report,
            |tensor| norm.copy_weight_from(tensor),
        )?;
    } else if loader_has_key(loader, &format!("{key_prefix}.weight")) {
        report.record_missing_optional(full_key(loader, &format!("{key_prefix}.weight")));
    }

    if let Some(bias) = norm.bias.as_ref() {
        let shape = bias.shape().dims().to_vec();
        let dtype = bias.dtype();
        let storage = bias.storage_dtype();
        process_required(
            loader,
            &format!("{key_prefix}.bias"),
            &shape,
            dtype,
            storage,
            mode,
            report,
            |tensor| norm.copy_bias_from(tensor),
        )?;
    } else if loader_has_key(loader, &format!("{key_prefix}.bias")) {
        report.record_missing_optional(full_key(loader, &format!("{key_prefix}.bias")));
    }

    Ok(())
}

fn load_rms_norm<L: TensorAccess>(
    norm: &mut OpsRmsNorm,
    loader: &L,
    key_prefix: &str,
    mode: LoadMode,
    report: &mut MmditLoadReport,
) -> Result<()> {
    if let Some(weight) = norm.weight.as_ref() {
        let weight_shape = weight.shape().dims().to_vec();
        let weight_dtype = weight.dtype();
        let weight_storage = weight.storage_dtype();
        process_required(
            loader,
            &format!("{key_prefix}.weight"),
            &weight_shape,
            weight_dtype,
            weight_storage,
            mode,
            report,
            |tensor| norm.copy_weight_from(tensor),
        )?;
    } else if loader_has_key(loader, &format!("{key_prefix}.weight")) {
        report.record_missing_optional(full_key(loader, &format!("{key_prefix}.weight")));
    }
    Ok(())
}

fn process_required<L, F>(
    loader: &L,
    key: &str,
    expected_shape: &[usize],
    expected_dtype: DType,
    expected_storage: DType,
    mode: LoadMode,
    report: &mut MmditLoadReport,
    mut apply: F,
) -> Result<()>
where
    L: TensorAccess,
    F: FnMut(&Tensor) -> Result<()>,
{
    let full = full_key(loader, key);
    match loader.get_tensor(key) {
        Ok(tensor) => {
            validate_tensor(
                &full,
                &tensor,
                expected_shape,
                expected_dtype,
                expected_storage,
                report,
            );
            if mode.should_copy() {
                apply(&tensor)?;
            }
            Ok(())
        }
        Err(err) => {
            report.record_missing_required(full);
            Err(err)
        }
    }
}

fn loader_has_key<L: TensorAccess>(loader: &L, key: &str) -> bool {
    loader.has_tensor(key)
}

fn full_key<L: TensorAccess>(loader: &L, key: &str) -> String {
    loader.full_key(key)
}

fn validate_tensor(
    full_key: &str,
    tensor: &Tensor,
    expected_shape: &[usize],
    expected_dtype: DType,
    expected_storage: DType,
    report: &mut MmditLoadReport,
) {
    let key = full_key.to_string();

    if tensor.shape().dims() != expected_shape {
        report.record_shape_mismatch(key.clone(), expected_shape, tensor.shape().dims());
    }

    if tensor.dtype() != expected_dtype {
        report.record_dtype_mismatch(key.clone(), expected_dtype, tensor.dtype());
    }

    if tensor.storage_dtype() != expected_storage {
        report.record_storage_mismatch(key, expected_storage, tensor.storage_dtype());
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn tensor_view_to_bf16_bits(view: &TensorView<'_>) -> Result<Vec<u16>> {
    match view.dtype() {
        safetensors::Dtype::BF16 => {
            let words = try_cast_slice::<u8, u16>(view.data()).map_err(|_| {
                Error::InvalidOperation("BF16 tensor data is not aligned to u16".into())
            })?;
            Ok(words.to_vec())
        }
        safetensors::Dtype::F32 => {
            let floats = try_cast_slice::<u8, f32>(view.data()).map_err(|_| {
                Error::InvalidOperation("F32 tensor data is not aligned to f32".into())
            })?;
            Ok(floats.iter().map(|&f| half::bf16::from_f32(f).to_bits()).collect())
        }
        safetensors::Dtype::F16 => {
            let halves = try_cast_slice::<u8, u16>(view.data()).map_err(|_| {
                Error::InvalidOperation("F16 tensor data is not aligned to u16".into())
            })?;
            Ok(halves
                .iter()
                .map(|&bits| {
                    let value = half::f16::from_bits(bits).to_f32();
                    half::bf16::from_f32(value).to_bits()
                })
                .collect())
        }
        other => Err(Error::InvalidOperation(format!(
            "Unsupported dtype {:?} for BF16 snapshot conversion",
            other
        ))),
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn tensor_view_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>> {
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let floats = try_cast_slice::<u8, f32>(view.data()).map_err(|_| {
                Error::InvalidOperation("F32 tensor data is not aligned to f32".into())
            })?;
            Ok(floats.to_vec())
        }
        safetensors::Dtype::BF16 => {
            let halves = try_cast_slice::<u8, u16>(view.data()).map_err(|_| {
                Error::InvalidOperation("BF16 tensor data is not aligned to u16".into())
            })?;
            Ok(halves.iter().map(|&bits| half::bf16::from_bits(bits).to_f32()).collect())
        }
        safetensors::Dtype::F16 => {
            let halves = try_cast_slice::<u8, u16>(view.data()).map_err(|_| {
                Error::InvalidOperation("F16 tensor data is not aligned to u16".into())
            })?;
            Ok(halves.iter().map(|&bits| half::f16::from_bits(bits).to_f32()).collect())
        }
        other => Err(Error::InvalidOperation(format!(
            "Unsupported dtype {:?} for F32 snapshot conversion",
            other
        ))),
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn map_safetensor_dtype(dtype: safetensors::Dtype) -> Option<DType> {
    match dtype {
        safetensors::Dtype::BF16 => Some(DType::BF16),
        safetensors::Dtype::F16 => Some(DType::F16),
        safetensors::Dtype::F32 => Some(DType::F32),
        _ => None,
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn dtype_satisfies(expected: DType, actual: DType) -> bool {
    match expected {
        DType::BF16 => matches!(actual, DType::BF16 | DType::F32 | DType::F16),
        DType::F32 => matches!(actual, DType::F32 | DType::BF16 | DType::F16),
        _ => expected == actual,
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn validate_view(
    key: &str,
    view: &TensorView<'_>,
    expected_shape: &[usize],
    expected_dtype: DType,
    report: &mut MmditLoadReport,
) {
    let actual_shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
    if actual_shape != expected_shape {
        report.record_shape_mismatch(key.to_string(), expected_shape, &actual_shape);
    }

    if let Some(actual_dtype) = map_safetensor_dtype(view.dtype()) {
        if !dtype_satisfies(expected_dtype, actual_dtype) {
            report.record_dtype_mismatch(key.to_string(), expected_dtype, actual_dtype);
        }
    } else {
        report.dtype_mismatches.push(format!(
            "{}: expected dtype {:?}, got unsupported safetensors dtype {:?}",
            key,
            expected_dtype,
            view.dtype()
        ));
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn fetch_required_bf16(
    tensors: &SafeTensors<'_>,
    key: &str,
    report: &mut MmditLoadReport,
) -> Result<(Vec<u16>, Vec<usize>)> {
    match tensors.tensor(key) {
        Ok(view) => {
            let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
            validate_view(key, &view, &shape, DType::BF16, report);
            let bits = tensor_view_to_bf16_bits(&view)?;
            Ok((bits, shape))
        }
        Err(_) => {
            report.record_missing_required(key.to_string());
            Err(Error::InvalidOperation(format!("Missing tensor {}", key)))
        }
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn fetch_optional_bf16(
    tensors: &SafeTensors<'_>,
    key: &str,
    report: &mut MmditLoadReport,
) -> Result<Option<(Vec<u16>, Vec<usize>)>> {
    match tensors.tensor(key) {
        Ok(view) => {
            let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
            validate_view(key, &view, &shape, DType::BF16, report);
            let bits = tensor_view_to_bf16_bits(&view)?;
            Ok(Some((bits, shape)))
        }
        Err(_) => Ok(None),
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn fetch_required_f32(
    tensors: &SafeTensors<'_>,
    key: &str,
    report: &mut MmditLoadReport,
) -> Result<(Vec<f32>, Vec<usize>)> {
    match tensors.tensor(key) {
        Ok(view) => {
            let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
            validate_view(key, &view, &shape, DType::F32, report);
            let values = tensor_view_to_f32(&view)?;
            Ok((values, shape))
        }
        Err(_) => {
            report.record_missing_required(key.to_string());
            Err(Error::InvalidOperation(format!("Missing tensor {}", key)))
        }
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn bf16_snapshot_from_bits(bits: Vec<u16>, dims: Vec<usize>) -> Result<Bf16CpuSnapshot> {
    let shape = Shape::from_dims(&dims);
    Bf16CpuSnapshot::new(bits, shape)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn f32_snapshot_from_values(values: Vec<f32>, dims: Vec<usize>) -> Result<F32CpuSnapshot> {
    let shape = Shape::from_dims(&dims);
    F32CpuSnapshot::new(values, shape)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn infer_inv_freq_from_pos_embed(values: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    if shape.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "Expected pos_embed shape [1, tokens, hidden], got {:?}",
            shape
        )));
    }
    if shape[0] != 1 {
        return Err(Error::InvalidOperation(format!(
            "pos_embed batch dimension must be 1, got {}",
            shape[0]
        )));
    }
    let tokens = shape[1];
    let hidden = shape[2];
    if hidden % 2 != 0 {
        return Err(Error::InvalidOperation(format!(
            "pos_embed hidden dimension {} must be even",
            hidden
        )));
    }
    let axis_dim = hidden / 2;
    if axis_dim % 2 != 0 {
        return Err(Error::InvalidOperation(format!(
            "pos_embed axis dimension {} must be even",
            axis_dim
        )));
    }
    let half_axis = axis_dim / 2;
    if half_axis == 0 {
        return Err(Error::InvalidOperation("pos_embed axis dimension is zero".into()));
    }
    let grid = (tokens as f64).sqrt().round() as usize;
    if grid * grid != tokens {
        return Err(Error::InvalidOperation(format!(
            "pos_embed token count {} is not a perfect square",
            tokens
        )));
    }
    if grid < 2 {
        return Err(Error::InvalidOperation(
            "Cannot infer RoPE frequencies from a single token".into(),
        ));
    }
    let token_stride = hidden;
    let base_index = 1 * token_stride; // h = 0, w = 1
    let mut inv_freq = Vec::with_capacity(half_axis);
    for i in 0..half_axis {
        let sin_val = values[base_index + i];
        let cos_val = values[base_index + half_axis + i];
        let mut angle = sin_val.atan2(cos_val);
        if angle < 0.0 {
            angle += 2.0 * std::f32::consts::PI;
        }
        inv_freq.push(angle);
    }
    Ok(inv_freq)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn select_key<'a>(key_set: &HashSet<String>, candidates: &[&'a str]) -> Option<&'a str> {
    candidates.iter().copied().find(|key| key_set.contains(*key))
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn attention_key_candidates(base_prefix: &str, primary: &str, alias: &str) -> Vec<String> {
    let mut keys = vec![format!("{base_prefix}.{primary}")];
    if let Some(parent) = base_prefix.strip_suffix("attn2") {
        keys.push(format!("{parent}attn.{alias}"));
    }
    keys
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn build_linear_snapshot(
    tensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: Option<&str>,
    key_set: &HashSet<String>,
    report: &mut MmditLoadReport,
) -> Result<LinearSnapshot> {
    let (weight_bits, weight_shape) = fetch_required_bf16(tensors, weight_key, report)?;
    let bias_bits = match bias_key {
        Some(key) => {
            if key_set.contains(key) {
                fetch_optional_bf16(tensors, key, report)?.map(|(bits, _)| bits)
            } else {
                report.record_missing_optional(key.to_string());
                None
            }
        }
        None => None,
    };
    let weight_shape = Shape::from_dims(&weight_shape);
    LinearSnapshot::new(weight_bits, weight_shape, bias_bits)
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn build_layer_norm_snapshot(
    tensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    normalized_shape: &[usize],
    key_set: &HashSet<String>,
    report: &mut MmditLoadReport,
    required: bool,
) -> Result<Option<LayerNormSnapshot>> {
    let has_weight = key_set.contains(weight_key);
    let has_bias = key_set.contains(bias_key);
    if !has_weight {
        if required {
            report.record_missing_required(weight_key.to_string());
            return Err(Error::InvalidOperation(format!("Missing tensor {}", weight_key)));
        } else {
            return Ok(None);
        }
    }
    let (weight_bits, _) = fetch_required_bf16(tensors, weight_key, report)?;
    let bias_bits = if has_bias {
        fetch_optional_bf16(tensors, bias_key, report)?.map(|(bits, _)| bits)
    } else {
        if required {
            report.record_missing_required(bias_key.to_string());
            return Err(Error::InvalidOperation(format!("Missing tensor {}", bias_key)));
        } else {
            report.record_missing_optional(bias_key.to_string());
            None
        }
    };
    let snapshot =
        LayerNormSnapshot::new(normalized_shape.to_vec(), 1e-6, Some(weight_bits), bias_bits)?;
    Ok(Some(snapshot))
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn build_rms_norm_snapshot(
    tensors: &SafeTensors<'_>,
    weight_key: &str,
    normalized_shape: &[usize],
    key_set: &HashSet<String>,
    report: &mut MmditLoadReport,
    required: bool,
) -> Result<Option<RmsNormSnapshot>> {
    if !key_set.contains(weight_key) {
        if required {
            report.record_missing_required(weight_key.to_string());
            return Err(Error::InvalidOperation(format!("Missing tensor {}", weight_key)));
        }
        return Ok(None);
    }
    let (weight_bits, _) = fetch_required_bf16(tensors, weight_key, report)?;
    let snapshot = RmsNormSnapshot::new(normalized_shape.to_vec(), 1e-6, Some(weight_bits))?;
    Ok(Some(snapshot))
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn build_qk_norm_snapshot(
    tensors: &SafeTensors<'_>,
    base_prefix: &str,
    head_dim: usize,
    key_set: &HashSet<String>,
    report: &mut MmditLoadReport,
) -> Result<QkNormSnapshot> {
    let ln_q_weight = format!("{base_prefix}.ln_q.weight");
    let ln_q_bias = format!("{base_prefix}.ln_q.bias");
    let ln_k_weight = format!("{base_prefix}.ln_k.weight");
    let ln_k_bias = format!("{base_prefix}.ln_k.bias");

    let has_q_weight = key_set.contains(&ln_q_weight);
    let has_q_bias = key_set.contains(&ln_q_bias);
    let has_k_weight = key_set.contains(&ln_k_weight);
    let has_k_bias = key_set.contains(&ln_k_bias);

    if !has_q_weight && !has_k_weight {
        return Ok(QkNormSnapshot {
            kind: QkNormKind::Disabled,
            norm_q: None,
            norm_k: None,
            rms_q: None,
            rms_k: None,
        });
    }

    let normalized_shape = vec![head_dim];
    if has_q_bias || has_k_bias {
        // Treat as LayerNorm
        let norm_q = build_layer_norm_snapshot(
            tensors,
            &ln_q_weight,
            &ln_q_bias,
            &normalized_shape,
            key_set,
            report,
            true,
        )?
        .ok_or_else(|| {
            Error::InvalidOperation(format!("Expected LayerNorm weights for {}", base_prefix))
        })?;
        let norm_k = build_layer_norm_snapshot(
            tensors,
            &ln_k_weight,
            &ln_k_bias,
            &normalized_shape,
            key_set,
            report,
            true,
        )?
        .ok_or_else(|| {
            Error::InvalidOperation(format!("Expected LayerNorm weights for {}", base_prefix))
        })?;
        Ok(QkNormSnapshot {
            kind: QkNormKind::Layer,
            norm_q: Some(norm_q),
            norm_k: Some(norm_k),
            rms_q: None,
            rms_k: None,
        })
    } else {
        // Treat as RMSNorm
        let norm_q = build_rms_norm_snapshot(
            tensors,
            &ln_q_weight,
            &normalized_shape,
            key_set,
            report,
            true,
        )?
        .ok_or_else(|| {
            Error::InvalidOperation(format!("Expected RMSNorm weights for {}", base_prefix))
        })?;
        let norm_k = build_rms_norm_snapshot(
            tensors,
            &ln_k_weight,
            &normalized_shape,
            key_set,
            report,
            true,
        )?
        .ok_or_else(|| {
            Error::InvalidOperation(format!("Expected RMSNorm weights for {}", base_prefix))
        })?;
        Ok(QkNormSnapshot {
            kind: QkNormKind::Rms,
            norm_q: None,
            norm_k: None,
            rms_q: Some(norm_q),
            rms_k: Some(norm_k),
        })
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn build_self_attention_snapshot(
    tensors: &SafeTensors<'_>,
    base_prefix: &str,
    num_heads: usize,
    key_set: &HashSet<String>,
    report: &mut MmditLoadReport,
) -> Result<SelfAttentionSnapshot> {
    let qkv_weight_candidates = attention_key_candidates(base_prefix, "qkv.weight", "qkv_1.weight");
    let qkv_weight_key =
        qkv_weight_candidates.iter().find(|k| key_set.contains(*k)).cloned().ok_or_else(|| {
            report.record_missing_required(format!("{base_prefix}.qkv.weight"));
            Error::InvalidOperation(format!(
                "Missing fused QKV weight for {} (tried {:?})",
                base_prefix, qkv_weight_candidates
            ))
        })?;
    let (fused_bits, fused_shape) = fetch_required_bf16(tensors, qkv_weight_key.as_str(), report)?;
    if fused_shape.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "{} expected rank-2 weight, got {:?}",
            qkv_weight_key, fused_shape
        )));
    }
    let rows = fused_shape[0];
    let cols = fused_shape[1];
    if rows % 3 != 0 {
        return Err(Error::InvalidOperation(format!(
            "{} rows {} not divisible by 3",
            qkv_weight_key, rows
        )));
    }
    let hidden = rows / 3;
    if hidden % num_heads != 0 {
        return Err(Error::InvalidOperation(format!(
            "hidden size {} not divisible by num_heads {}",
            hidden, num_heads
        )));
    }
    let head_dim = hidden / num_heads;
    let per_matrix = hidden * cols;
    if fused_bits.len() != rows * cols {
        return Err(Error::InvalidOperation(format!(
            "{} data length {} does not match shape {:?}",
            qkv_weight_key,
            fused_bits.len(),
            fused_shape
        )));
    }
    let (q_bits, rest) = fused_bits.split_at(per_matrix);
    let (k_bits, rest) = rest.split_at(per_matrix);
    let (v_bits, _) = rest.split_at(per_matrix);

    let weight_shape = Shape::from_dims(&[hidden, cols]);
    let mut q_bias_bits: Option<Vec<u16>> = None;
    let mut k_bias_bits: Option<Vec<u16>> = None;
    let mut v_bias_bits: Option<Vec<u16>> = None;
    let qkv_bias_candidates = attention_key_candidates(base_prefix, "qkv.bias", "qkv_1.bias");
    if let Some(qkv_bias_key) = qkv_bias_candidates.iter().find(|k| key_set.contains(*k)).cloned() {
        if let Some((bias_bits, bias_shape)) =
            fetch_optional_bf16(tensors, qkv_bias_key.as_str(), report)?
        {
            if bias_shape.len() != 1 || bias_shape[0] != hidden * 3 {
                return Err(Error::InvalidOperation(format!(
                    "{} expected length {}, got {:?}",
                    qkv_bias_key,
                    hidden * 3,
                    bias_shape
                )));
            }
            let (q_bias, rest) = bias_bits.split_at(hidden);
            let (k_bias, rest) = rest.split_at(hidden);
            let (v_bias, _) = rest.split_at(hidden);
            q_bias_bits = Some(q_bias.to_vec());
            k_bias_bits = Some(k_bias.to_vec());
            v_bias_bits = Some(v_bias.to_vec());
        }
    }

    let q_snapshot = LinearSnapshot::new(q_bits.to_vec(), weight_shape.clone(), q_bias_bits)?;
    let k_snapshot = LinearSnapshot::new(k_bits.to_vec(), weight_shape.clone(), k_bias_bits)?;
    let v_snapshot = LinearSnapshot::new(v_bits.to_vec(), weight_shape.clone(), v_bias_bits)?;

    let proj_weight_candidates =
        attention_key_candidates(base_prefix, "proj.weight", "proj_1.weight");
    let proj_weight_key = proj_weight_candidates.iter().find(|k| key_set.contains(*k)).cloned();
    let proj_snapshot = proj_weight_key
        .map(|weight_key| {
            let proj_bias_candidates =
                attention_key_candidates(base_prefix, "proj.bias", "proj_1.bias");
            let proj_bias_key =
                proj_bias_candidates.iter().find(|k| key_set.contains(*k)).map(|s| s.as_str());
            build_linear_snapshot(tensors, weight_key.as_str(), proj_bias_key, key_set, report)
        })
        .transpose()?;

    let qk_norm = build_qk_norm_snapshot(tensors, base_prefix, head_dim, key_set, report)?;

    Ok(SelfAttentionSnapshot {
        num_heads,
        head_dim,
        q: q_snapshot,
        k: k_snapshot,
        v: v_snapshot,
        proj: proj_snapshot,
        qk_norm,
    })
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn build_mlp_snapshot(
    tensors: &SafeTensors<'_>,
    base_prefix: &str,
    key_set: &HashSet<String>,
    report: &mut MmditLoadReport,
) -> Result<MlpSnapshot> {
    let fc1 = build_linear_snapshot(
        tensors,
        &format!("{base_prefix}.fc1.weight"),
        Some(&format!("{base_prefix}.fc1.bias")),
        key_set,
        report,
    )?;
    let fc2 = build_linear_snapshot(
        tensors,
        &format!("{base_prefix}.fc2.weight"),
        Some(&format!("{base_prefix}.fc2.bias")),
        key_set,
        report,
    )?;
    Ok(MlpSnapshot { fc1, fc2 })
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn build_block_snapshot(
    tensors: &SafeTensors<'_>,
    block_prefix: &str,
    hidden: usize,
    num_heads: usize,
    expected_pre_only: bool,
    key_set: &HashSet<String>,
    report: &mut MmditLoadReport,
) -> Result<BlockSnapshot> {
    let attn2_weight_key = format!("{block_prefix}.attn2.qkv.weight");
    let pre_only = expected_pre_only;
    let self_attn = key_set.contains(&attn2_weight_key);

    let norm1 = match build_layer_norm_snapshot(
        tensors,
        &format!("{block_prefix}.norm1.weight"),
        &format!("{block_prefix}.norm1.bias"),
        &[hidden],
        key_set,
        report,
        false,
    )? {
        Some(snapshot) => snapshot,
        None => {
            let ones = vec![bf16::from_f32(1.0).to_bits(); hidden];
            let zeros = vec![bf16::from_f32(0.0).to_bits(); hidden];
            LayerNormSnapshot::new(vec![hidden], 1e-6, Some(ones), Some(zeros))?
        }
    };

    let attn = build_self_attention_snapshot(
        tensors,
        &format!("{block_prefix}.attn"),
        num_heads,
        key_set,
        report,
    )?;

    let attn2 = if self_attn {
        Some(build_self_attention_snapshot(
            tensors,
            &format!("{block_prefix}.attn2"),
            num_heads,
            key_set,
            report,
        )?)
    } else {
        None
    };

    let norm2 = if pre_only {
        None
    } else {
        match build_layer_norm_snapshot(
            tensors,
            &format!("{block_prefix}.norm2.weight"),
            &format!("{block_prefix}.norm2.bias"),
            &[hidden],
            key_set,
            report,
            false,
        )? {
            Some(snapshot) => Some(snapshot),
            None => {
                let ones = vec![bf16::from_f32(1.0).to_bits(); hidden];
                let zeros = vec![bf16::from_f32(0.0).to_bits(); hidden];
                Some(LayerNormSnapshot::new(vec![hidden], 1e-6, Some(ones), Some(zeros))?)
            }
        }
    };

    let mlp = if pre_only {
        None
    } else {
        Some(build_mlp_snapshot(tensors, &format!("{block_prefix}.mlp"), key_set, report)?)
    };

    let modulation = build_linear_snapshot(
        tensors,
        &format!("{block_prefix}.adaLN_modulation.1.weight"),
        Some(&format!("{block_prefix}.adaLN_modulation.1.bias")),
        key_set,
        report,
    )?;

    Ok(BlockSnapshot {
        hidden,
        num_heads,
        pre_only,
        self_attn,
        norm1,
        attn,
        attn2,
        norm2,
        mlp,
        modulation,
    })
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn build_joint_block_snapshot(
    tensors: &SafeTensors<'_>,
    index: usize,
    depth: usize,
    hidden: usize,
    num_heads: usize,
    key_set: &HashSet<String>,
    report: &mut MmditLoadReport,
) -> Result<JointBlockSnapshot> {
    let base = format!("model.diffusion_model.joint_blocks.{index}");
    let context = build_block_snapshot(
        tensors,
        &format!("{base}.context_block"),
        hidden,
        num_heads,
        index + 1 == depth,
        key_set,
        report,
    )?;
    let x = build_block_snapshot(
        tensors,
        &format!("{base}.x_block"),
        hidden,
        num_heads,
        false,
        key_set,
        report,
    )?;
    Ok(JointBlockSnapshot { context, x })
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
fn build_mmdit_cpu_snapshot_from_views(
    tensors: &SafeTensors<'_>,
    report: &mut MmditLoadReport,
) -> Result<MmditCpuSnapshot> {
    let key_set: HashSet<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();

    // Determine depth and self-attention limit
    let mut depth = 0usize;
    let mut x_self_attn_limit: Option<usize> = None;
    for key in &key_set {
        if let Some(rest) = key.strip_prefix("model.diffusion_model.joint_blocks.") {
            if let Some((idx_str, tail)) = rest.split_once('.') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    depth = depth.max(idx + 1);
                    if tail.starts_with("x_block.attn2") {
                        x_self_attn_limit = Some(x_self_attn_limit.map_or(idx, |cur| cur.max(idx)));
                    }
                }
            }
        }
    }
    if depth == 0 {
        return Err(Error::InvalidOperation(
            "No joint_blocks.* tensors found in safetensors".into(),
        ));
    }

    // Patch embed snapshot and config seeds
    let patch_weight_key = "model.diffusion_model.x_embedder.proj.weight";
    let (patch_weight_bits, patch_weight_shape) =
        fetch_required_bf16(tensors, patch_weight_key, report)?;
    if patch_weight_shape.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "{} expected rank-4 tensor, got {:?}",
            patch_weight_key, patch_weight_shape
        )));
    }
    let embed_dim = patch_weight_shape[0];
    let in_channels = patch_weight_shape[1];
    let kernel_h = patch_weight_shape[2];
    let kernel_w = patch_weight_shape[3];
    if kernel_h != kernel_w {
        return Err(Error::InvalidOperation(format!(
            "{} kernel dims {:?} not square",
            patch_weight_key, patch_weight_shape
        )));
    }
    let patch_size = kernel_h;
    let patch_weight_snapshot = bf16_snapshot_from_bits(patch_weight_bits, patch_weight_shape)?;

    let patch_bias_key = "model.diffusion_model.x_embedder.proj.bias";
    let patch_bias_snapshot = if key_set.contains(patch_bias_key) {
        fetch_optional_bf16(tensors, patch_bias_key, report)?
            .map(|(bits, dims)| bf16_snapshot_from_bits(bits, dims))
            .transpose()?
    } else {
        report.record_missing_optional(patch_bias_key.to_string());
        None
    };

    let patch_embed_snapshot = PatchEmbedSnapshot {
        proj: Conv2dSnapshot {
            in_channels,
            out_channels: embed_dim,
            kernel_size: (patch_size, patch_size),
            stride: (patch_size, patch_size),
            padding: (0, 0),
            groups: 1,
            weight: patch_weight_snapshot,
            bias: patch_bias_snapshot,
        },
        flatten: true,
        dynamic_img_pad: true,
        patch_size,
    };

    // Determine hidden size / head dim / num heads from first block attention
    let first_attn_weight_key =
        "model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight";
    let (_, first_attn_shape) = fetch_required_bf16(tensors, first_attn_weight_key, report)?;
    if first_attn_shape.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "{} expected rank-2 tensor, got {:?}",
            first_attn_weight_key, first_attn_shape
        )));
    }
    let hidden_size = first_attn_shape[0] / 3;
    if hidden_size == 0 {
        return Err(Error::InvalidOperation("Derived hidden size is zero".into()));
    }

    let ln_q_key = "model.diffusion_model.joint_blocks.0.context_block.attn.ln_q.weight";
    let head_dim = if key_set.contains(ln_q_key) {
        let view = tensors
            .tensor(ln_q_key)
            .map_err(|_| Error::InvalidOperation(format!("Failed to read {}", ln_q_key)))?;
        let dims: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        dims.get(0).copied().unwrap_or(hidden_size)
    } else {
        hidden_size
    };
    if head_dim == 0 || hidden_size % head_dim != 0 {
        return Err(Error::InvalidOperation(format!(
            "Cannot infer num_heads from hidden {} and head_dim {}",
            hidden_size, head_dim
        )));
    }
    let num_heads = hidden_size / head_dim;

    let qkv_bias_present = key_set
        .contains("model.diffusion_model.joint_blocks.0.context_block.attn.qkv.bias")
        || key_set.contains("model.diffusion_model.joint_blocks.0.context_block.attn.qkv_1.bias");

    // Timestep embedder
    let t_embed_linear1 = build_linear_snapshot(
        tensors,
        "model.diffusion_model.t_embedder.mlp.0.weight",
        Some("model.diffusion_model.t_embedder.mlp.0.bias"),
        &key_set,
        report,
    )?;
    let t_embed_linear2 = build_linear_snapshot(
        tensors,
        "model.diffusion_model.t_embedder.mlp.2.weight",
        Some("model.diffusion_model.t_embedder.mlp.2.bias"),
        &key_set,
        report,
    )?;
    let frequency_embedding_size =
        t_embed_linear1.weight.shape().dims().get(1).copied().unwrap_or(0);

    let timestep_embedder_snapshot = TimestepEmbedderSnapshot {
        linear1: t_embed_linear1,
        linear2: t_embed_linear2,
        frequency_embedding_size,
    };

    // Optional vector embedder
    let vector_embedder_snapshot =
        if key_set.contains("model.diffusion_model.y_embedder.mlp.0.weight") {
            let vec_linear1 = build_linear_snapshot(
                tensors,
                "model.diffusion_model.y_embedder.mlp.0.weight",
                Some("model.diffusion_model.y_embedder.mlp.0.bias"),
                &key_set,
                report,
            )?;
            let vec_linear2 = build_linear_snapshot(
                tensors,
                "model.diffusion_model.y_embedder.mlp.2.weight",
                Some("model.diffusion_model.y_embedder.mlp.2.bias"),
                &key_set,
                report,
            )?;
            let input_dim = vec_linear1.weight.shape().dims().get(1).copied().unwrap_or(0);
            Some(VectorEmbedderSnapshot { linear1: vec_linear1, linear2: vec_linear2, input_dim })
        } else {
            None
        };

    // Context embedder
    let context_embedder_snapshot = build_linear_snapshot(
        tensors,
        "model.diffusion_model.context_embedder.weight",
        Some("model.diffusion_model.context_embedder.bias"),
        &key_set,
        report,
    )?;
    let context_dim = context_embedder_snapshot.weight.shape().dims().get(1).copied().unwrap_or(0);

    // Final layer snapshot
    let final_layer_norm = match build_layer_norm_snapshot(
        tensors,
        "model.diffusion_model.final_layer.norm_final.weight",
        "model.diffusion_model.final_layer.norm_final.bias",
        &[hidden_size],
        &key_set,
        report,
        false,
    )? {
        Some(snapshot) => snapshot,
        None => {
            let ones = vec![bf16::from_f32(1.0).to_bits(); hidden_size];
            let zeros = vec![bf16::from_f32(0.0).to_bits(); hidden_size];
            LayerNormSnapshot::new(vec![hidden_size], 1e-6, Some(ones), Some(zeros))?
        }
    };
    let final_layer_modulation = build_linear_snapshot(
        tensors,
        "model.diffusion_model.final_layer.adaLN_modulation.1.weight",
        Some("model.diffusion_model.final_layer.adaLN_modulation.1.bias"),
        &key_set,
        report,
    )?;
    let final_layer_proj_weight = select_key(
        &key_set,
        &[
            "model.diffusion_model.final_layer.proj.weight",
            "model.diffusion_model.final_layer.linear.weight",
        ],
    )
    .ok_or_else(|| {
        Error::InvalidOperation(
            "Missing final layer projection weight (proj.weight / linear.weight)".into(),
        )
    })?;
    let final_layer_proj_bias = select_key(
        &key_set,
        &[
            "model.diffusion_model.final_layer.proj.bias",
            "model.diffusion_model.final_layer.linear.bias",
        ],
    );
    let final_layer_proj = build_linear_snapshot(
        tensors,
        final_layer_proj_weight,
        final_layer_proj_bias,
        &key_set,
        report,
    )?;
    let proj_rows =
        final_layer_proj.weight.shape().dims().get(0).copied().unwrap_or(patch_size * patch_size);
    let out_channels = proj_rows / (patch_size * patch_size);

    let final_layer_snapshot = FinalLayerSnapshot {
        norm: final_layer_norm,
        modulation: final_layer_modulation,
        proj: final_layer_proj,
        patch_size,
        out_channels,
    };

    // Position embedding snapshot
    let pos_frequencies = if key_set.contains("model.diffusion_model.pos_embed.freqs") {
        let (pos_values, pos_shape) =
            fetch_required_f32(tensors, "model.diffusion_model.pos_embed.freqs", report)?;
        f32_snapshot_from_values(pos_values, pos_shape)?
    } else if key_set.contains("model.diffusion_model.pos_embed") {
        let (pos_tokens, pos_shape) =
            fetch_required_f32(tensors, "model.diffusion_model.pos_embed", report)?;
        let inv_freq = infer_inv_freq_from_pos_embed(&pos_tokens, &pos_shape)?;
        f32_snapshot_from_values(inv_freq.clone(), vec![inv_freq.len()])?
    } else {
        report.record_missing_required("model.diffusion_model.pos_embed".into());
        return Err(Error::InvalidOperation(
            "Missing positional embedding tensors in checkpoint".into(),
        ));
    };

    // MLP ratio (use first block x_block fc1 shape if available)
    let mlp_ratio =
        if key_set.contains("model.diffusion_model.joint_blocks.0.x_block.mlp.fc1.weight") {
            let view = tensors
                .tensor("model.diffusion_model.joint_blocks.0.x_block.mlp.fc1.weight")
                .map_err(|_| {
                    Error::InvalidOperation(
                        "Failed to read joint_blocks.0.x_block.mlp.fc1.weight".into(),
                    )
                })?;
            let dims: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
            if dims.len() == 2 && dims[1] == hidden_size {
                dims[0] as f32 / hidden_size as f32
            } else {
                4.0
            }
        } else {
            4.0
        };

    // Build block snapshots
    let mut blocks = Vec::with_capacity(depth);
    for idx in 0..depth {
        blocks.push(build_joint_block_snapshot(
            tensors,
            idx,
            depth,
            hidden_size,
            num_heads,
            &key_set,
            report,
        )?);
    }

    let config = MMDiTConfig {
        hidden_size,
        num_heads,
        depth,
        mlp_ratio,
        qkv_bias: qkv_bias_present,
        qk_norm: blocks.first().map(|blk| blk.x.attn.qk_norm.kind).unwrap_or(QkNormKind::Disabled),
        pos_embed_max_size: pos_frequencies.shape().dims().get(0).copied().unwrap_or(0),
        x_self_attn_layers: x_self_attn_limit,
        patch_size,
        in_channels,
        out_channels,
        frequency_embedding_size,
        context_dim,
        pooled_dim: vector_embedder_snapshot
            .as_ref()
            .and_then(|snap| snap.linear1.weight.shape().dims().get(1).copied()),
    };

    Ok(MmditCpuSnapshot {
        config,
        patch_embed: patch_embed_snapshot,
        timestep_embedder: timestep_embedder_snapshot,
        vector_embedder: vector_embedder_snapshot,
        context_embedder: context_embedder_snapshot,
        blocks,
        final_layer: final_layer_snapshot,
        pos_frequencies,
    })
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
pub fn build_mmdit_cpu_snapshot(
    loader: &WeightLoader,
) -> Result<(MmditCpuSnapshot, MmditLoadReport)> {
    if !loader.has_mmap() {
        return Err(Error::InvalidOperation(
            "WeightLoader does not retain a safetensors mmap; rebuild via \
             WeightLoader::from_safetensors_with_dtype for CPU snapshot support."
                .into(),
        ));
    }
    let mut report = MmditLoadReport::default();
    let snapshot = loader
        .with_safetensors(|tensors| build_mmdit_cpu_snapshot_from_views(tensors, &mut report))?;
    Ok((snapshot, report))
}
