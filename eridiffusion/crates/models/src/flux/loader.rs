//! Strict Flux weight loaders (GPU-only, BF16 storage).

use std::path::Path;

use anyhow::Result;
use eridiffusion_core::Device;
use flame_core::{DType, Tensor};

use crate::devtensor::BF16;
use super::keys::{default_keyconv, KeyConv};
use super::loader_utils::{
    load_tensor_to_device,
    split_qkv_bias,
    split_qkv_weight,
    transpose_out_in,
    STFile,
};

#[derive(Debug)]
pub struct FluxBlockWeights {
    pub q_w: Tensor,
    pub k_w: Tensor,
    pub v_w: Tensor,
    pub o_w: Tensor,
    pub q_b: Option<Tensor>,
    pub k_b: Option<Tensor>,
    pub v_b: Option<Tensor>,
    pub o_b: Option<Tensor>,
    pub fc1_w: Tensor,
    pub fc2_w: Tensor,
    pub fc1_b: Option<Tensor>,
    pub fc2_b: Option<Tensor>,
}

#[derive(Debug, Default)]
pub struct FluxBackboneWeights {}

#[derive(Debug)]
pub struct FluxPacks {
    pub blocks: Vec<FluxBlockWeights>,
    pub backbone: FluxBackboneWeights,
}

pub fn load_flux_block(st: &STFile, device: &Device, index: usize, kc: &KeyConv) -> Result<FluxBlockWeights> {
    let force = Some(BF16);

    let qkv_w = must(st, &kc.img_attn_qkv_weight(index), device, force)?;
    let (mut q_w, mut k_w, mut v_w) = split_qkv_weight(qkv_w)?;
    q_w = transpose_out_in(q_w)?;
    k_w = transpose_out_in(k_w)?;
    v_w = transpose_out_in(v_w)?;

    let qkv_b = maybe(st, &kc.img_attn_qkv_bias(index), device, force)?;
    let (q_b, k_b, v_b) = if let Some(b) = qkv_b {
        let (qb, kb, vb) = split_qkv_bias(b)?;
        (Some(qb), Some(kb), Some(vb))
    } else {
        (None, None, None)
    };

    let o_w = transpose_out_in(must(st, &kc.img_attn_proj_weight(index), device, force)?)?;
    let o_b = maybe(st, &kc.img_attn_proj_bias(index), device, force)?;

    let fc1_w = must(st, &kc.img_mlp_fc1_weight(index), device, force)?;
    let fc1_b = maybe(st, &kc.img_mlp_fc1_bias(index), device, force)?;
    let fc2_w = must(st, &kc.img_mlp_fc2_weight(index), device, force)?;
    let fc2_b = maybe(st, &kc.img_mlp_fc2_bias(index), device, force)?;

    Ok(FluxBlockWeights {
        q_w,
        k_w,
        v_w,
        o_w,
        q_b,
        k_b,
        v_b,
        o_b,
        fc1_w,
        fc2_w,
        fc1_b,
        fc2_b,
    })
}

pub fn load_flux_backbone(_st: &STFile, _device: &Device, _kc: &KeyConv) -> Result<FluxBackboneWeights> {
    Ok(FluxBackboneWeights::default())
}

pub fn load_flux_packs(path: impl AsRef<Path>, device: &Device) -> Result<FluxPacks> {
    load_flux_packs_with(path, device, default_keyconv(), None)
}

pub fn load_flux_packs_with(
    path: impl AsRef<Path>,
    device: &Device,
    kc: KeyConv,
    num_blocks: Option<usize>,
) -> Result<FluxPacks> {
    let st = STFile::open(path)?;
    let total = match num_blocks {
        Some(n) => n,
        None => infer_num_blocks(&st)?,
    };
    let mut blocks = Vec::with_capacity(total);
    for i in 0..total {
        blocks.push(load_flux_block(&st, device, i, &kc)?);
    }
    let backbone = load_flux_backbone(&st, device, &kc)?;
    Ok(FluxPacks { blocks, backbone })
}

fn must(st: &STFile, key: &str, device: &Device, force: Option<DType>) -> Result<Tensor> {
    load_tensor_to_device(st, key, device, force)
        .map_err(|e| anyhow::anyhow!("{key}: {e}"))
}

fn maybe(st: &STFile, key: &str, device: &Device, force: Option<DType>) -> Result<Option<Tensor>> {
    Ok(match st.tensor(key) {
        Some(_) => Some(load_tensor_to_device(st, key, device, force)?),
        None => None,
    })
}

fn infer_num_blocks(st: &STFile) -> Result<usize> {
    let mut max = None::<usize>;
    for key in st.keys() {
        if let Some(rem) = key.strip_prefix("double_blocks.") {
            if let Some((idx_str, rest)) = rem.split_once('.') {
                if rest.starts_with("img_attn.qkv.weight") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        max = Some(max.map_or(idx, |m| m.max(idx)));
                    }
                }
            }
        }
    }
    max.map(|m| m + 1).ok_or_else(|| anyhow::anyhow!("unable to infer block count"))
}
