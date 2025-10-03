//! Flux weight provider backed by safetensors + GPU BF16 tensors.
//! Replaces the placeholder StrictWeights implementation with the real streaming loader.

use std::{collections::HashSet, path::Path, sync::Arc};

use anyhow::{anyhow, ensure, Result};
use eridiffusion_core::Device;
use eridiffusion_models::common_io::{load_tensor_to_device, STFile};
use flame_core::Tensor;

use crate::streaming::{DeviceWeights, WeightProvider};

const LOGICAL_SUFFIXES: &[&str] = &[
    // Image attention (logical projections)
    ".img_attn.qkv.weight",
    ".img_attn.qkv.bias",
    ".img_attn.proj.weight",
    ".img_attn.proj.bias",
    ".img_attn.norm.key_norm.scale",
    ".img_attn.norm.query_norm.scale",
    // Image MLP + modulation
    ".img_mlp.0.weight",
    ".img_mlp.0.bias",
    ".img_mlp.2.weight",
    ".img_mlp.2.bias",
    ".img_mod.lin.weight",
    ".img_mod.lin.bias",
    // Text attention
    ".txt_attn.qkv.weight",
    ".txt_attn.qkv.bias",
    ".txt_attn.proj.weight",
    ".txt_attn.proj.bias",
    ".txt_attn.norm.key_norm.scale",
    ".txt_attn.norm.query_norm.scale",
    // Text MLP + modulation
    ".txt_mlp.0.weight",
    ".txt_mlp.0.bias",
    ".txt_mlp.2.weight",
    ".txt_mlp.2.bias",
    ".txt_mod.lin.weight",
    ".txt_mod.lin.bias",
];

fn infer_num_blocks(st: &STFile) -> Result<usize> {
    let mut max_idx = None::<usize>;
    for name in st.keys() {
        if let Some(rem) = name.strip_prefix("double_blocks.") {
            if let Some((idx, rest)) = rem.split_once('.') {
                if rest.starts_with("img_attn.qkv.weight") {
                    if let Ok(i) = idx.parse::<usize>() {
                        max_idx = Some(max_idx.map_or(i, |m| m.max(i)));
                    }
                }
            }
        }
    }
    max_idx.map(|m| m + 1).ok_or_else(|| anyhow!("unable to infer Flux block count"))
}

fn gather_block_keys(st: &STFile, num_blocks: usize) -> Vec<Vec<String>> {
    let mut per_block: Vec<Vec<String>> = vec![Vec::new(); num_blocks];
    let mut seen: Vec<HashSet<String>> = vec![HashSet::new(); num_blocks];

    for idx in 0..num_blocks {
        let base = format!("double_blocks.{idx}");
        for suffix in LOGICAL_SUFFIXES {
            let logical = format!("{base}{suffix}");
            per_block[idx].push(logical.clone());
            seen[idx].insert(logical);
        }
    }

    for key in st.keys() {
        if let Some(rem) = key.strip_prefix("double_blocks.") {
            if let Some((idx_str, _rest)) = rem.split_once('.') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx < num_blocks {
                        // Skip fused qkv keys; logical projections cover them via resolve_qkv_key
                        if key.ends_with("img_attn.qkv.weight")
                            || key.ends_with("img_attn.qkv.bias")
                            || key.ends_with("txt_attn.qkv.weight")
                            || key.ends_with("txt_attn.qkv.bias")
                        {
                            continue;
                        }
                        if !seen[idx].contains(&key) {
                            per_block[idx].push(key.clone());
                            seen[idx].insert(key.clone());
                        }
                    }
                }
            }
        }
    }

    for keys in per_block.iter_mut() {
        keys.sort();
    }
    per_block
}

pub struct FluxWeightProvider {
    st: Arc<STFile>,
    device: Device,
    num_blocks: usize,
    block_keys: Vec<Vec<String>>,
}

impl FluxWeightProvider {
    pub fn from_path(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let st = Arc::new(STFile::open(path)?);
        let num_blocks = infer_num_blocks(&st)?;
        let block_keys = gather_block_keys(&st, num_blocks);
        Ok(Self { st, device, num_blocks, block_keys })
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn block_keys(&self, block: usize) -> &[String] {
        &self.block_keys[block]
    }

    pub fn named_block_tensors(&self, block: usize) -> Result<Vec<(String, Tensor)>> {
        let tensors = self.tensors_for_keys(block)?;
        let keys = self.block_keys(block);
        ensure!(
            keys.len() == tensors.len(),
            "block {} key/tensor mismatch {} vs {}",
            block,
            keys.len(),
            tensors.len()
        );
        Ok(keys.iter().cloned().zip(tensors.into_iter()).collect())
    }

    fn load_tensor(&self, key: &str) -> Result<Tensor> {
        load_tensor_to_device(
            &self.st,
            key,
            &self.device,
            Some(eridiffusion_models::devtensor::BF16),
        )
        .map_err(|e| anyhow!("{key}: {e}"))
    }

    fn tensors_for_keys(&self, block: usize) -> Result<Vec<Tensor>> {
        if block >= self.block_keys.len() {
            return Err(anyhow!("missing block index {block}"));
        }
        let mut out = Vec::with_capacity(self.block_keys[block].len());
        for key in &self.block_keys[block] {
            out.push(self.load_tensor(key)?);
        }
        Ok(out)
    }
}

impl WeightProvider for FluxWeightProvider {
    fn load_block_to_gpu(&self, block_id: usize) -> Result<DeviceWeights> {
        if block_id >= self.num_blocks {
            return Err(anyhow!("invalid flux block index {block_id}/{n}", n = self.num_blocks));
        }
        let tensors = self.tensors_for_keys(block_id)?;
        Ok(DeviceWeights { tensors })
    }

    fn load_head_to_gpu(&self) -> Result<DeviceWeights> {
        Ok(DeviceWeights { tensors: Vec::new() })
    }

    fn prefetch_block(&self, _block_id: usize) -> Result<()> {
        Ok(())
    }

    fn release_block(&self, _block_id: isize) -> Result<()> {
        Ok(())
    }
}
