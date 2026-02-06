//! Helpers for working with cached Flux latents & embeddings.
//! The actual preprocessing pipeline is deferred; this module provides
//! lightweight loaders that map cached safetensors into GPU BF16 tensors.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use bytemuck::cast_slice;
use eridiffusion_core::Device;
use eridiffusion_models::devtensor::{tensor_from_slice_on, BF16, F32_};
use flame_core::{Shape, Tensor};
use half::{bf16, f16};
use safetensors::{tensor::Dtype as SafeDtype, SafeTensors};
use serde::{Deserialize, Serialize};

/// Preprocessed item stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessedFluxItem {
    pub latents_path: PathBuf,
    pub t5_embeds_path: PathBuf,
    pub clip_pooled_path: PathBuf,
    pub caption: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Dataset wrapper backed by cached safetensors files.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DatasetStats {
    pub len: usize,
    pub total_latent_bytes: u64,
    pub total_t5_bytes: u64,
    pub total_clip_bytes: u64,
    /// Optional SHA-256 of the manifest that produced this dataset (if known).
    pub manifest_hash: Option<String>,
}

pub struct PreprocessedFluxDataset {
    items: Vec<PreprocessedFluxItem>,
    device: Device,
    stats: Option<DatasetStats>,
}

impl PreprocessedFluxDataset {
    pub fn new(
        items: Vec<PreprocessedFluxItem>,
        device: Device,
        stats: Option<DatasetStats>,
    ) -> Self {
        Self { items, device, stats }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn stats(&self) -> Option<&DatasetStats> {
        self.stats.as_ref()
    }

    pub fn get_item(&self, idx: usize) -> anyhow::Result<PreprocessedFluxBatch> {
        let item = &self.items[idx];
        let latents = self.load_tensor(&item.latents_path)?;
        let t5_embeds = self.load_tensor(&item.t5_embeds_path)?;
        let clip_pooled = self.load_tensor(&item.clip_pooled_path)?;
        Ok(PreprocessedFluxBatch {
            latents,
            t5_embeds,
            clip_pooled,
            caption: item.caption.clone(),
            metadata: item.metadata.clone(),
        })
    }

    fn load_tensor(&self, path: &Path) -> anyhow::Result<Tensor> {
        let bytes = fs::read(path)?;
        let safetensors = SafeTensors::deserialize(&bytes)?;

        // Try common key names: "data", "latent", "text_embed", or first available tensor
        let names = safetensors.names();
        let key = if names.iter().any(|&s| s == "data") {
            "data"
        } else if names.iter().any(|&s| s == "latent") {
            "latent"
        } else if names.iter().any(|&s| s == "text_embed") {
            "text_embed"
        } else {
            names
                .first()
                .ok_or_else(|| anyhow!("safetensors file {} has no tensors", path.display()))?
                .as_str()
        };

        let view = safetensors.tensor(key)?;
        let shape_dims: Vec<usize> = view.shape().iter().copied().collect();

        let data_f32: Vec<f32> = match view.dtype() {
            SafeDtype::F32 => cast_slice::<u8, f32>(view.data()).to_vec(),
            SafeDtype::BF16 => view
                .data()
                .chunks_exact(2)
                .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                .collect(),
            SafeDtype::F16 => view
                .data()
                .chunks_exact(2)
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                .collect(),
            other => return Err(anyhow!("Unsupported safetensor dtype: {:?}", other)),
        };

        let mut tensor_f32 =
            tensor_from_slice_on(&data_f32, Shape::from_dims(&shape_dims), &self.device, F32_)?;

        // Squeeze batch dimension if present [1, seq_len, hidden] -> [seq_len, hidden]
        if shape_dims.len() == 3 && shape_dims[0] == 1 {
            tensor_f32 =
                tensor_f32.squeeze(Some(0)).map_err(|e| anyhow!("squeeze failed: {}", e))?;
        }

        Ok(tensor_f32.to_dtype(BF16)?)
    }
}

/// A single cached sample (latents + text embeddings).
pub struct PreprocessedFluxBatch {
    pub latents: Tensor,
    pub t5_embeds: Tensor,
    pub clip_pooled: Tensor,
    pub caption: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Simple diagnostic helper retained from the legacy preprocessor.
pub fn print_memory_savings(dataset_size: usize) {
    println!("\n💾 Memory Savings with Preprocessing:");
    println!("─────────────────────────────────────");

    let vae_mem = 1.7; // GB
    let t5_mem = 11.0; // GB
    let clip_mem = 0.5; // GB
    let flux_mem = 12.0; // GB
    let total_without = vae_mem + t5_mem + clip_mem + flux_mem;

    println!("Without preprocessing:");
    println!("  VAE:        {:>5.1} GB", vae_mem);
    println!("  T5-XXL:     {:>5.1} GB", t5_mem);
    println!("  CLIP-L:     {:>5.1} GB", clip_mem);
    println!("  Flux:       {:>5.1} GB", flux_mem);
    println!("  Total:      {:>5.1} GB ❌ Won't fit!", total_without);

    let flux_only = 12.0; // GB
    let gradients = 3.0; // GB
    let optimizer = 6.0; // GB
    let activations = 2.0; // GB
    let total_with = flux_only + gradients + optimizer + activations;

    println!("\nWith preprocessing:");
    println!("  Flux only:  {:>5.1} GB", flux_only);
    println!("  Gradients:  {:>5.1} GB", gradients);
    println!("  Optimizer:  {:>5.1} GB", optimizer);
    println!("  Activations:{:>5.1} GB", activations);
    println!("  Total:      {:>5.1} GB ✅ Fits in 24GB!", total_with);

    let images_size = dataset_size as f64 * 1024.0 * 1024.0 * 3.0 / 1e9;
    let latents_size = dataset_size as f64 * 128.0 * 128.0 * 16.0 * 4.0 / 1e9;
    let text_size = dataset_size as f64 * (512.0 * 4096.0 + 768.0) * 4.0 / 1e9;

    println!("\n📈 Dataset preprocessing:");
    println!("  Raw images:    {:>6.1} GB", images_size);
    println!("  → Latents:     {:>6.1} GB", latents_size);
    println!("  → Text embeds: {:>6.1} GB", text_size);
    println!("  Disk total:    {:>6.1} GB", latents_size + text_size);
}
