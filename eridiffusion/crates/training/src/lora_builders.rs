use std::collections::HashMap;

use eridiffusion_core::Result;

// ---- Target key lists (stable order) ----

pub fn flux_lora_keys() -> &'static [&'static str] {
    &["attn.q", "attn.k", "attn.v", "attn.o", "mlp.fc1", "mlp.fc2"]
}

pub fn sdxl_lora_keys() -> &'static [&'static str] {
    &[
        "xattn.q", "xattn.k", "xattn.v", "xattn.o", "sattn.q", "sattn.k", "sattn.v", "sattn.o",
        "ffn.fc1", "ffn.fc2",
    ]
}

pub fn sd35_lora_keys() -> &'static [&'static str] {
    &[
        "txt_attn.q",
        "txt_attn.k",
        "txt_attn.v",
        "txt_attn.o",
        "img_attn.q",
        "img_attn.k",
        "img_attn.v",
        "img_attn.o",
        "mlp.fc1",
        "mlp.fc2",
    ]
}

// ---- Shape provider trait ----
// Implement this per model to tell builders the (in,out) dims for a given key.
pub trait LoraShapeProvider {
    /// Return (in_dim, out_dim) for the given target key, or None if unsupported.
    fn in_out(&self, key: &str) -> Option<(usize, usize)>;
}

// ---- Spec type (decoupled from concrete LoRA impl) ----
#[derive(Clone, Debug)]
pub struct LoraSpec {
    pub key: &'static str,
    pub in_d: usize,
    pub out_d: usize,
    pub rank: usize,
    pub alpha: f32,
    pub zero_init: bool,
}

/// Build specs for the given key list using a provided shape provider.
pub fn build_specs<P: LoraShapeProvider>(
    provider: &P,
    keys: &[&'static str],
    rank: usize,
    alpha: f32,
    zero_init: bool,
) -> Result<Vec<LoraSpec>> {
    let mut out = Vec::with_capacity(keys.len());
    for &k in keys {
        if let Some((in_d, out_d)) = provider.in_out(k) {
            out.push(LoraSpec { key: k, in_d, out_d, rank, alpha, zero_init });
        }
    }
    Ok(out)
}

/// Build a stable key→index map for vectors ordered by `keys`.
pub fn build_index(keys: &[&'static str]) -> HashMap<&'static str, usize> {
    let mut idx = HashMap::with_capacity(keys.len());
    for (i, &k) in keys.iter().enumerate() {
        idx.insert(k, i);
    }
    idx
}
