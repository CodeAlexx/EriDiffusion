//! Stable LoRA key enumeration for Flux blocks.
//! Produces canonical parameter IDs for Q/K/V/O and MLP FC layers to ensure adapter states
//! can be saved/loaded consistently across runs.

#![allow(dead_code)]
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd)]
pub enum LoraSiteKind {
    QProj,
    KProj,
    VProj,
    OProj,
    Fc1,
    Fc2,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd)]
pub struct LoraKey {
    /// e.g., "block_06.attn.q"
    pub name: String,
    pub kind: LoraSiteKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoraSpec {
    pub rank: usize,
    pub alpha: f32,
    pub zero_init: bool,
}

/// Returns a deterministic, sorted map from logical site -> stable key string.
pub fn enumerate_flux_lora_keys(num_blocks: usize, with_mid: bool) -> BTreeMap<LoraKey, String> {
    let mut map = BTreeMap::<LoraKey, String>::new();

    // Encoder/decoder blocks
    for i in 0..num_blocks {
        let bi = format!("{i:02}");
        map.insert(
            LoraKey { name: format!("block_{bi}.attn.q"), kind: LoraSiteKind::QProj },
            format!("blocks.{bi}.attn.q"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.attn.k"), kind: LoraSiteKind::KProj },
            format!("blocks.{bi}.attn.k"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.attn.v"), kind: LoraSiteKind::VProj },
            format!("blocks.{bi}.attn.v"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.attn.o"), kind: LoraSiteKind::OProj },
            format!("blocks.{bi}.attn.o"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.mlp.fc1"), kind: LoraSiteKind::Fc1 },
            format!("blocks.{bi}.mlp.fc1"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.mlp.fc2"), kind: LoraSiteKind::Fc2 },
            format!("blocks.{bi}.mlp.fc2"),
        );
    }

    if with_mid {
        // Middle block (often present in UNet/DiT-like stacks)
        let bi = "mid";
        map.insert(
            LoraKey { name: format!("block_{bi}.attn.q"), kind: LoraSiteKind::QProj },
            format!("blocks.{bi}.attn.q"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.attn.k"), kind: LoraSiteKind::KProj },
            format!("blocks.{bi}.attn.k"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.attn.v"), kind: LoraSiteKind::VProj },
            format!("blocks.{bi}.attn.v"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.attn.o"), kind: LoraSiteKind::OProj },
            format!("blocks.{bi}.attn.o"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.mlp.fc1"), kind: LoraSiteKind::Fc1 },
            format!("blocks.{bi}.mlp.fc1"),
        );
        map.insert(
            LoraKey { name: format!("block_{bi}.mlp.fc2"), kind: LoraSiteKind::Fc2 },
            format!("blocks.{bi}.mlp.fc2"),
        );
    }

    map
}

/// Compose a final tensor key inside safetensors for a given LoRA site and A/B tensor part.
pub fn compose_lora_tensor_key(stable_key: &str, part: &str) -> String {
    // Example final keys:
    //   lora.blocks.06.attn.q.A  and  lora.blocks.06.attn.q.B
    format!("lora.{stable_key}.{part}")
}
