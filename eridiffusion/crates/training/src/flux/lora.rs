//! Flux LoRA helpers (image-branch adapters).
//! Provides stable storage for low-rank adapters on Q/K/V/O and MLP FC1/FC2 per transformer block.

use std::collections::HashMap;

use anyhow::{ensure, Result};
use eridiffusion_core::Device as EriDevice;
use eridiffusion_models::devtensor::{randn_scaled_on, shape2, zeros_on};
use flame_core::{DType, Parameter, Tensor};
use serde::{Deserialize, Serialize};

/// Low-rank adapter attached to a single linear transformation.
#[derive(Clone)]
pub struct FluxLoraLinear {
    pub a: Parameter, // [rank, in]
    pub b: Parameter, // [out, rank]
    pub rank: usize,
    pub alpha: f32,
}

impl FluxLoraLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        device: flame_core::Device,
    ) -> Result<Self> {
        let rank = rank.max(1);
        let scale = (1.0f32 / rank as f32).sqrt();
        let eri_device = EriDevice::Cuda(device.ordinal());
        let a = randn_scaled_on(
            shape2(rank as i64, in_dim as i64),
            &eri_device,
            DType::BF16,
            0.0,
            scale,
            None,
        )?
        .requires_grad_(true);
        let b = zeros_on(shape2(out_dim as i64, rank as i64), &eri_device, DType::BF16)?
            .requires_grad_(true);
        Ok(Self { a: Parameter::new(a), b: Parameter::new(b), rank, alpha })
    }

    /// Compute the low-rank delta for input tokens `[B,T,in]` (returns `[B,T,out]`).
    pub fn forward_delta(&self, tokens: &Tensor) -> Result<Tensor> {
        let dims = tokens.shape().dims().to_vec();
        ensure!(dims.len() == 3, "expected tokens [B,T,IN], got {:?}", dims);
        let (b, t, in_dim) = (dims[0], dims[1], dims[2]);
        let x = tokens.reshape(&[b * t, in_dim])?; // [BS, IN]

        let a = self.a.tensor()?; // [rank, IN]
        let b_w = self.b.tensor()?; // [OUT, rank]

        let a_t = a.transpose_dims(0, 1)?; // [IN, rank]
        let u = x.matmul(&a_t)?; // [BS, rank]
        let b_t = b_w.transpose_dims(0, 1)?; // [rank, OUT]
        let delta_flat = u.matmul(&b_t)?; // [BS, OUT]
        let scale = self.alpha / self.rank as f32;
        let delta = delta_flat.affine(scale, 0.0f32)?;
        Ok(delta.reshape(&[b, t, b_w.shape().dims()[0]])?)
    }

    pub fn parameters(&self) -> [Parameter; 2] {
        [self.a.clone(), self.b.clone()]
    }
}

/// Per-block LoRA adapters (image branch only).
#[derive(Clone, Default)]
pub struct FluxBlockLora {
    pub q: Option<FluxLoraLinear>,
    pub k: Option<FluxLoraLinear>,
    pub v: Option<FluxLoraLinear>,
    pub o: Option<FluxLoraLinear>,
    pub fc1: Option<FluxLoraLinear>,
    pub fc2: Option<FluxLoraLinear>,
}

impl FluxBlockLora {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn named_parameters(&self, block_id: usize) -> HashMap<String, Parameter> {
        let mut map = HashMap::new();
        let prefix = format!("block{:02}", block_id);
        if let Some(q) = &self.q {
            let [a, b] = q.parameters();
            map.insert(format!("{}.attn.q.lora.A", prefix), a);
            map.insert(format!("{}.attn.q.lora.B", prefix), b);
        }
        if let Some(k) = &self.k {
            let [a, b] = k.parameters();
            map.insert(format!("{}.attn.k.lora.A", prefix), a);
            map.insert(format!("{}.attn.k.lora.B", prefix), b);
        }
        if let Some(v) = &self.v {
            let [a, b] = v.parameters();
            map.insert(format!("{}.attn.v.lora.A", prefix), a);
            map.insert(format!("{}.attn.v.lora.B", prefix), b);
        }
        if let Some(o) = &self.o {
            let [a, b] = o.parameters();
            map.insert(format!("{}.attn.o.lora.A", prefix), a);
            map.insert(format!("{}.attn.o.lora.B", prefix), b);
        }
        if let Some(fc1) = &self.fc1 {
            let [a, b] = fc1.parameters();
            map.insert(format!("{}.mlp.fc1.lora.A", prefix), a);
            map.insert(format!("{}.mlp.fc1.lora.B", prefix), b);
        }
        if let Some(fc2) = &self.fc2 {
            let [a, b] = fc2.parameters();
            map.insert(format!("{}.mlp.fc2.lora.A", prefix), a);
            map.insert(format!("{}.mlp.fc2.lora.B", prefix), b);
        }
        map
    }

    fn parameters_in_order(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::new();
        let push = |list: &mut Vec<(String, Parameter)>, name: String, param: &FluxLoraLinear| {
            let [a, b] = param.parameters();
            list.push((format!("{}.A", name), a));
            list.push((format!("{}.B", name), b));
        };
        if let Some(q) = &self.q {
            push(&mut out, "attn.q.lora".into(), q);
        }
        if let Some(k) = &self.k {
            push(&mut out, "attn.k.lora".into(), k);
        }
        if let Some(v) = &self.v {
            push(&mut out, "attn.v.lora".into(), v);
        }
        if let Some(o) = &self.o {
            push(&mut out, "attn.o.lora".into(), o);
        }
        if let Some(fc1) = &self.fc1 {
            push(&mut out, "mlp.fc1.lora".into(), fc1);
        }
        if let Some(fc2) = &self.fc2 {
            push(&mut out, "mlp.fc2.lora".into(), fc2);
        }
        out
    }
}

/// Collection of LoRA adapters for all blocks.
#[derive(Clone, Default)]
pub struct FluxLoraHandles {
    pub per_block: Vec<FluxBlockLora>,
}

impl FluxLoraHandles {
    pub fn new() -> Self {
        Self { per_block: Vec::new() }
    }

    pub fn push_block(&mut self, block: FluxBlockLora) {
        self.per_block.push(block);
    }

    pub fn block(&self, idx: usize) -> Option<&FluxBlockLora> {
        self.per_block.get(idx)
    }

    pub fn block_mut(&mut self, idx: usize) -> Option<&mut FluxBlockLora> {
        self.per_block.get_mut(idx)
    }

    pub fn len(&self) -> usize {
        self.per_block.len()
    }

    pub fn is_empty(&self) -> bool {
        self.per_block.is_empty()
    }

    pub fn blocks(&self) -> &[FluxBlockLora] {
        &self.per_block
    }

    pub fn blocks_mut(&mut self) -> &mut [FluxBlockLora] {
        &mut self.per_block
    }

    pub fn all_params(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        for block in &self.per_block {
            if let Some(q) = &block.q {
                out.extend(q.parameters());
            }
            if let Some(k) = &block.k {
                out.extend(k.parameters());
            }
            if let Some(v) = &block.v {
                out.extend(v.parameters());
            }
            if let Some(o) = &block.o {
                out.extend(o.parameters());
            }
            if let Some(fc1) = &block.fc1 {
                out.extend(fc1.parameters());
            }
            if let Some(fc2) = &block.fc2 {
                out.extend(fc2.parameters());
            }
        }
        out
    }

    pub fn parameters_in_order(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        for (_idx, block) in self.per_block.iter().enumerate() {
            for (_, param) in block.parameters_in_order() {
                out.push(param);
            }
        }
        out
    }

    pub fn names_in_order(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (idx, block) in self.per_block.iter().enumerate() {
            let prefix = format!("block{:02}", idx);
            for (suffix, _) in block.parameters_in_order() {
                let name = match suffix.as_str() {
                    s if s.starts_with("attn") || s.starts_with("mlp") => {
                        format!("{}.{}", prefix, suffix)
                    }
                    _ => format!("{}.{}", prefix, suffix),
                };
                out.push(name);
            }
        }
        out
    }

    pub fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut map = HashMap::new();
        for (idx, block) in self.per_block.iter().enumerate() {
            for (name, param) in block.named_parameters(idx) {
                map.insert(name, param);
            }
        }
        map
    }
}

/// Serialized LoRA configuration for Flux.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FluxLoraSpec {
    pub rank: usize,
    pub alpha: f32,
    #[serde(default)]
    pub zero_init: bool,
}
