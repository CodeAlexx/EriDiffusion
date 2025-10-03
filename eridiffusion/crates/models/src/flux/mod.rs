pub mod dtype_policy;
pub mod op_dtype;
pub mod debug_asserts;

pub mod ae;
pub mod keys;
pub mod loader;
pub mod canonicalize;
pub mod dit;
pub mod keymap;
pub mod loader_utils;
pub mod lora;
pub mod text;
pub mod timestep;
pub mod weight_utils;

pub use loader::{FluxBlockWeights, FluxBackboneWeights, FluxPacks, load_flux_packs, load_flux_packs_with, load_flux_block};
pub use keys::{KeyConv, Schema as KeySchema, default_keyconv};

use anyhow::{ensure, Result};
use dit::{DiTBlock, WeightPack};
use dtype_policy::MatmulDTypePolicy;
use eridiffusion_core::Device as CoreDevice;
use flame_core::{DType, Device, Shape, Tensor};
use safetensors::Dtype as SafeDtype;
use std::sync::atomic::{AtomicUsize, Ordering};
use timestep::timestep_embedding;

pub static FORWARD_STEP: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Minimal Flux DiT config
#[derive(Clone, Debug)]
pub struct FluxConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub param_dtype: DType,
    pub matmul_policy: MatmulDTypePolicy,
}

impl Default for FluxConfig {
    fn default() -> Self {
        Self {
            hidden: 0,
            heads: 0,
            layers: 0,
            param_dtype: DType::BF16,
            matmul_policy: MatmulDTypePolicy::default(),
        }
    }
}

/// Simplified Flux DiT model: positional encoding + [Attn -> MLP] x N with residuals.
pub struct FluxModel {
    cfg: FluxConfig,
    pub device: Device,
    pub param_dtype: DType,
    pub matmul_policy: MatmulDTypePolicy,
    blocks: Vec<DiTBlock>,
    base_frozen: bool,
}

impl FluxModel {
    pub fn new(mut cfg: FluxConfig, device: Device, param_dtype: DType) -> Result<Self> {
        // Initialize empty weight packs; callers should replace with from_packs
        let mut blocks = Vec::with_capacity(cfg.layers);
        cfg.param_dtype = param_dtype;
        let matmul_policy = cfg.matmul_policy;
        for _ in 0..cfg.layers {
            let dummy = WeightPack {
                wq: Tensor::zeros_dtype(
                    Shape::from_dims(&[cfg.hidden, cfg.hidden]),
                    param_dtype,
                    device.cuda_device().clone(),
                )?,
                wk: Tensor::zeros_dtype(
                    Shape::from_dims(&[cfg.hidden, cfg.hidden]),
                    param_dtype,
                    device.cuda_device().clone(),
                )?,
                wv: Tensor::zeros_dtype(
                    Shape::from_dims(&[cfg.hidden, cfg.hidden]),
                    param_dtype,
                    device.cuda_device().clone(),
                )?,
                wo: Tensor::zeros_dtype(
                    Shape::from_dims(&[cfg.hidden, cfg.hidden]),
                    param_dtype,
                    device.cuda_device().clone(),
                )?,
                fc1: Tensor::zeros_dtype(
                    Shape::from_dims(&[cfg.hidden, cfg.hidden * 4]),
                    param_dtype,
                    device.cuda_device().clone(),
                )?,
                fc2: Tensor::zeros_dtype(
                    Shape::from_dims(&[cfg.hidden * 4, cfg.hidden]),
                    param_dtype,
                    device.cuda_device().clone(),
                )?,
            };
            blocks.push(DiTBlock::new(
                cfg.hidden,
                cfg.heads,
                dummy,
                param_dtype,
                matmul_policy,
            )?);
        }
        let param_dtype_final = param_dtype;
        Ok(Self {
            cfg,
            device,
            param_dtype: param_dtype_final,
            matmul_policy,
            blocks,
            base_frozen: false,
        })
    }

    /// Construct from validated weight packs.
    pub fn from_packs(
        mut cfg: FluxConfig,
        device: Device,
        dtype: DType,
        packs: Vec<WeightPack>,
    ) -> Result<Self> {
        ensure!(packs.len() == cfg.layers, "packs must match layer count");
        let mut blocks = Vec::with_capacity(cfg.layers);
        cfg.param_dtype = dtype;
        let matmul_policy = cfg.matmul_policy;
        for p in packs {
            blocks.push(DiTBlock::new(
                cfg.hidden,
                cfg.heads,
                p,
                cfg.param_dtype,
                matmul_policy,
            )?);
        }
        let param_dtype = cfg.param_dtype;
        Ok(Self {
            cfg,
            device,
            param_dtype,
            matmul_policy,
            blocks,
            base_frozen: false,
        })
    }

    pub fn hidden_dim(&self) -> usize {
        self.cfg.hidden
    }
    pub fn num_heads(&self) -> usize {
        self.cfg.heads
    }
    pub fn num_layers(&self) -> usize {
        self.cfg.layers
    }

    /// Strictly load Flux weight packs from a safetensors file, validating keys and shapes.
    pub fn from_safetensors_strict_packs(
        path: &std::path::Path,
        mut cfg: FluxConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let core_device = eridiffusion_core::Device::from_flame_cuda(device.cuda_device());
        let packs = loader::load_flux_packs(path, &core_device)?;
        let hidden = packs
            .blocks
            .first()
            .map(|bw| bw.q_w.shape().dims()[1] as usize)
            .unwrap_or(cfg.hidden);
        cfg.hidden = hidden;
        cfg.layers = packs.blocks.len();
        cfg.param_dtype = dtype;
        let blocks = packs
            .blocks
            .into_iter()
            .map(|bw| WeightPack {
                wq: bw.q_w,
                wk: bw.k_w,
                wv: bw.v_w,
                wo: bw.o_w,
                fc1: bw.fc1_w,
                fc2: bw.fc2_w,
            })
            .collect();
        Self::from_packs(cfg, device, dtype, blocks)
}

    /// Load from a safetensors path (strictly checks file presence). Real loader would map keys.
    pub fn from_safetensors_strict(
        path: &str,
        cfg: FluxConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        if !std::path::Path::new(path).exists() {
            anyhow::bail!("Flux weights not found at {}", path);
        }
        let mut cfg = cfg;
        cfg.param_dtype = dtype;
        Self::new(cfg, device, dtype)
    }

    /// Forward contract: x [B,Seq,Hid], returns same shape.
    pub fn forward(&self, x: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        ensure!(
            dims.len() == 3 && dims[2] == self.cfg.hidden,
            "Flux forward expects [B,Seq,{}]",
            self.cfg.hidden
        );
        let (b, seq, hid) = (dims[0], dims[1], dims[2]);
        let step = FORWARD_STEP.load(Ordering::Relaxed);
        // Positional encoding (sin/cos) add
        let core_dev = CoreDevice::from_flame_cuda(self.device.cuda_device());
        let timesteps_f32 = timesteps.to_dtype(DType::F32)?;
        let pos = timestep_embedding(&timesteps_f32, hid, &core_dev, DType::F32)?; // [B,Hid] in F32
        let pos_bsh = pos
            .reshape(&[b, 1, hid])?
            .broadcast_to(&Shape::from_dims(&[b, seq, hid]))?;
        let pos_bf16 = if pos_bsh.dtype() == DType::BF16 {
            pos_bsh
        } else {
            pos_bsh.to_dtype(DType::BF16)?
        };
        let x_bf16 = if x.dtype() == DType::BF16 { x.clone() } else { x.to_dtype(DType::BF16)? };
        let y_added_bf16 = flame_core::ops::elt::add_same_dtype(&x_bf16, &pos_bf16)?;
        let mut y = if self.param_dtype == DType::BF16 {
            y_added_bf16
        } else {
            y_added_bf16.to_dtype(self.param_dtype)?
        };
        for (idx, blk) in self.blocks.iter().enumerate() {
            y = blk.forward_with_probe(&y, step, idx)?;
        }
        Ok(y)
    }

    pub fn freeze_base(&mut self) {
        self.base_frozen = true;
    }
    pub fn is_base_frozen(&self) -> bool {
        self.base_frozen
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum PackKind {
    Canonical,
    DoubleBlocks,
    SingleBlocks,
}

#[allow(dead_code)]
fn detect_pack_kind<I>(keys: I) -> PackKind
where
    I: IntoIterator<Item = String>,
{
    let mut has_blocks = false;
    let mut has_double = false;
    let mut has_single = false;
    for key in keys {
        if key.starts_with("block") {
            has_blocks = true;
        } else if key.starts_with("double_blocks.") {
            has_double = true;
        } else if key.starts_with("single_blocks.") {
            has_single = true;
        }
    }
    if has_blocks {
        PackKind::Canonical
    } else if has_double {
        PackKind::DoubleBlocks
    } else if has_single {
        PackKind::SingleBlocks
    } else {
        PackKind::Canonical
    }
}

#[allow(dead_code)]
fn map_safetensors_dtype(dt: SafeDtype) -> Result<DType> {
    Ok(match dt {
        SafeDtype::F32 => DType::F32,
        SafeDtype::F16 => DType::F16,
        SafeDtype::BF16 => DType::BF16,
        SafeDtype::I32 => DType::I32,
        SafeDtype::I64 => DType::I64,
        SafeDtype::U32 => DType::U32,
        SafeDtype::U8 => DType::U8,
        SafeDtype::BOOL => DType::Bool,
        other => anyhow::bail!("unsupported dtype: {:?}", other),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devtensor::{randn_on, tensor_from_vec_on, zeros_on, shape3, shape2, shape1};
    use eridiffusion_core::Device as EDevice;
    #[test]
    fn forward_contract_shape() -> anyhow::Result<()> {
        let dev_flame = flame_core::Device::cuda(0)?;
        let dev = EDevice::Cuda(0);
        let dtype = DType::BF16;
        let cfg = FluxConfig {
            hidden: 64,
            heads: 4,
            layers: 2,
            param_dtype: DType::BF16,
            matmul_policy: MatmulDTypePolicy::MatchParams,
        };
        let make_pack = || -> anyhow::Result<WeightPack> {
            let zeros = |rows: usize, cols: usize| -> anyhow::Result<Tensor> {
                zeros_on(shape2(rows as i64, cols as i64), &dev, dtype)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))
            };
            Ok(WeightPack {
                wq: zeros(cfg.hidden, cfg.hidden)?,
                wk: zeros(cfg.hidden, cfg.hidden)?,
                wv: zeros(cfg.hidden, cfg.hidden)?,
                wo: zeros(cfg.hidden, cfg.hidden)?,
                fc1: zeros(cfg.hidden, cfg.hidden * 4)?,
                fc2: zeros(cfg.hidden * 4, cfg.hidden)?,
            })
        };
        let mut packs = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            packs.push(make_pack()?);
        }
        let model = FluxModel::from_packs(cfg.clone(), dev_flame.clone(), dtype, packs)?;
        let x = randn_on(shape3(2, 17, cfg.hidden as i64), &dev, dtype, None)?;
        let t = tensor_from_vec_on(vec![10.0f32, 30.0f32], shape1(2), &dev, DType::F32)?;
        let out = model.forward(&x, &t)?;
        assert_eq!(out.shape().dims(), &[2, 17, cfg.hidden]);
        Ok(())
    }
}
