#![cfg(feature = "cpu_snapshots")]
use flame_core::conv::Conv2d;
use flame_core::cpu::{
    linear::LinearSnapshot,
    norm::{LayerNormSnapshot, RmsNormSnapshot},
    snapshot::{Bf16CpuSnapshot, F32CpuSnapshot},
};
use flame_core::{DType, Error, Result};

use crate::models::mmdit_blocks::{MMDiTConfig, QkNormKind};

#[derive(Clone, Debug)]
pub struct Conv2dSnapshot {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub groups: usize,
    pub weight: Bf16CpuSnapshot,
    pub bias: Option<Bf16CpuSnapshot>,
}

impl Conv2dSnapshot {
    pub fn from_conv(conv: &Conv2d) -> Result<Self> {
        if conv.weight.dtype() != DType::BF16 || conv.weight.storage_dtype() != DType::BF16 {
            return Err(Error::InvalidInput(
                "Conv2dSnapshot::from_conv expects BF16 weight".into(),
            ));
        }
        let weight = Bf16CpuSnapshot::from_tensor(&conv.weight)?;
        let bias = match &conv.bias {
            Some(b) => Some(Bf16CpuSnapshot::from_tensor(b)?),
            None => None,
        };
        Ok(Self {
            in_channels: conv.config.in_channels,
            out_channels: conv.config.out_channels,
            kernel_size: conv.config.kernel_size,
            stride: conv.config.stride,
            padding: conv.config.padding,
            groups: conv.config.groups,
            weight,
            bias,
        })
    }
}

#[derive(Clone, Debug)]
pub struct QkNormSnapshot {
    pub kind: QkNormKind,
    pub norm_q: Option<LayerNormSnapshot>,
    pub norm_k: Option<LayerNormSnapshot>,
    pub rms_q: Option<RmsNormSnapshot>,
    pub rms_k: Option<RmsNormSnapshot>,
}

#[derive(Clone, Debug)]
pub struct SelfAttentionSnapshot {
    pub num_heads: usize,
    pub head_dim: usize,
    pub q: LinearSnapshot,
    pub k: LinearSnapshot,
    pub v: LinearSnapshot,
    pub proj: Option<LinearSnapshot>,
    pub qk_norm: QkNormSnapshot,
}

#[derive(Clone, Debug)]
pub struct MlpSnapshot {
    pub fc1: LinearSnapshot,
    pub fc2: LinearSnapshot,
}

#[derive(Clone, Debug)]
pub struct BlockSnapshot {
    pub hidden: usize,
    pub num_heads: usize,
    pub pre_only: bool,
    pub self_attn: bool,
    pub norm1: LayerNormSnapshot,
    pub attn: SelfAttentionSnapshot,
    pub attn2: Option<SelfAttentionSnapshot>,
    pub norm2: Option<LayerNormSnapshot>,
    pub mlp: Option<MlpSnapshot>,
    pub modulation: LinearSnapshot,
}

#[derive(Clone, Debug)]
pub struct JointBlockSnapshot {
    pub context: BlockSnapshot,
    pub x: BlockSnapshot,
}

#[derive(Clone, Debug)]
pub struct PatchEmbedSnapshot {
    pub proj: Conv2dSnapshot,
    pub flatten: bool,
    pub dynamic_img_pad: bool,
    pub patch_size: usize,
}

#[derive(Clone, Debug)]
pub struct TimestepEmbedderSnapshot {
    pub linear1: LinearSnapshot,
    pub linear2: LinearSnapshot,
    pub frequency_embedding_size: usize,
}

#[derive(Clone, Debug)]
pub struct VectorEmbedderSnapshot {
    pub linear1: LinearSnapshot,
    pub linear2: LinearSnapshot,
    pub input_dim: usize,
}

#[derive(Clone, Debug)]
pub struct FinalLayerSnapshot {
    pub norm: LayerNormSnapshot,
    pub modulation: LinearSnapshot,
    pub proj: LinearSnapshot,
    pub patch_size: usize,
    pub out_channels: usize,
}

#[derive(Clone, Debug)]
pub struct MmditCpuSnapshot {
    pub config: MMDiTConfig,
    pub patch_embed: PatchEmbedSnapshot,
    pub timestep_embedder: TimestepEmbedderSnapshot,
    pub vector_embedder: Option<VectorEmbedderSnapshot>,
    pub context_embedder: LinearSnapshot,
    pub blocks: Vec<JointBlockSnapshot>,
    pub final_layer: FinalLayerSnapshot,
    pub pos_frequencies: F32CpuSnapshot,
}

impl MmditCpuSnapshot {
    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    pub fn load_into(&self, model: &mut crate::models::mmdit_blocks::MMDiT) -> Result<()> {
        model.apply_cpu_snapshot(self)
    }
}
