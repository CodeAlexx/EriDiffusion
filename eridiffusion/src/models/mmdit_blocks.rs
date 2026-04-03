//! SD3.5 MMDiT blocks ported from the canonical C++ implementation.
//!
//! Reference: `/home/alex/codex-text/stable-diffusion.cpp/mmdit.hpp`.
//! We mirror the structure of the C++ `DismantledBlock` / `JointBlock`
//! pipeline so that the trainer matches the inference harness while
//! preserving STRICT_BF16 invariants.

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
use crate::models::mmdit_cpu::{
    BlockSnapshot, Conv2dSnapshot, FinalLayerSnapshot, JointBlockSnapshot, MlpSnapshot,
    MmditCpuSnapshot, PatchEmbedSnapshot, QkNormSnapshot, SelfAttentionSnapshot,
    TimestepEmbedderSnapshot, VectorEmbedderSnapshot,
};
use crate::ops::{Conv2d, LayerNorm, Linear, RMSNorm};
#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
use flame_core::cpu::{
    linear::LinearSnapshot,
    norm::{LayerNormSnapshot, RmsNormSnapshot},
    snapshot::{Bf16CpuSnapshot, F32CpuSnapshot},
};
use flame_core::device::{CudaStreamRawPtrExt, Device};
use flame_core::ops::elt::{add_inplace_same_dtype, gate_mul_bf16_inplace, mul_inplace_same_dtype};
use flame_core::{cuda_ops_bf16, sdpa, DType, Error, Result, Shape, Tensor};
use half::bf16;
use std::borrow::Cow;
use std::env;
use std::sync::{Arc, OnceLock};

pub use flame_core::staging::ArenaScratch;

const ARENA_ALIGN: usize = ArenaScratch::DEFAULT_ALIGN;

fn freeze_linear(linear: &mut Linear, label: &str) {
    let weight_before = linear.weight.requires_grad();
    let bias_before = linear.bias.as_ref().map(|b| b.requires_grad()).unwrap_or(false);
    let weight = linear.weight.alias().requires_grad_(false);
    linear.weight = weight;
    if let Some(bias) = linear.bias.as_ref() {
        linear.bias = Some(bias.alias().requires_grad_(false));
    }
    let weight_after = linear.weight.requires_grad();
    let bias_after = linear.bias.as_ref().map(|b| b.requires_grad()).unwrap_or(false);
    log::trace!(
        "freeze_linear {label}: weight {} -> {}, bias {} -> {}",
        weight_before,
        weight_after,
        bias_before,
        bias_after
    );
}

fn trace_linear_state(linear: &Linear, label: &str) {
    let weight_grad = linear.weight.requires_grad();
    let bias_grad = linear.bias.as_ref().map(|b| b.requires_grad()).unwrap_or(false);
    log::trace!(
        "linear_state {label}: weight_requires_grad={} bias_requires_grad={}",
        weight_grad,
        bias_grad
    );
}

fn streaming_trace_enabled() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| match env::var("STREAMING_TRACE_FORWARD") {
        Ok(val) => matches!(val.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    })
}

fn recompute_secondary_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| match env::var("STREAMING_RECOMPUTE_SECONDARY") {
        Ok(val) => matches!(val.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => true,
    })
}

fn maybe_detach_tensor<'a>(tensor: &'a Tensor, label: &str) -> Result<Cow<'a, Tensor>> {
    if tensor.requires_grad() {
        let alias = tensor.alias().requires_grad_(false);
        log::trace!("grad_clear {label}: true -> false");
        Ok(Cow::Owned(alias))
    } else {
        Ok(Cow::Borrowed(tensor))
    }
}

fn promote_to_owning_bf16(mut tensor: Tensor, label: &str) -> Result<Tensor> {
    if tensor.dtype() == DType::BF16 && tensor.storage_dtype() == DType::BF16 {
        #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
        {
            if tensor.is_bf16_arena() {
                log::trace!("promote_to_owning_bf16 {label}: arena -> owning");
                tensor = tensor.clone_result()?;
            }
        }
    }
    Ok(tensor)
}

fn detach_and_promote_bf16(tensor: &Tensor, label: &str) -> Result<Tensor> {
    let cow = maybe_detach_tensor(tensor, label)?;
    let owned = match cow {
        Cow::Owned(owned) => owned,
        Cow::Borrowed(_) => tensor.alias(),
    };
    promote_to_owning_bf16(owned, label)
}

fn ensure_bf16_accumulator(mut tensor: Tensor, label: &str) -> Result<Tensor> {
    if tensor.storage_dtype() != DType::BF16 {
        tensor = tensor.to_dtype(DType::BF16)?;
    }
    promote_to_owning_bf16(tensor, label)
}

fn tensor_size_mebibytes(tensor: &Tensor) -> f64 {
    let numel = tensor.shape().elem_count() as u64;
    let dtype_bytes = tensor.dtype().size_in_bytes() as u64;
    (numel.saturating_mul(dtype_bytes) as f64) / (1024.0 * 1024.0)
}

fn log_tensor_trace(trace: bool, label: &str, tensor: &Tensor) {
    if !trace {
        return;
    }
    let dims = tensor.shape().dims();
    let dtype = tensor.dtype();
    let storage = tensor.storage_dtype();
    let mib = tensor_size_mebibytes(tensor);
    log::debug!(
        "JointBlock::trace {label}: shape={:?} dtype={:?} storage={:?} ~{:.3} MiB",
        dims,
        dtype,
        storage,
        mib
    );
}

fn log_optional_tensor_trace(trace: bool, label: &str, tensor: &Option<Tensor>) {
    if let Some(t) = tensor.as_ref() {
        log_tensor_trace(trace, label, t);
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn log_vram_trace(trace: bool, label: &str) {
    if !trace {
        return;
    }
    if let Some(free_mb) = flame_core::cuda::utils::cuda_mem_get_free_mb() {
        log::debug!("JointBlock::trace VRAM {label}: free={} MiB", free_mb);
    }
}

#[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
fn log_vram_trace(_: bool, _: &str) {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkNormKind {
    Disabled,
    Layer,
    Rms,
}

impl Default for QkNormKind {
    fn default() -> Self {
        QkNormKind::Layer
    }
}

impl From<bool> for QkNormKind {
    fn from(value: bool) -> Self {
        if value {
            QkNormKind::Layer
        } else {
            QkNormKind::Disabled
        }
    }
}

#[derive(Clone, Debug)]
pub struct MMDiTConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub mlp_ratio: f32,
    pub qkv_bias: bool,
    pub qk_norm: QkNormKind,
    pub pos_embed_max_size: usize,
    /// Highest block index (inclusive) whose `x_block` performs self-attention.
    /// `None` disables the self-attention branch.
    pub x_self_attn_layers: Option<usize>,
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub frequency_embedding_size: usize,
    pub context_dim: usize,
    pub pooled_dim: Option<usize>,
}

impl Default for MMDiTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1536,
            num_heads: 24,
            depth: 38,
            mlp_ratio: 4.0,
            qkv_bias: true,
            qk_norm: QkNormKind::default(),
            pos_embed_max_size: 192,
            x_self_attn_layers: None,
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            frequency_embedding_size: 256,
            context_dim: 4096,
            pooled_dim: Some(2048),
        }
    }
}

fn reshape_gate(g: &Tensor) -> Result<Tensor> {
    let mut dims = g.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(Error::InvalidShape(format!("expected gating tensor rank 2, got {:?}", dims)));
    }
    dims.insert(1, 1);
    g.reshape(&dims)
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let scale = reshape_gate(scale)?;
    let shift = reshape_gate(shift)?;
    x.mul(&(scale.add_scalar(1.0)?))?.add(&shift)
}

fn modulate_inplace(arena: &mut Tensor, shift: &Tensor, scale: &Tensor) -> Result<()> {
    // Optimized BF16 path using in-place gate operations to avoid expansion allocations
    if arena.dtype() == DType::BF16 && arena.storage_dtype() == DType::BF16 {
        let target = arena.shape().dims();
        if target.len() != 3 {
            return Err(Error::InvalidShape(format!(
                "modulation expects rank-3 tensor, got {:?}",
                target
            )));
        }
        let batch = target[0];
        let hidden = target[2];

        // Prepare scale: (1 + scale)
        // reshape_gate returns [B, 1, H], reshape to [B, H] for gate ops
        let scale_view = reshape_gate(scale)?.reshape(&[batch, hidden])?;
        let mut scale_ready =
            if scale_view.dtype() == DType::BF16 && scale_view.storage_dtype() == DType::BF16 {
                scale_view.clone_result()?
            } else {
                scale_view.to_dtype(DType::BF16)?
            };
        scale_ready = scale_ready.add_scalar(1.0)?;

        // Apply scale: arena = arena * (1 + scale)
        gate_mul_bf16_inplace(arena, &scale_ready)?;

        // Prepare shift
        let shift_view = reshape_gate(shift)?.reshape(&[batch, hidden])?;
        let shift_ready =
            if shift_view.dtype() == DType::BF16 && shift_view.storage_dtype() == DType::BF16 {
                shift_view.clone_result()?
            } else {
                shift_view.to_dtype(DType::BF16)?
            };

        // Apply shift: arena = arena + shift
        flame_core::cuda_kernels::gate_add_bf16_inplace(arena, &shift_ready)?;

        return Ok(());
    }

    // Fallback for non-BF16
    let scale = reshape_gate(scale)?;
    let shift = reshape_gate(shift)?;
    let scale_plus_one = scale.add_scalar(1.0)?;

    let target_shape = arena.shape().dims();
    let scale_expanded = scale_plus_one.expand(target_shape)?;
    let shift_expanded = shift.expand(target_shape)?;

    mul_inplace_same_dtype(arena, &scale_expanded)?;
    add_inplace_same_dtype(arena, &shift_expanded)?;

    Ok(())
}

pub struct QkvSet {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

struct BlockIntermediates {
    residual: Tensor,
    gate_msa: Option<Tensor>,
    shift_mlp: Option<Tensor>,
    scale_mlp: Option<Tensor>,
    gate_mlp: Option<Tensor>,
    gate_msa2: Option<Tensor>,
    shift_msa2: Option<Tensor>,
    scale_msa2: Option<Tensor>,
    recompute_secondary: bool,
}

struct PreAttentionOutput {
    primary: QkvSet,
    intermediates: BlockIntermediates,
}

/// 2D image to patch embedding (mirrors the C++ PatchEmbed block).
pub struct PatchEmbed {
    proj: Conv2d,
    flatten: bool,
    dynamic_img_pad: bool,
    patch_size: usize,
}

impl PatchEmbed {
    pub fn new(
        img_size: Option<usize>,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        bias: bool,
        flatten: bool,
        dynamic_img_pad: bool,
        device: &Device,
    ) -> Result<Self> {
        let _ = img_size;
        let proj = if bias {
            Conv2d::new_with_bias_zeroed(
                in_chans,
                embed_dim,
                patch_size,
                patch_size,
                0,
                device.cuda_device().clone(),
                bias,
            )?
        } else {
            Conv2d::new_zeroed(
                in_chans,
                embed_dim,
                patch_size,
                patch_size,
                0,
                device.cuda_device().clone(),
            )?
        };
        Ok(Self { proj, flatten, dynamic_img_pad, patch_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.proj.forward(x)?;
        // Conv2d returns NCHW, but we need NHWC for flattening/reshaping
        out = out.permute(&[0, 2, 3, 1])?;
        log::debug!("PatchEmbed output dtype {:?} storage {:?}", out.dtype(), out.storage_dtype());
        if self.flatten {
            let dims = out.shape().dims();
            let n = dims[0];
            let h = dims[1];
            let w = dims[2];
            let c = dims[3];
            out.reshape_inplace(&[n, h * w, c])?;
            log::debug!(
                "PatchEmbed flattened dtype {:?} storage {:?}",
                out.dtype(),
                out.storage_dtype()
            );
        }
        Ok(out)
    }

    pub fn proj_mut(&mut self) -> &mut Conv2d {
        &mut self.proj
    }

    pub fn proj(&self) -> &Conv2d {
        &self.proj
    }

    pub fn flatten_enabled(&self) -> bool {
        self.flatten
    }

    pub fn dynamic_img_pad_enabled(&self) -> bool {
        self.dynamic_img_pad
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn in_channels(&self) -> usize {
        self.proj.config.in_channels
    }

    pub fn weight_shape(&self) -> &[usize] {
        self.proj.weight.shape().dims()
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn cpu_snapshot(&self) -> Result<PatchEmbedSnapshot> {
        Ok(PatchEmbedSnapshot {
            proj: Conv2dSnapshot::from_conv(&self.proj)?,
            flatten: self.flatten,
            dynamic_img_pad: self.dynamic_img_pad,
            patch_size: self.patch_size,
        })
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &PatchEmbedSnapshot) -> Result<()> {
        if self.proj.config.in_channels != snapshot.proj.in_channels
            || self.proj.config.out_channels != snapshot.proj.out_channels
        {
            return Err(Error::InvalidInput("PatchEmbed snapshot channel shape mismatch".into()));
        }
        self.flatten = snapshot.flatten;
        self.dynamic_img_pad = snapshot.dynamic_img_pad;
        self.patch_size = snapshot.patch_size;
        let device = self.proj.weight.device().clone();
        let weight = snapshot.proj.weight.to_cuda_tensor(device.clone())?;
        self.proj.copy_weight_from(&weight)?;
        match (&snapshot.proj.bias, &self.proj.bias) {
            (Some(b), Some(_)) => {
                let tensor = b.to_cuda_tensor(device)?;
                self.proj.copy_bias_from(&tensor)?;
            }
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput(
                    "PatchEmbed snapshot bias presence mismatch".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Scalar timestep embedder (sinusoidal + MLP).
pub struct TimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(
        hidden_size: usize,
        frequency_embedding_size: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::new_zeroed(
                frequency_embedding_size,
                hidden_size,
                true,
                &device.cuda_device(),
            )?,
            linear_2: Linear::new_zeroed(hidden_size, hidden_size, true, &device.cuda_device())?,
            frequency_embedding_size,
        })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let embedding = Self::timestep_embedding(
            t,
            self.frequency_embedding_size,
            &Device::from(t.device().clone()),
        )?
        .to_dtype(DType::BF16)?;
        let x = self.linear_1.forward(&embedding)?;
        let x = x.silu()?;
        self.linear_2.forward(&x)
    }

    fn timestep_embedding(
        timesteps: &Tensor,
        embedding_dim: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let half_dim = embedding_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            // Matches Diffusers get_timestep_embedding with downscale_freq_shift=1 (default)
            // exp(-i * ln(10000) / (half_dim - 1))
            inv_freq.push(-(i as f32) / (half_dim as f32));
        }
        let inv_freq = Tensor::from_vec(
            inv_freq,
            Shape::from_dims(&[half_dim]),
            device.cuda_device().clone(),
        )?
        .mul_scalar(10000f32.ln())?
        .exp()?;
        let t = timesteps.unsqueeze(1)?;
        let sinusoid = t.mul(&inv_freq.unsqueeze(0)?)?;
        let sin = sinusoid.sin()?.to_dtype(DType::BF16)?;
        let cos = sinusoid.cos()?.to_dtype(DType::BF16)?;
        // Matches Diffusers flip_sin_to_cos=True (default) -> [cos, sin]
        Tensor::cat(&[&cos, &sin], 1)
    }

    pub fn linear_layers_mut(&mut self) -> (&mut Linear, &mut Linear) {
        (&mut self.linear_1, &mut self.linear_2)
    }

    pub fn linear_layers(&self) -> (&Linear, &Linear) {
        (&self.linear_1, &self.linear_2)
    }

    pub fn frequency_embedding_size(&self) -> usize {
        self.frequency_embedding_size
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn cpu_snapshot(&self) -> Result<TimestepEmbedderSnapshot> {
        Ok(TimestepEmbedderSnapshot {
            linear1: LinearSnapshot::from_linear(&self.linear_1)?,
            linear2: LinearSnapshot::from_linear(&self.linear_2)?,
            frequency_embedding_size: self.frequency_embedding_size,
        })
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &TimestepEmbedderSnapshot) -> Result<()> {
        self.frequency_embedding_size = snapshot.frequency_embedding_size;
        snapshot.linear1.apply_to(&mut self.linear_1)?;
        snapshot.linear2.apply_to(&mut self.linear_2)?;
        Ok(())
    }
}

/// Flat vector embedder (used for pooled text embeddings).
pub struct VectorEmbedder {
    linear_1: Linear,
    linear_2: Linear,
}

impl VectorEmbedder {
    pub fn new(input_dim: usize, hidden_size: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::new_zeroed(input_dim, hidden_size, true, &device.cuda_device())?,
            linear_2: Linear::new_zeroed(hidden_size, hidden_size, true, &device.cuda_device())?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(x)?;
        let x = x.silu()?;
        self.linear_2.forward(&x)
    }

    pub fn linear_layers_mut(&mut self) -> (&mut Linear, &mut Linear) {
        (&mut self.linear_1, &mut self.linear_2)
    }

    pub fn linear_layers(&self) -> (&Linear, &Linear) {
        (&self.linear_1, &self.linear_2)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn cpu_snapshot(&self) -> Result<VectorEmbedderSnapshot> {
        Ok(VectorEmbedderSnapshot {
            linear1: LinearSnapshot::from_linear(&self.linear_1)?,
            linear2: LinearSnapshot::from_linear(&self.linear_2)?,
            input_dim: self.linear_1.in_features(),
        })
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &VectorEmbedderSnapshot) -> Result<()> {
        if snapshot.input_dim != self.linear_1.in_features() {
            return Err(Error::InvalidInput(
                "VectorEmbedder snapshot input dimension mismatch".into(),
            ));
        }
        snapshot.linear1.apply_to(&mut self.linear_1)?;
        snapshot.linear2.apply_to(&mut self.linear_2)?;
        Ok(())
    }
}

pub(crate) fn manual_rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    // x can be BF16 or F32. We want to keep it in its original dtype as much as possible.
    let dims = x.shape().dims();
    let last_dim = dims[dims.len() - 1];
    let norm_dims = [dims.len() - 1];
    
    // Use mul(x) instead of pow(2.0) to avoid F32 cast if x is BF16
    let x_sq = x.mul(x)?;
    
    // mean_along_dims supports BF16 (via sum_dim_keepdim)
    // It ignores keepdim, so we must reshape manually
    let mean_sq = x_sq.mean_along_dims(&norm_dims, true)?;
    
    // Construct shape with 1 at the end for broadcasting
    let mut broadcast_shape = dims.to_vec();
    broadcast_shape[dims.len() - 1] = 1;
    let mean_sq = mean_sq.reshape(&broadcast_shape)?;

    // Calculate rsqrt in F32 for precision
    let mean_sq_f32 = mean_sq.to_dtype(DType::F32)?;
    let rsqrt = (mean_sq_f32.add_scalar(eps))?.rsqrt()?;
    
    // Cast back to input dtype for multiplication
    let rsqrt_casted = rsqrt.to_dtype(x.dtype())?;
    let norm = x.mul(&rsqrt_casted)?;
    
    if let Some(w) = weight {
        // Debug weight stats
        let w_f32 = w.to_dtype(DType::F32)?;
        let w_mean = w_f32.mean()?.to_vec()?[0];
        let w_std = w_f32.std(Some(&[0]), true)?.to_vec()?[0];
        log::debug!("manual_rms_norm weight: mean={:.4} std={:.4}", w_mean, w_std);

        // Ensure weight matches input dtype
        let w_casted = w.to_dtype(x.dtype())?;
        let w_reshaped = w_casted.reshape(&[1, 1, last_dim])?; // Broadcast for [B*H, T, D]
        norm.mul(&w_reshaped)
    } else {
        log::debug!("manual_rms_norm: no weight");
        Ok(norm)
    }
}

fn manual_layer_norm(x: &Tensor, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims();
    let last_dim = dims[dims.len() - 1];
    let norm_dims = [dims.len() - 1];
    
    // Construct shape with 1 at the end for broadcasting
    let mut broadcast_shape = dims.to_vec();
    broadcast_shape[dims.len() - 1] = 1;

    // Mean in original dtype
    let mean = x.mean_along_dims(&norm_dims, true)?;
    let mean = mean.reshape(&broadcast_shape)?;
    
    let centered = x.sub(&mean)?;
    
    // Var in original dtype
    let var = centered.mul(&centered)?.mean_along_dims(&norm_dims, true)?;
    let var = var.reshape(&broadcast_shape)?;
    
    // Rsqrt in F32
    let var_f32 = var.to_dtype(DType::F32)?;
    let rsqrt = (var_f32.add_scalar(eps))?.rsqrt()?;
    let rsqrt_casted = rsqrt.to_dtype(x.dtype())?;
    
    let norm = centered.mul(&rsqrt_casted)?;
    
    let mut out = norm;
    if let Some(w) = weight {
        let w_casted = w.to_dtype(x.dtype())?;
        let w_reshaped = w_casted.reshape(&[1, 1, last_dim])?;
        out = out.mul(&w_reshaped)?;
    }
    if let Some(b) = bias {
        let b_casted = b.to_dtype(x.dtype())?;
        let b_reshaped = b_casted.reshape(&[1, 1, last_dim])?;
        out = out.add(&b_reshaped)?;
    }
    Ok(out)
}

use flame_core::CudaDevice;

pub(crate) enum Norm {
    LayerNorm(LayerNorm),
    RmsNorm(RMSNorm),
}

pub struct QKNorm {
    pub kind: QkNormKind,
    pub norm_q: Option<Norm>,
    pub norm_k: Option<Norm>,
    pub rms_q: Option<RMSNorm>,
    pub rms_k: Option<RMSNorm>,
}

impl QKNorm {
    pub fn new(
        kind: QkNormKind,
        head_dim: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let (norm_q, norm_k, rms_q, rms_k) = match kind {
            QkNormKind::Disabled => (None, None, None, None),
            QkNormKind::Layer => {
                let q = Norm::LayerNorm(LayerNorm::new(vec![head_dim], 1e-6, device.clone())?);
                let k = Norm::LayerNorm(LayerNorm::new(vec![head_dim], 1e-6, device.clone())?);
                (Some(q), Some(k), None, None)
            }
            QkNormKind::Rms => {
                let q_rms = RMSNorm::new(vec![head_dim], 1e-6, true, device.clone())?;
                let k_rms = RMSNorm::new(vec![head_dim], 1e-6, true, device.clone())?;
                
                let q_rms_clone = RMSNorm {
                    eps: q_rms.eps,
                    elementwise_affine: q_rms.elementwise_affine,
                    normalized_shape: q_rms.normalized_shape.clone(),
                    weight: q_rms.weight.clone(),
                };
                let k_rms_clone = RMSNorm {
                    eps: k_rms.eps,
                    elementwise_affine: k_rms.elementwise_affine,
                    normalized_shape: k_rms.normalized_shape.clone(),
                    weight: k_rms.weight.clone(),
                };
                
                let q = Norm::RmsNorm(q_rms_clone);
                let k = Norm::RmsNorm(k_rms_clone);
                (Some(q), Some(k), Some(q_rms), Some(k_rms))
            }
        };

        Ok(Self {
            kind,
            norm_q,
            norm_k,
            rms_q,
            rms_k,
        })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        self.forward_with_scratch(q, k, None)
    }

    pub fn forward_with_scratch(
        &self,
        q: &Tensor,
        k: &Tensor,
        scratch: Option<&ArenaScratch>,
    ) -> Result<(Tensor, Tensor)> {
        match self.kind {
            QkNormKind::Disabled => Ok((q.clone(), k.clone())),
            QkNormKind::Layer => {
                let dims = q.shape().dims();
                if dims.len() != 4 {
                    return Err(Error::InvalidShape(format!(
                        "expected [B, H, T, D] for q, got {:?}",
                        dims
                    )));
                }
                let b = dims[0];
                let h = dims[1];
                let t = dims[2];
                let d = dims[3];
                let reshaped = &[b * h, t, d];
                
                // No cast to F32!
                let q_view = q.reshape(reshaped)?;
                let k_view = k.reshape(reshaped)?;

                // Use manual LayerNorm
                let norm_q = match self.norm_q.as_ref().unwrap() {
                    Norm::LayerNorm(ln) => ln,
                    _ => return Err(Error::InvalidInput("Expected LayerNorm".into())),
                };
                let mut q_norm = manual_layer_norm(&q_view, norm_q.weight.as_ref(), norm_q.bias.as_ref(), norm_q.eps)?;
                q_norm.reshape_inplace(dims)?;
                
                let norm_k = match self.norm_k.as_ref().unwrap() {
                    Norm::LayerNorm(ln) => ln,
                    _ => return Err(Error::InvalidInput("Expected LayerNorm".into())),
                };
                let mut k_norm = manual_layer_norm(&k_view, norm_k.weight.as_ref(), norm_k.bias.as_ref(), norm_k.eps)?;
                k_norm.reshape_inplace(dims)?;
                
                Ok((q_norm, k_norm))
            }
            QkNormKind::Rms => {
                let dims = q.shape().dims();
                if dims.len() != 4 {
                    return Err(Error::InvalidShape(format!(
                        "expected [B, H, T, D] for q, got {:?}",
                        dims
                    )));
                }
                let b = dims[0];
                let h = dims[1];
                let t = dims[2];
                let d = dims[3];
                let reshaped = &[b * h, t, d];
                
                // No cast to F32!
                let q_view = q.reshape(reshaped)?;
                let k_view = k.reshape(reshaped)?;
                
                // Use manual RMSNorm
                let rms_q = self.rms_q.as_ref().unwrap();
                let mut q_norm = manual_rms_norm(&q_view, rms_q.weight.as_ref(), rms_q.eps)?;
                q_norm.reshape_inplace(dims)?;
                
                let rms_k = self.rms_k.as_ref().unwrap();
                let mut k_norm = manual_rms_norm(&k_view, rms_k.weight.as_ref(), rms_k.eps)?;
                k_norm.reshape_inplace(dims)?;
                
                Ok((q_norm, k_norm))
            }
        }
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    fn cpu_snapshot(&self) -> Result<QkNormSnapshot> {
        let norm_q = self.norm_q.as_ref().map(|n| n.cpu_snapshot()).transpose()?;
        let norm_k = self.norm_k.as_ref().map(|n| n.cpu_snapshot()).transpose()?;
        let rms_q = self.rms_q.as_ref().map(|n| n.cpu_snapshot()).transpose()?;
        let rms_k = self.rms_k.as_ref().map(|n| n.cpu_snapshot()).transpose()?;
        Ok(QkNormSnapshot { kind: self.kind, norm_q, norm_k, rms_q, rms_k })
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &QkNormSnapshot) -> Result<()> {
        if snapshot.kind != self.kind {
            return Err(Error::InvalidInput("QkNorm snapshot kind mismatch".into()));
        }

        match (&mut self.norm_q, &snapshot.norm_q) {
            (Some(Norm::LayerNorm(norm)), Some(snap)) => norm.apply_cpu_snapshot(snap)?,
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput(
                    "QkNorm snapshot expected LayerNorm mismatch".into(),
                ));
            }
        }
        match (&mut self.norm_k, &snapshot.norm_k) {
            (Some(Norm::LayerNorm(norm)), Some(snap)) => norm.apply_cpu_snapshot(snap)?,
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput(
                    "QkNorm snapshot expected LayerNorm mismatch".into(),
                ));
            }
        }
        match (&mut self.rms_q, &snapshot.rms_q) {
            (Some(norm), Some(snap)) => norm.apply_cpu_snapshot(snap)?,
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput(
                    "QkNorm snapshot expected RMSNorm mismatch".into(),
                ));
            }
        }
        match (&mut self.rms_k, &snapshot.rms_k) {
            (Some(norm), Some(snap)) => norm.apply_cpu_snapshot(snap)?,
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput(
                    "QkNorm snapshot expected RMSNorm mismatch".into(),
                ));
            }
        }

        Ok(())
    }

    pub fn kind(&self) -> QkNormKind {
        self.kind
    }

    pub fn layer_norms_mut(&mut self) -> (Option<&mut LayerNorm>, Option<&mut LayerNorm>) {
        let q = match self.norm_q.as_mut() {
            Some(Norm::LayerNorm(ln)) => Some(ln),
            _ => None,
        };
        let k = match self.norm_k.as_mut() {
            Some(Norm::LayerNorm(ln)) => Some(ln),
            _ => None,
        };
        (q, k)
    }

    pub fn rms_norms_mut(&mut self) -> (Option<&mut RMSNorm>, Option<&mut RMSNorm>) {
        (self.rms_q.as_mut(), self.rms_k.as_mut())
    }
}

pub struct SelfAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    proj: Option<Linear>,
    qk_norm: QKNorm,
}

impl SelfAttention {
    pub fn new(
        hidden: usize,
        num_heads: usize,
        qkv_bias: bool,
        qk_norm: QkNormKind,
        pre_only: bool,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = hidden / num_heads;
        let q_proj = Linear::new_zeroed(hidden, hidden, qkv_bias, &device.cuda_device())?;
        let k_proj = Linear::new_zeroed(hidden, hidden, qkv_bias, &device.cuda_device())?;
        let v_proj = Linear::new_zeroed(hidden, hidden, qkv_bias, &device.cuda_device())?;
        let proj = if pre_only {
            None
        } else {
            Some(Linear::new_zeroed(hidden, hidden, true, &device.cuda_device())?)
        };

        Ok(Self {
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            proj,
            qk_norm: QKNorm::new(qk_norm, head_dim, device.cuda_device_arc())?,
        })
    }

    pub fn pre_attention(&self, x: &Tensor, scratch: &ArenaScratch) -> Result<QkvSet> {
        let dims = x.shape().dims();
        if dims.len() != 3 {
            return Err(Error::InvalidShape(format!(
                "self-attention expects [B, T, C], got {:?}",
                dims
            )));
        }
        let b = dims[0];
        let t = dims[1];
        let hidden = self.num_heads * self.head_dim;

        trace_linear_state(&self.q_proj, "self_attn.q");
        trace_linear_state(&self.k_proj, "self_attn.k");
        trace_linear_state(&self.v_proj, "self_attn.v");
        if let Some(proj) = &self.proj {
            trace_linear_state(proj, "self_attn.proj");
        }

        let attn_input = detach_and_promote_bf16(x, "self_attn.input")?;
        let x_ref = &attn_input;

        let fast_path = !x_ref.requires_grad()
            && !self.q_proj.weight.requires_grad()
            && !self.k_proj.weight.requires_grad()
            && !self.v_proj.weight.requires_grad()
            && self.q_proj.bias.as_ref().map(|b| !b.requires_grad()).unwrap_or(true)
            && self.k_proj.bias.as_ref().map(|b| !b.requires_grad()).unwrap_or(true)
            && self.v_proj.bias.as_ref().map(|b| !b.requires_grad()).unwrap_or(true);
        log::trace!(
            "SelfAttention::pre_attention fast_path={} input_requires_grad={}",
            fast_path,
            x_ref.requires_grad()
        );

        let (mut q, mut k, mut v) = if fast_path {
            println!("DEBUG: SelfAttention: q_proj.forward...");
            let q_fast = self.q_proj.forward_with_scratch(x_ref, scratch);
            if let Ok(q) = &q_fast { let _ = q.to_vec(); } // Sync
            
            println!("DEBUG: SelfAttention: k_proj.forward...");
            let k_fast = self.k_proj.forward_with_scratch(x_ref, scratch);
            if let Ok(k) = &k_fast { let _ = k.to_vec(); } // Sync

            println!("DEBUG: SelfAttention: v_proj.forward...");
            let v_fast = self.v_proj.forward_with_scratch(x_ref, scratch);
            if let Ok(v) = &v_fast { let _ = v.to_vec(); } // Sync

            if let (Ok(q), Ok(k), Ok(v)) = (q_fast, k_fast, v_fast) {
                (q, k, v)
            } else {
                log::trace!("SelfAttention::pre_attention arena fast path unavailable; falling back to standard linear forwards");
                println!("DEBUG: SelfAttention: fallback linear...");
                (
                    self.q_proj.forward(x_ref)?,
                    self.k_proj.forward(x_ref)?,
                    self.v_proj.forward(x_ref)?,
                )
            }
        } else {
            println!("DEBUG: SelfAttention: standard linear...");
            (self.q_proj.forward(x_ref)?, self.k_proj.forward(x_ref)?, self.v_proj.forward(x_ref)?)
        };
        println!("DEBUG: SelfAttention: promote_to_owning_bf16...");
        q = promote_to_owning_bf16(q, "self_attn.q_tensor")?;
        k = promote_to_owning_bf16(k, "self_attn.k_tensor")?;
        v = promote_to_owning_bf16(v, "self_attn.v_tensor")?;

        println!("DEBUG: SelfAttention: reshape_inplace...");
        q.reshape_inplace(&[b, t, self.num_heads, self.head_dim])?;
        k.reshape_inplace(&[b, t, self.num_heads, self.head_dim])?;
        v.reshape_inplace(&[b, t, self.num_heads, self.head_dim])?;
        
        println!("DEBUG: SelfAttention: permute to [B, H, T, D]...");
        let mut q = q.permute(&[0, 2, 1, 3])?.affine(1.0, 0.0)?;
        let mut k = k.permute(&[0, 2, 1, 3])?.affine(1.0, 0.0)?;
        let v = v.permute(&[0, 2, 1, 3])?.affine(1.0, 0.0)?;

        let (q, k) = self.qk_norm.forward(&q, &k)?;
        
        Ok(QkvSet { q, k, v })
    }

    pub fn project_output(&self, attn_ctx: &Tensor, scratch: &ArenaScratch) -> Result<Tensor> {
        let dims = attn_ctx.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidShape(format!(
                "attention context must be [B, H, T, D], got {:?}",
                dims
            )));
        }

        let b = dims[0];
        let t = dims[2];
        let hidden = self.num_heads * self.head_dim;
        let mut merged = attn_ctx.permute(&[0, 2, 1, 3])?;
        merged.reshape_inplace(&[b, t, hidden])?;

        if let Some(proj) = &self.proj {
            let merged_ready = promote_to_owning_bf16(merged, "self_attn.proj_input")?;
            match proj.forward_with_scratch(&merged_ready, scratch) {
                Ok(out) => Ok(out),
                Err(err) => {
                    log::trace!(
                        "SelfAttention::project_output scratch path unavailable; falling back to fresh tensor: {err}"
                    );
                    proj.forward(&merged_ready)
                }
            }
        } else {
            Ok(merged)
        }
    }

    pub fn q_proj_mut(&mut self) -> &mut Linear {
        &mut self.q_proj
    }

    pub fn q_proj(&self) -> &Linear {
        &self.q_proj
    }

    pub fn k_proj_mut(&mut self) -> &mut Linear {
        &mut self.k_proj
    }

    pub fn k_proj(&self) -> &Linear {
        &self.k_proj
    }

    pub fn v_proj_mut(&mut self) -> &mut Linear {
        &mut self.v_proj
    }

    pub fn v_proj(&self) -> &Linear {
        &self.v_proj
    }

    pub fn hidden_size(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn in_features(&self) -> usize {
        self.q_proj.in_features()
    }

    pub fn copy_qkv_weight_from(&mut self, fused: &Tensor) -> Result<()> {
        let dims = fused.shape().dims();
        let hidden = self.hidden_size();
        if dims.len() != 2 || dims[0] != hidden * 3 || dims[1] != self.in_features() {
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(&[hidden * 3, self.in_features()]),
                got: fused.shape().clone(),
            });
        }

        let q_weight = fused.slice_1d_device(0, 0, hidden)?;
        self.q_proj.copy_weight_from(&q_weight)?;
        let k_weight = fused.slice_1d_device(0, hidden, hidden)?;
        self.k_proj.copy_weight_from(&k_weight)?;
        let v_weight = fused.slice_1d_device(0, hidden * 2, hidden)?;
        self.v_proj.copy_weight_from(&v_weight)?;
        Ok(())
    }

    pub fn copy_qkv_bias_from(&mut self, fused: &Tensor) -> Result<()> {
        let dims = fused.shape().dims();
        let hidden = self.hidden_size();
        if dims.len() != 1 || dims[0] != hidden * 3 {
            return Err(Error::ShapeMismatch {
                expected: Shape::from_dims(&[hidden * 3]),
                got: fused.shape().clone(),
            });
        }

        if self.q_proj.bias.is_some() {
            let q_bias = fused.slice_1d_device(0, 0, hidden)?;
            self.q_proj.copy_bias_from(&q_bias)?;
        }

        if self.k_proj.bias.is_some() {
            let k_bias = fused.slice_1d_device(0, hidden, hidden)?;
            self.k_proj.copy_bias_from(&k_bias)?;
        }

        if self.v_proj.bias.is_some() {
            let v_bias = fused.slice_1d_device(0, hidden * 2, hidden)?;
            self.v_proj.copy_bias_from(&v_bias)?;
        }
        Ok(())
    }

    pub fn disable_grads(&mut self) {
        freeze_linear(&mut self.q_proj, "self_attn.q");
        freeze_linear(&mut self.k_proj, "self_attn.k");
        freeze_linear(&mut self.v_proj, "self_attn.v");
        if let Some(proj) = &mut self.proj {
            freeze_linear(proj, "self_attn.proj");
        }
    }

    pub fn proj_mut(&mut self) -> Option<&mut Linear> {
        self.proj.as_mut()
    }

    pub fn proj(&self) -> Option<&Linear> {
        self.proj.as_ref()
    }

    pub fn qk_norm_mut(&mut self) -> &mut QKNorm {
        &mut self.qk_norm
    }

    pub fn qk_norm(&self) -> &QKNorm {
        &self.qk_norm
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn cpu_snapshot(&self) -> Result<SelfAttentionSnapshot> {
        let proj_snapshot = match &self.proj {
            Some(proj) => Some(LinearSnapshot::from_linear(proj)?),
            None => None,
        };
        Ok(SelfAttentionSnapshot {
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            q: LinearSnapshot::from_linear(&self.q_proj)?,
            k: LinearSnapshot::from_linear(&self.k_proj)?,
            v: LinearSnapshot::from_linear(&self.v_proj)?,
            proj: proj_snapshot,
            qk_norm: self.qk_norm.cpu_snapshot()?,
        })
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &SelfAttentionSnapshot) -> Result<()> {
        if snapshot.num_heads != self.num_heads || snapshot.head_dim != self.head_dim {
            return Err(Error::InvalidInput(
                "SelfAttention snapshot head configuration mismatch".into(),
            ));
        }
        snapshot.q.apply_to(self.q_proj_mut())?;
        snapshot.k.apply_to(self.k_proj_mut())?;
        snapshot.v.apply_to(self.v_proj_mut())?;
        match (self.proj.as_mut(), &snapshot.proj) {
            (Some(linear), Some(snap)) => snap.apply_to(linear)?,
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput(
                    "SelfAttention snapshot projection presence mismatch".into(),
                ));
            }
        }
        self.qk_norm.apply_cpu_snapshot(&snapshot.qk_norm)?;
        Ok(())
    }
}

pub struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    pub fn new(hidden: usize, mlp_ratio: f32, device: &Device) -> Result<Self> {
        let hidden_dim = (hidden as f32 * mlp_ratio).round() as usize;
        Ok(Self {
            fc1: Linear::new_zeroed(hidden, hidden_dim, true, &device.cuda_device())?,
            fc2: Linear::new_zeroed(hidden_dim, hidden, true, &device.cuda_device())?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        trace_linear_state(&self.fc1, "mlp.fc1");
        trace_linear_state(&self.fc2, "mlp.fc2");
        let x_ready = detach_and_promote_bf16(x, "mlp.input")?;
        log::trace!("MLP::forward input_requires_grad={}", x_ready.requires_grad());
        /*
        if x_ready.dtype() == DType::BF16
            && x_ready.storage_dtype() == DType::BF16
            && !x_ready.requires_grad()
        {
            let scratch = ArenaScratch::from_tensor_with_align(&x_ready, ARENA_ALIGN);
            if let Ok(hidden_result) = self.fc1.forward_with_scratch(&x_ready, &scratch) {
                let mut hidden = hidden_result;
                let hidden_clone = hidden.clone_result()?;
                cuda_ops_bf16::gelu_bf16_into(&hidden_clone, &mut hidden)?;
                hidden = promote_to_owning_bf16(hidden, "mlp.hidden")?;
                if let Ok(out) = self.fc2.forward_with_scratch(&hidden, &scratch) {
                    return Ok(out);
                } else {
                    log::trace!(
                        "MLP::forward fc2 scratch fast path unavailable; falling back to standard forward"
                    );
                    return self.fc2.forward(&hidden);
                }
            } else {
                log::trace!(
                    "MLP::forward fc1 scratch fast path unavailable; falling back to standard path"
                );
            }
        }
        */

        let act = self.fc1.forward(&x_ready)?;
        // Use approximate GELU to match Diffusers
        let act_approx = gelu_approx(&act)?;
        self.fc2.forward(&act_approx)
    }


    pub fn fc_layers_mut(&mut self) -> (&mut Linear, &mut Linear) {
        (&mut self.fc1, &mut self.fc2)
    }

    pub fn fc_layers(&self) -> (&Linear, &Linear) {
        (&self.fc1, &self.fc2)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn cpu_snapshot(&self) -> Result<MlpSnapshot> {
        Ok(MlpSnapshot {
            fc1: LinearSnapshot::from_linear(&self.fc1)?,
            fc2: LinearSnapshot::from_linear(&self.fc2)?,
        })
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &MlpSnapshot) -> Result<()> {
        snapshot.fc1.apply_to(&mut self.fc1)?;
        snapshot.fc2.apply_to(&mut self.fc2)?;
        Ok(())
    }
}

fn gelu_approx(x: &Tensor) -> Result<Tensor> {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c1 = (2.0f32 / std::f32::consts::PI).sqrt();
    let c2 = 0.044715f32;
    let x_cubed = x.mul(x)?.mul(x)?;
    let inner = x.add(&x_cubed.mul_scalar(c2)?)?.mul_scalar(c1)?;
    let tanh = inner.tanh()?;
    let out = x.mul_scalar(0.5)?.mul(&tanh.add_scalar(1.0)?)?;
    Ok(out)
}


pub struct DismantledBlock {
    pub norm1: LayerNorm,
    pub attn: SelfAttention,
    pub norm2: Option<LayerNorm>,
    pub mlp: Option<MLP>,
    pub modulation: Linear,
    pub attn2: Option<SelfAttention>,
    pub pre_only: bool,
    pub self_attn: bool,
    pub hidden: usize,
    pub num_heads: usize,
}

impl DismantledBlock {
    fn new(
        hidden: usize,
        cond_dim: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qk_norm: QkNormKind,
        qkv_bias: bool,
        pre_only: bool,
        self_attn: bool,
        device: &Device,
    ) -> Result<Self> {
        let modulation_slots = if self_attn {
            9
        } else if pre_only {
            2
        } else {
            6
        };

        Ok(Self {
            hidden,
            num_heads,
            pre_only,
            self_attn,
            norm1: LayerNorm::new(vec![hidden], 1e-6, device.cuda_device().clone())?,
            attn: SelfAttention::new(hidden, num_heads, qkv_bias, qk_norm, pre_only, device)?,
            attn2: if self_attn {
                Some(SelfAttention::new(hidden, num_heads, qkv_bias, qk_norm, false, device)?)
            } else {
                None
            },
            norm2: if pre_only {
                None
            } else {
                Some(LayerNorm::new(vec![hidden], 1e-6, device.cuda_device().clone())?)
            },
            mlp: if pre_only { None } else { Some(MLP::new(hidden, mlp_ratio, device)?) },
            modulation: Linear::new_zeroed(
                cond_dim,
                modulation_slots * hidden,
                true,
                &device.cuda_device(),
            )?,
        })
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    pub(crate) fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub(crate) fn pre_only(&self) -> bool {
        self.pre_only
    }

    pub(crate) fn self_attn(&self) -> bool {
        self.self_attn
    }

    pub(crate) fn norm1(&self) -> &LayerNorm {
        &self.norm1
    }

    pub(crate) fn norm2(&self) -> Option<&LayerNorm> {
        self.norm2.as_ref()
    }

    pub(crate) fn modulation(&self) -> &Linear {
        &self.modulation
    }

    pub(crate) fn attn(&self) -> &SelfAttention {
        &self.attn
    }

    pub(crate) fn attn2(&self) -> Option<&SelfAttention> {
        self.attn2.as_ref()
    }

    pub(crate) fn mlp(&self) -> Option<&MLP> {
        self.mlp.as_ref()
    }

    fn compute_modulations(&self, cond: &Tensor, scratch: &ArenaScratch) -> Result<Vec<Tensor>> {
        let cond_last = cond.shape().dims().last().copied().ok_or_else(|| {
            Error::InvalidShape("conditioning tensor must have trailing dim".into())
        })?;
        if cond_last != self.modulation.in_features() {
            return Err(Error::InvalidShape(format!(
                "conditioning dim {} does not match modulation input {}",
                cond_last,
                self.modulation.in_features()
            )));
        }
        
        let cond_silu = cond.silu()?;
        let mods = self
            .modulation
            .forward_with_scratch(&cond_silu, scratch)
            .or_else(|_| self.modulation.forward(&cond_silu))?;
        
        let out_dim = self.modulation.weight.shape().dims()[0];
        let in_dim = cond.shape().dims()[1];
        let chunks_wrong = out_dim / in_dim;
        let chunks_correct = out_dim / self.hidden;
        log::debug!("compute_modulations: out_dim={} in_dim={} hidden={} chunks_wrong={} chunks_correct={}", 
                    out_dim, in_dim, self.hidden, chunks_wrong, chunks_correct);
        
        // Fix the bug immediately if confirmed
        let chunks = chunks_correct; 
        let mods = mods.chunk(chunks, 1)?;
        Ok(mods)
    }

    fn pre_attention(
        &self,
        x: &Tensor,
        cond: &Tensor,
        scratch: &ArenaScratch,
    ) -> Result<PreAttentionOutput> {
        let (primary, _) = self.pre_attention_with_secondary(x, cond, scratch)?;
        Ok(primary)
    }

    fn pre_attention_with_secondary(
        &self,
        x: &Tensor,
        cond: &Tensor,
        scratch: &ArenaScratch,
    ) -> Result<(PreAttentionOutput, Option<QkvSet>)> {
        println!("DEBUG: x_block: compute_modulations...");
        let mods = self.compute_modulations(cond, scratch)?;

        println!("DEBUG: x_block: mods len={}", mods.len());
        let shift_msa = mods[0].clone();
        let scale_msa = mods[1].clone();
        let gate_msa = if self.pre_only { None } else { Some(mods.get(2).cloned().unwrap()) };

        let shift_mlp = if self.pre_only { None } else { Some(mods.get(3).cloned().unwrap()) };
        let scale_mlp = if self.pre_only { None } else { Some(mods.get(4).cloned().unwrap()) };
        let gate_mlp = if self.pre_only { None } else { Some(mods.get(5).cloned().unwrap()) };

        let shift_msa2 = if self.self_attn { Some(mods.get(6).cloned().unwrap()) } else { None };
        let scale_msa2 = if self.self_attn { Some(mods.get(7).cloned().unwrap()) } else { None };
        let gate_msa2 = if self.self_attn { Some(mods.get(8).cloned().unwrap()) } else { None };

        println!("DEBUG: x_block: norm1.forward...");
        let normed = self.norm1.forward(x)?;
        let mut attn_in = normed.clone();
        println!("DEBUG: x_block: modulate...");
        if attn_in.dtype() == DType::BF16 && attn_in.storage_dtype() == DType::BF16 {
            modulate_inplace(&mut attn_in, &shift_msa, &scale_msa)?;
        } else {
            attn_in = modulate(&attn_in, &shift_msa, &scale_msa)?;
        }

        println!("DEBUG: x_block: attn.pre_attention...");
        let primary = self.attn.pre_attention(&attn_in, scratch)?;

        let recompute_secondary = recompute_secondary_enabled() && self.self_attn;
        let secondary = if !recompute_secondary {
            if let (Some(shift2), Some(scale2)) = (&shift_msa2, &scale_msa2) {
            let mut attn2_in = normed;
            if attn2_in.dtype() == DType::BF16 && attn2_in.storage_dtype() == DType::BF16 {
                modulate_inplace(&mut attn2_in, shift2, scale2)?;
            } else {
                attn2_in = modulate(&attn2_in, shift2, scale2)?;
            }
            println!("DEBUG: x_block: attn2.pre_attention...");
            Some(
                    self.attn2
                        .as_ref()
                        .ok_or_else(|| Error::InvalidOperation("attn2 missing".into()))?
                        .pre_attention(&attn2_in, scratch)?,
                )
            } else {
                None
            }
        } else {
            None
        };

        Ok((
            PreAttentionOutput {
                primary,
                intermediates: BlockIntermediates {
                    residual: x.clone(),
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    gate_msa2,
                    shift_msa2,
                    scale_msa2,
                    recompute_secondary,
                },
            },
            secondary,
        ))
    }

    fn project_primary(&self, heads: &Tensor, scratch: &ArenaScratch) -> Result<Tensor> {
        self.attn.project_output(heads, scratch)
    }

    fn project_secondary(&self, heads: &Tensor, scratch: &ArenaScratch) -> Result<Tensor> {
        self.attn2
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("attn2 missing".into()))?
            .project_output(heads, scratch)
    }

    fn post_attention(
        &self,
        attn_out: Tensor,
        extra: Option<Tensor>,
        mut interm: BlockIntermediates,
    ) -> Result<Tensor> {
        if self.pre_only {
            return Ok(interm.residual);
        }

        let mut x = interm.residual;
        if let Some(gate_tensor) = &interm.gate_msa {
            if attn_out.dtype() == DType::BF16 && attn_out.storage_dtype() == DType::BF16 {
                x = ensure_bf16_accumulator(x, "msa.residual")?;
                let mut tmp = attn_out.clone();
                if tmp.storage_dtype() != DType::BF16 {
                    tmp = tmp.to_dtype(DType::BF16)?;
                }
                tmp = promote_to_owning_bf16(tmp, "msa.tmp")?;
                let gate = reshape_gate(gate_tensor)?;
                if let Err(err) = gate_mul_bf16_inplace(&mut tmp, &gate) {
                    log::trace!("gate msa_inplace fallback: {:?}", err);
                    let mut gate = gate;
                    if gate.shape().dims() != tmp.shape().dims() {
                        gate = gate.expand(tmp.shape().dims())?;
                    }
                    if gate.dtype() != tmp.dtype() {
                        gate = gate.to_dtype(tmp.dtype())?;
                    }
                    mul_inplace_same_dtype(&mut tmp, &gate)?;
                }
                if let Err(err) = add_inplace_same_dtype(&mut x, &tmp) {
                    log::trace!("add_inplace msa fallback: {:?}", err);
                    x = x.add(&tmp)?;
                }
            } else {
                let mut gate = reshape_gate(gate_tensor)?;
                if gate.shape().dims() != attn_out.shape().dims() {
                    gate = gate.expand(attn_out.shape().dims())?;
                }
                x = x.add(&attn_out.mul(&gate)?)?;
            }
        } else {
            x = x.add(&attn_out)?;
        }

        if let (Some(extra), Some(gate2)) = (extra, &interm.gate_msa2) {
            if extra.dtype() == DType::BF16 && extra.storage_dtype() == DType::BF16 {
                x = ensure_bf16_accumulator(x, "msa2.residual")?;
                let mut tmp = extra;
                if tmp.storage_dtype() != DType::BF16 {
                    tmp = tmp.to_dtype(DType::BF16)?;
                }
                tmp = promote_to_owning_bf16(tmp, "msa2.tmp")?;
                let gate = reshape_gate(gate2)?;
                if let Err(err) = gate_mul_bf16_inplace(&mut tmp, &gate) {
                    log::trace!("gate msa2_inplace fallback: {:?}", err);
                    let mut gate = gate;
                    if gate.shape().dims() != tmp.shape().dims() {
                        gate = gate.expand(tmp.shape().dims())?;
                    }
                    if gate.dtype() != tmp.dtype() {
                        gate = gate.to_dtype(tmp.dtype())?;
                    }
                    mul_inplace_same_dtype(&mut tmp, &gate)?;
                }
                if let Err(err) = add_inplace_same_dtype(&mut x, &tmp) {
                    log::trace!("add_inplace msa2 fallback: {:?}", err);
                    x = x.add(&tmp)?;
                }
            } else {
                let mut gate = reshape_gate(gate2)?;
                if gate.shape().dims() != extra.shape().dims() {
                    gate = gate.expand(extra.shape().dims())?;
                }
                x = x.add(&extra.mul(&gate)?)?;
            }
        }

        if let (Some(norm2), Some(mlp), Some(shift), Some(scale), Some(gate_mlp)) =
            (&self.norm2, &self.mlp, &interm.shift_mlp, &interm.scale_mlp, &interm.gate_mlp)
        {
            let mut mlp_in = norm2.forward(&x)?;
            if let Some(w) = &norm2.weight {
                let w_vec = w.to_vec()?;
                let min = w_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = w_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                log::info!("DEBUG: norm2 weight min={} max={}", min, max);
            }
            if let Some(b) = &norm2.bias {
                 let b_vec = b.to_vec()?;
                 let min = b_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                 let max = b_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                 log::info!("DEBUG: norm2 bias min={} max={}", min, max);
            }
            if mlp_in.dtype() == DType::BF16 && mlp_in.storage_dtype() == DType::BF16 {
                modulate_inplace(&mut mlp_in, shift, scale)?;
            } else {
                mlp_in = modulate(&mlp_in, shift, scale)?;
            }
            let mlp_out = mlp.forward(&mlp_in)?;
            if mlp_out.dtype() == DType::BF16 && mlp_out.storage_dtype() == DType::BF16 {
                x = ensure_bf16_accumulator(x, "mlp.residual")?;
                let mut tmp = mlp_out;
                tmp = promote_to_owning_bf16(tmp, "mlp.tmp")?;
                let gate = reshape_gate(gate_mlp)?;
                if let Err(err) = gate_mul_bf16_inplace(&mut tmp, &gate) {
                    log::trace!("gate mlp_inplace fallback: {:?}", err);
                    let mut gate = gate;
                    if gate.shape().dims() != tmp.shape().dims() {
                        gate = gate.expand(tmp.shape().dims())?;
                    }
                    if gate.dtype() != tmp.dtype() {
                        gate = gate.to_dtype(tmp.dtype())?;
                    }
                    mul_inplace_same_dtype(&mut tmp, &gate)?;
                }
                add_inplace_same_dtype(&mut x, &tmp)?;
            } else {
                let mut gate = reshape_gate(gate_mlp)?;
                if gate.shape().dims() != mlp_out.shape().dims() {
                    gate = gate.expand(mlp_out.shape().dims())?;
                }
                x = x.add(&mlp_out.mul(&gate)?)?;
            }
        }

        Ok(x)
    }

    pub fn norm1_mut(&mut self) -> &mut LayerNorm {
        &mut self.norm1
    }

    pub fn norm2_mut(&mut self) -> Option<&mut LayerNorm> {
        self.norm2.as_mut()
    }

    pub fn modulation_mut(&mut self) -> &mut Linear {
        &mut self.modulation
    }

    pub fn attn_mut(&mut self) -> &mut SelfAttention {
        &mut self.attn
    }

    pub fn attn2_mut(&mut self) -> Option<&mut SelfAttention> {
        self.attn2.as_mut()
    }

    pub fn mlp_mut(&mut self) -> Option<&mut MLP> {
        self.mlp.as_mut()
    }

    pub fn disable_grads(&mut self) {
        self.attn.disable_grads();
        if let Some(attn2) = self.attn2.as_mut() {
            attn2.disable_grads();
        }
        freeze_linear(&mut self.modulation, "block.modulation");
        if let Some(mlp) = self.mlp.as_mut() {
            let (fc1, fc2) = mlp.fc_layers_mut();
            freeze_linear(fc1, "block.mlp.fc1");
            freeze_linear(fc2, "block.mlp.fc2");
        }
    }
}

pub struct JointTransformerBlock {
    pub context_block: DismantledBlock,
    pub x_block: DismantledBlock,
}

impl JointTransformerBlock {
    pub fn new(
        hidden: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        qk_norm: QkNormKind,
        cond_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::with_flags(
            hidden, num_heads, mlp_ratio, qkv_bias, qk_norm, cond_dim, false, false, device,
        )
    }

    pub fn with_flags(
        hidden: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        qk_norm: QkNormKind,
        cond_dim: usize,
        context_pre_only: bool,
        x_self_attn: bool,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            context_block: DismantledBlock::new(
                hidden,
                cond_dim,
                num_heads,
                mlp_ratio,
                qk_norm,
                qkv_bias,
                context_pre_only,
                false,
                device,
            )?,
            x_block: DismantledBlock::new(
                hidden,
                cond_dim,
                num_heads,
                mlp_ratio,
                qk_norm,
                qkv_bias,
                false,
                x_self_attn,
                device,
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor, cond: &Tensor) -> Result<(Tensor, Tensor)> {
        println!(
            "DEBUG: JointBlock forward: x shape={:?} context shape={:?} cond shape={:?}",
            x.shape().dims(),
            context.shape().dims(),
            cond.shape().dims()
        );
        let mut scratch = ArenaScratch::from_tensor_with_align(x, ARENA_ALIGN);
        let trace = streaming_trace_enabled();

        log::debug!(
            "JointBlock::forward start pre_only={} self_attn={} x_shape={:?} ctx_shape={:?}",
            self.context_block.pre_only,
            self.x_block.self_attn,
            x.shape().dims(),
            context.shape().dims()
        );
        log_tensor_trace(trace, "inputs.x", x);
        log_tensor_trace(trace, "inputs.context", context);
        log_tensor_trace(trace, "inputs.cond", cond);
        log_vram_trace(trace, "joint_block_start");

        // Check inputs for NaNs
        let x_data = x.to_vec()?;
        if x_data.iter().any(|v| v.is_nan()) { log::warn!("JointBlock: x has NaN!"); }
        let ctx_data = context.to_vec()?;
        if ctx_data.iter().any(|v| v.is_nan()) { log::warn!("JointBlock: context has NaN!"); }
        let cond_data = cond.to_vec()?;
        if cond_data.iter().any(|v| v.is_nan()) { log::warn!("JointBlock: cond has NaN!"); }

        let PreAttentionOutput {
            primary: mut context_primary,
            intermediates: context_intermediates,
        } = self.context_block.pre_attention(context, cond, &scratch)?;

        let (x_pre, extra_qkv) = self.x_block.pre_attention_with_secondary(x, cond, &scratch)?;
        let PreAttentionOutput { primary: mut x_primary, intermediates: x_intermediates } = x_pre;
        log_tensor_trace(trace, "context_pre.q", &context_primary.q);
        log_tensor_trace(trace, "context_pre.k", &context_primary.k);
        log_tensor_trace(trace, "context_pre.v", &context_primary.v);
        log_optional_tensor_trace(trace, "context_pre.gate_msa", &context_intermediates.gate_msa);
        log_tensor_trace(trace, "x_pre.q", &x_primary.q);
        log_tensor_trace(trace, "x_pre.k", &x_primary.k);
        log_tensor_trace(trace, "x_pre.v", &x_primary.v);
        log_optional_tensor_trace(trace, "x_pre.gate_msa", &x_intermediates.gate_msa);
        if trace {
            if let Some(extra) = extra_qkv.as_ref() {
                log_tensor_trace(trace, "x_pre.extra_q", &extra.q);
                log_tensor_trace(trace, "x_pre.extra_k", &extra.k);
                log_tensor_trace(trace, "x_pre.extra_v", &extra.v);
            }
            log_vram_trace(trace, "after_pre_attention");
        }

        let q = Tensor::cat(&[&x_primary.q, &context_primary.q], 2)?;
        let k = Tensor::cat(&[&x_primary.k, &context_primary.k], 2)?;
        let v = Tensor::cat(&[&x_primary.v, &context_primary.v], 2)?;
        drop(context_primary);
        drop(x_primary);
        log_tensor_trace(trace, "joint.q_concat", &q);
        log_tensor_trace(trace, "joint.k_concat", &k);
        log_tensor_trace(trace, "joint.v_concat", &v);
        log_vram_trace(trace, "after_joint_qkv_cat");

        log::trace!(
            "JointBlock::forward sdpa launch: q={:?} k={:?} v={:?}",
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims()
        );
        let attn_all = sdpa::forward(&q, &k, &v, None).map_err(|err| {
            log::error!(
                "sdpa::forward failed: block context_pre_only={} x_self_attn={} q={:?} k={:?} v={:?} err={:?}",
                self.context_block.pre_only,
                self.x_block.self_attn,
                q.shape().dims(),
                k.shape().dims(),
                v.shape().dims(),
                err
            );
            err
        })?;
        drop(q);
        drop(k);
        drop(v);
        log::debug!(
            "JointBlock::forward sdpa ok pre_only={} self_attn={} attn_all={:?}",
            self.context_block.pre_only,
            self.x_block.self_attn,
            attn_all.shape().dims()
        );
        log_tensor_trace(trace, "joint.attn_all", &attn_all);
        log_vram_trace(trace, "after_joint_sdpa");
        let heads = attn_all.shape().dims()[1];
        let ctx_tokens = context.shape().dims()[1];
        let x_tokens = x.shape().dims()[1];

        // Diffusers order: [image, text]
        // So first x_tokens are image, next ctx_tokens are text
        let x_heads = attn_all.slice(&[
            (0, attn_all.shape().dims()[0]),
            (0, heads),
            (0, x_tokens),
            (0, self.x_block.attn.head_dim),
        ])?;
        let context_heads = attn_all.slice(&[
            (0, attn_all.shape().dims()[0]),
            (0, heads),
            (x_tokens, x_tokens + ctx_tokens),
            (0, self.context_block.attn.head_dim),
        ])?;
        drop(attn_all);
        log_tensor_trace(trace, "joint.context_heads", &context_heads);
        log_tensor_trace(trace, "joint.x_heads", &x_heads);

        let context_proj = self.context_block.project_primary(&context_heads, &scratch)?;
        let x_proj = self.x_block.project_primary(&x_heads, &scratch)?;
        drop(context_heads);
        drop(x_heads);
        log::debug!("JointBlock::forward projections ok");
        log_tensor_trace(trace, "joint.context_proj", &context_proj);
        log_tensor_trace(trace, "joint.x_proj", &x_proj);

        let secondary_proj = if let Some(sec) = extra_qkv {
            let ctx2 = sdpa::forward(&sec.q, &sec.k, &sec.v, None)?;
            log::debug!("JointBlock::forward secondary sdpa ok");
            if trace {
                log_tensor_trace(trace, "joint.secondary_ctx2", &ctx2);
            }
            let proj = self.x_block.project_secondary(&ctx2, &scratch)?;
            drop(ctx2);
            if trace {
                log_tensor_trace(trace, "joint.secondary_proj", &proj);
            }
            Some(proj)
        } else if x_intermediates.recompute_secondary {
            self.x_block.recompute_secondary_projection(x, &scratch, &x_intermediates, trace)?
        } else {
            None
        };
        if trace {
            log_optional_tensor_trace(trace, "joint.secondary_proj_opt", &secondary_proj);
            log_vram_trace(trace, "after_secondary_path");
        }

        let context_out =
            self.context_block.post_attention(context_proj, None, context_intermediates)?;
        let x_out = self.x_block.post_attention(x_proj, secondary_proj, x_intermediates)?;
        log::debug!("JointBlock::forward post_attention ok");
        log_tensor_trace(trace, "outputs.x_out", &x_out);
        log_tensor_trace(trace, "outputs.context_out", &context_out);
        log_vram_trace(trace, "joint_block_end");

        Ok((x_out, context_out))
    }

    pub fn context_block_mut(&mut self) -> &mut DismantledBlock {
        &mut self.context_block
    }

    pub fn context_block(&self) -> &DismantledBlock {
        &self.context_block
    }

    pub fn x_block_mut(&mut self) -> &mut DismantledBlock {
        &mut self.x_block
    }

    pub fn x_block(&self) -> &DismantledBlock {
        &self.x_block
    }

    pub fn disable_grads(&mut self) {
        self.context_block.disable_grads();
        self.x_block.disable_grads();
    }
}

impl DismantledBlock {
    fn recompute_secondary_projection(
        &self,
        x: &Tensor,
        scratch: &ArenaScratch,
        interm: &BlockIntermediates,
        trace: bool,
    ) -> Result<Option<Tensor>> {
        if !self.self_attn {
            return Ok(None);
        }
        let shift2 = match &interm.shift_msa2 {
            Some(t) => t,
            None => return Ok(None),
        };
        let scale2 = match &interm.scale_msa2 {
            Some(t) => t,
            None => return Ok(None),
        };
        let mut attn2_in = self.norm1.forward(x)?;
        if attn2_in.dtype() == DType::BF16 && attn2_in.storage_dtype() == DType::BF16 {
            modulate_inplace(&mut attn2_in, shift2, scale2)?;
        } else {
            attn2_in = modulate(&attn2_in, shift2, scale2)?;
        }
        if trace {
            log_tensor_trace(trace, "joint.secondary_attn2_in", &attn2_in);
        }
        let attn2 = match &self.attn2 {
            Some(attn) => attn,
            None => return Ok(None),
        };
        let qkv = attn2.pre_attention(&attn2_in, scratch)?;
        if trace {
            log_tensor_trace(trace, "joint.secondary_q", &qkv.q);
            log_tensor_trace(trace, "joint.secondary_k", &qkv.k);
            log_tensor_trace(trace, "joint.secondary_v", &qkv.v);
        }
        drop(attn2_in);
        let ctx2 = sdpa::forward(&qkv.q, &qkv.k, &qkv.v, None)?;
        if trace {
            log_tensor_trace(trace, "joint.secondary_ctx2", &ctx2);
        }
        drop(qkv);
        let proj = self.project_secondary(&ctx2, scratch)?;
        drop(ctx2);
        if trace {
            log_tensor_trace(trace, "joint.secondary_proj", &proj);
        }
        Ok(Some(proj))
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
impl DismantledBlock {
    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &BlockSnapshot) -> Result<()> {
        if snapshot.hidden != self.hidden || snapshot.num_heads != self.num_heads {
            return Err(Error::InvalidInput("Block snapshot configuration mismatch".into()));
        }
        if snapshot.pre_only != self.pre_only || snapshot.self_attn != self.self_attn {
            return Err(Error::InvalidInput(format!(
                "Block snapshot flag mismatch (snapshot pre_only={} self_attn={}, block pre_only={} self_attn={})",
                snapshot.pre_only, snapshot.self_attn, self.pre_only, self.self_attn
            )));
        }
        self.norm1.apply_cpu_snapshot(&snapshot.norm1)?;
        self.attn.apply_cpu_snapshot(&snapshot.attn)?;
        match (&mut self.attn2, &snapshot.attn2) {
            (Some(attn), Some(snap)) => attn.apply_cpu_snapshot(snap)?,
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput("Block snapshot attn2 presence mismatch".into()));
            }
        }
        match (&mut self.norm2, &snapshot.norm2) {
            (Some(norm), Some(snap)) => norm.apply_cpu_snapshot(snap)?,
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput("Block snapshot norm2 presence mismatch".into()));
            }
        }
        match (&mut self.mlp, &snapshot.mlp) {
            (Some(mlp), Some(snap)) => mlp.apply_cpu_snapshot(snap)?,
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput("Block snapshot MLP presence mismatch".into()));
            }
        }
        snapshot.modulation.apply_to(&mut self.modulation)?;
        Ok(())
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
impl JointTransformerBlock {
    pub(crate) fn cpu_snapshot(&self) -> Result<JointBlockSnapshot> {
        Ok(JointBlockSnapshot {
            context: self.context_block.cpu_snapshot()?,
            x: self.x_block.cpu_snapshot()?,
        })
    }

    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &JointBlockSnapshot) -> Result<()> {
        self.context_block.apply_cpu_snapshot(&snapshot.context)?;
        self.x_block.apply_cpu_snapshot(&snapshot.x)?;
        Ok(())
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
impl DismantledBlock {
    pub(crate) fn cpu_snapshot(&self) -> Result<BlockSnapshot> {
        Ok(BlockSnapshot {
            hidden: self.hidden,
            num_heads: self.num_heads,
            pre_only: self.pre_only,
            self_attn: self.self_attn,
            norm1: self.norm1.cpu_snapshot()?,
            attn: self.attn.cpu_snapshot()?,
            attn2: self.attn2.as_ref().map(|a| a.cpu_snapshot()).transpose()?,
            norm2: self.norm2.as_ref().map(|n| n.cpu_snapshot()).transpose()?,
            mlp: self.mlp.as_ref().map(|m| m.cpu_snapshot()).transpose()?,
            modulation: LinearSnapshot::from_linear(&self.modulation)?,
        })
    }
}

pub struct RoPE2D {
    inv_freq: Tensor,
    max_size: usize,
    hidden_size: usize,
}

impl RoPE2D {
    pub fn new(hidden: usize, num_heads: usize, max_size: usize, device: &Device) -> Result<Self> {
        let _ = num_heads;
        if hidden % 2 != 0 {
            return Err(Error::InvalidShape(format!(
                "positional embedding hidden size {} must be even",
                hidden
            )));
        }
        let axis_dim = hidden / 2;
        if axis_dim % 2 != 0 {
            return Err(Error::InvalidShape(format!(
                "positional embedding axis dim {} must be even",
                axis_dim
            )));
        }
        let half_axis = axis_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_axis);
        for i in 0..half_axis {
            let exponent = (i as f32) / (half_axis as f32);
            inv_freq.push(1.0 / 10000_f32.powf(exponent));
        }
        Ok(Self {
            inv_freq: Tensor::from_vec(
                inv_freq,
                Shape::from_dims(&[half_axis]),
                device.cuda_device().clone(),
            )?,
            max_size,
            hidden_size: hidden,
        })
    }

    pub fn embed(
        &self,
        grid_h: usize,
        grid_w: usize,
        batch: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        if grid_h > self.max_size || grid_w > self.max_size {
            return Err(Error::InvalidShape(format!(
                "requested grid {}x{} exceeds max size {}",
                grid_h, grid_w, self.max_size
            )));
        }

        let axis_dim = self.hidden_size / 2;
        let half_axis = axis_dim / 2;
        if half_axis == 0 {
            return Err(Error::InvalidShape(
                "positional embedding axis dimension must be positive".into(),
            ));
        }

        let top = (self.max_size - grid_h) / 2;
        let left = (self.max_size - grid_w) / 2;

        let h_positions =
            Tensor::arange(top as f32, (top + grid_h) as f32, 1.0_f32, device.cuda_device().clone())?;
        let w_positions =
            Tensor::arange(left as f32, (left + grid_w) as f32, 1.0_f32, device.cuda_device().clone())?;

        let target_device = device.cuda_device();
        if !Arc::ptr_eq(self.inv_freq.device(), &target_device) {
            return Err(Error::InvalidInput(
                "RoPE2D frequency buffer lives on a different device; reinitialize MMDiT".into(),
            ));
        }
        let inv = self.inv_freq.clone();

        let h_angles = h_positions.unsqueeze(1)?.mul(&inv.unsqueeze(0)?)?;
        let w_angles = w_positions.unsqueeze(1)?.mul(&inv.unsqueeze(0)?)?;

        let mut h_embed = Tensor::cat(&[&h_angles.sin()?, &h_angles.cos()?], 1)?;
        h_embed.reshape_inplace(&[grid_h, 1, axis_dim])?;
        let h_embed = h_embed.expand(&[grid_h, grid_w, axis_dim])?;

        let mut w_embed = Tensor::cat(&[&w_angles.sin()?, &w_angles.cos()?], 1)?;
        w_embed.reshape_inplace(&[1, grid_w, axis_dim])?;
        let w_embed = w_embed.expand(&[grid_h, grid_w, axis_dim])?;

        let mut pos = Tensor::cat(&[&w_embed, &h_embed], 2)?;
        pos.reshape_inplace(&[1, grid_h * grid_w, self.hidden_size])?;

        if batch > 1 {
            pos = pos.expand(&[batch, grid_h * grid_w, self.hidden_size])?;
        }

        if pos.dtype() != dtype {
            pos = pos.to_dtype(dtype)?;
        }
        Ok(pos)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn cpu_snapshot(&self) -> Result<F32CpuSnapshot> {
        F32CpuSnapshot::from_tensor(&self.inv_freq)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &F32CpuSnapshot) -> Result<()> {
        let device = self.inv_freq.device().clone();
        let tensor = snapshot.to_cuda_tensor(device)?;
        self.inv_freq = tensor;
        Ok(())
    }
}

pub struct AdaLayerNorm {
    norm: LayerNorm,
    linear: Linear,
}

impl AdaLayerNorm {
    pub fn new(hidden: usize, cond_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::new(vec![hidden], 1e-6, device.cuda_device().clone())?,
            linear: Linear::new_zeroed(cond_dim, hidden * 2, true, &device.cuda_device())?,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(x)?;
        let params = self.linear.forward(cond)?;
        let chunks = params.chunk(2, params.shape().rank() - 1)?;
        let shift = chunks[0].clone();
        let scale = chunks[1].clone();
        modulate(&normed, &shift, &scale)
    }

    pub fn modulation_linear(&self) -> &Linear {
        &self.linear
    }

    pub fn modulation_linear_mut(&mut self) -> &mut Linear {
        &mut self.linear
    }

    pub fn norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.norm
    }
}

pub struct FinalLayer {
    norm: LayerNorm,
    modulation: Linear,
    proj: Linear,
    patch_size: usize,
    out_channels: usize,
}

impl FinalLayer {
    pub fn new(
        hidden_size: usize,
        patch_size: usize,
        out_channels: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::new(vec![hidden_size], 1e-6, device.cuda_device().clone())?,
            modulation: Linear::new_zeroed(
                hidden_size,
                hidden_size * 2,
                true,
                &device.cuda_device(),
            )?,
            proj: Linear::new_zeroed(
                hidden_size,
                patch_size * patch_size * out_channels,
                true,
                &device.cuda_device(),
            )?,
            patch_size,
            out_channels,
        })
    }

    fn apply_modulation(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let cond_dim = cond.shape().dims().last().copied().ok_or_else(|| {
            Error::InvalidShape("conditioning tensor missing last dimension".into())
        })?;
        if cond_dim != self.modulation.in_features() {
            return Err(Error::InvalidShape(format!(
                "FinalLayer expected conditioning dim {}, got {}",
                self.modulation.in_features(),
                cond_dim
            )));
        }
        let params = self.modulation.forward(&cond.silu()?)?;
        let chunks = params.chunk(2, params.shape().rank() - 1)?;
        let shift = chunks[0].clone();
        let scale = chunks[1].clone();
        modulate(x, &shift, &scale)
    }

    fn unpatchify(&self, x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let dims = x.shape().dims();
        if dims.len() != 3 {
            return Err(Error::InvalidShape(format!(
                "FinalLayer expects [B, HW, hidden], got {:?}",
                dims
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        let hidden = dims[2];

        let patch_dim = self.patch_size * self.patch_size * self.out_channels;
        if hidden != patch_dim {
            return Err(Error::InvalidShape(format!(
                "FinalLayer projection dim {} does not match patch {}^2 * channels {} = {}",
                hidden, self.patch_size, self.out_channels, patch_dim
            )));
        }

        let grid_h = (height + self.patch_size - 1) / self.patch_size;
        let grid_w = (width + self.patch_size - 1) / self.patch_size;
        if tokens != grid_h * grid_w {
            return Err(Error::InvalidShape(format!(
                "FinalLayer expected {} tokens for grid {}x{}, got {}",
                grid_h * grid_w,
                grid_h,
                grid_w,
                tokens
            )));
        }

        let reshaped = x.reshape(&[
            batch,
            grid_h,
            grid_w,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        ])?;
        let mut permuted = reshaped.permute(&[0, 5, 1, 3, 2, 4])?;
        permuted.reshape_inplace(&[
            batch,
            self.out_channels,
            grid_h * self.patch_size,
            grid_w * self.patch_size,
        ])?;
        Ok(permuted)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cond: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let normed = self.norm.forward(x)?;
        let modulated = self.apply_modulation(&normed, cond)?;
        let projected = self.proj.forward(&modulated)?;
        let mut out = self.unpatchify(&projected, height, width)?;
        let out_dims = out.shape().dims();
        if out_dims.len() != 4 {
            return Err(Error::InvalidShape(format!(
                "FinalLayer expected unpatchified tensor rank 4, got {:?}",
                out_dims
            )));
        }
        if out_dims[2] > height || out_dims[3] > width {
            out = out.slice(&[(0, out_dims[0]), (0, out_dims[1]), (0, height), (0, width)])?;
        }
        Ok(out)
    }

    pub fn norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.norm
    }

    pub fn norm(&self) -> &LayerNorm {
        &self.norm
    }

    pub fn modulation_mut(&mut self) -> &mut Linear {
        &mut self.modulation
    }

    pub fn modulation(&self) -> &Linear {
        &self.modulation
    }

    pub fn proj_mut(&mut self) -> &mut Linear {
        &mut self.proj
    }

    pub fn proj(&self) -> &Linear {
        &self.proj
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn cpu_snapshot(&self) -> Result<FinalLayerSnapshot> {
        Ok(FinalLayerSnapshot {
            norm: self.norm.cpu_snapshot()?,
            modulation: LinearSnapshot::from_linear(&self.modulation)?,
            proj: LinearSnapshot::from_linear(&self.proj)?,
            patch_size: self.patch_size,
            out_channels: self.out_channels,
        })
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub(crate) fn apply_cpu_snapshot(&mut self, snapshot: &FinalLayerSnapshot) -> Result<()> {
        if snapshot.patch_size != self.patch_size || snapshot.out_channels != self.out_channels {
            return Err(Error::InvalidInput("FinalLayer snapshot configuration mismatch".into()));
        }
        self.norm.apply_cpu_snapshot(&snapshot.norm)?;
        snapshot.modulation.apply_to(&mut self.modulation)?;
        snapshot.proj.apply_to(&mut self.proj)?;
        Ok(())
    }
}

pub struct MMDiT {
    pub config: MMDiTConfig,
    x_embedder: PatchEmbed,
    t_embedder: TimestepEmbedder,
    y_embedder: Option<VectorEmbedder>,
    context_embedder: Linear,
    pub blocks: Vec<JointTransformerBlock>,
    final_layer: FinalLayer,
    pos_embed: RoPE2D,
}

impl MMDiT {
    pub fn new(config: MMDiTConfig, device: &Device) -> Result<Self> {
        let mut blocks = Vec::new();
        let cond_dim = config.hidden_size;
        for idx in 0..config.depth {
            let context_pre_only = idx == config.depth - 1;
            let x_self_attn = config.x_self_attn_layers.map(|limit| idx <= limit).unwrap_or(false);
            blocks.push(JointTransformerBlock::with_flags(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                config.qkv_bias,
                config.qk_norm,
                cond_dim,
                context_pre_only,
                x_self_attn,
                device,
            )?);
        }

        Ok(Self {
            config: config.clone(),
            x_embedder: PatchEmbed::new(
                None,
                config.patch_size,
                config.in_channels,
                config.hidden_size,
                true,
                true,
                true,
                device,
            )?,
            t_embedder: TimestepEmbedder::new(
                config.hidden_size,
                config.frequency_embedding_size,
                device,
            )?,
            y_embedder: if let Some(dim) = config.pooled_dim {
                Some(VectorEmbedder::new(dim, config.hidden_size, device)?)
            } else {
                None
            },
            context_embedder: Linear::new_zeroed(
                config.context_dim,
                config.hidden_size,
                true,
                &device.cuda_device(),
            )?,
            blocks,
            final_layer: FinalLayer::new(
                config.hidden_size,
                config.patch_size,
                config.out_channels,
                device,
            )?,
            pos_embed: RoPE2D::new(
                config.hidden_size,
                config.num_heads,
                config.pos_embed_max_size,
                device,
            )?,
        })
    }

    pub fn forward(
        &self,
        latents: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
        pooled: Option<&Tensor>,
    ) -> Result<Tensor> {
        let latent_dims = latents.shape().dims();
        if latent_dims.len() != 4 {
            return Err(Error::InvalidShape(format!(
                "MMDiT expects latent input [B, C, H, W], got {:?}",
                latent_dims
            )));
        }
        let batch = latent_dims[0];
        let in_channels = latent_dims[1];
        let height = latent_dims[2];
        let width = latent_dims[3];
        if in_channels != self.config.in_channels {
            return Err(Error::InvalidShape(format!(
                "latent channel dim {} does not match config {}",
                in_channels, self.config.in_channels
            )));
        }

        let grid_h = (height + self.config.patch_size - 1) / self.config.patch_size;
        let grid_w = (width + self.config.patch_size - 1) / self.config.patch_size;

        // Conv2d expects NCHW, so we pass latents directly (already NCHW)
        log::info!(
            "PatchEmbed latents shape {:?}, weight shape {:?}",
            latents.shape().dims(),
            self.x_embedder.weight_shape()
        );
        let mut x_tokens = self.x_embedder.forward(latents)?;
        let token_dims = x_tokens.shape().dims();
        if token_dims.len() != 3 || token_dims[1] != grid_h * grid_w {
            return Err(Error::InvalidShape(format!(
                "PatchEmbed produced token shape {:?}, expected [{} tokens]",
                token_dims,
                grid_h * grid_w
            )));
        }
        log::debug!("x_tokens dtype {:?} storage {:?}", x_tokens.dtype(), x_tokens.storage_dtype());
        let pos_embed = self.pos_embed.embed(
            grid_h,
            grid_w,
            batch,
            &Device::from(latents.device().clone()),
            x_tokens.dtype(),
        )?;
        x_tokens = x_tokens.add(&pos_embed)?;

        let mut cond = self.t_embedder.forward(timesteps)?;
        if let (Some(embedder), Some(pooled_tensor)) = (self.y_embedder.as_ref(), pooled) {
            let mut pooled_proj = embedder.forward(pooled_tensor)?;
            let cond_dims = cond.shape().dims();
            let pooled_dims = pooled_proj.shape().dims();
            if pooled_dims.len() == 2
                && pooled_dims[0] == 1
                && cond_dims.len() == 2
                && cond_dims[0] != 1
            {
                pooled_proj = pooled_proj.expand(&[cond_dims[0], pooled_dims[1]])?;
            }
            cond = cond.add(&pooled_proj)?;
        }

        let mut context_tokens = self.context_embedder.forward(context)?;
        log::debug!(
            "context_tokens dtype {:?} storage {:?}",
            context_tokens.dtype(),
            context_tokens.storage_dtype()
        );
        
        // TODO: arena scratch re-enable once BF16 arena copy supports mutable slices.
        
        // Check inputs before loop
        println!("DEBUG: MMDiT: Checking x_tokens...");
        let _ = x_tokens.to_vec()?;
        println!("DEBUG: MMDiT: Checking context_tokens...");
        let _ = context_tokens.to_vec()?;
        println!("DEBUG: MMDiT: Checking cond...");
        let _ = cond.to_vec()?;
        println!("DEBUG: MMDiT: Inputs OK. Starting loop...");

        for block in &self.blocks {
            let (new_x, new_context) = block.forward(&x_tokens, &context_tokens, &cond)?;
            x_tokens = new_x;
            context_tokens = new_context;
        }

        self.final_layer.forward(&x_tokens, &cond, height, width)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub fn cpu_snapshot(&self) -> Result<MmditCpuSnapshot> {
        let patch_embed = self.x_embedder.cpu_snapshot()?;
        let timestep_embedder = self.t_embedder.cpu_snapshot()?;
        let vector_embedder = self.y_embedder.as_ref().map(|y| y.cpu_snapshot()).transpose()?;
        let context_embedder = LinearSnapshot::from_linear(&self.context_embedder)?;
        let blocks =
            self.blocks.iter().map(|block| block.cpu_snapshot()).collect::<Result<Vec<_>>>()?;
        let final_layer = self.final_layer.cpu_snapshot()?;
        let pos_frequencies = self.pos_embed.cpu_snapshot()?;

        Ok(MmditCpuSnapshot {
            config: self.config.clone(),
            patch_embed,
            timestep_embedder,
            vector_embedder,
            context_embedder,
            blocks,
            final_layer,
            pos_frequencies,
        })
    }

    pub fn blocks_mut(&mut self) -> &mut [JointTransformerBlock] {
        &mut self.blocks
    }

    pub fn config(&self) -> &MMDiTConfig {
        &self.config
    }

    pub fn final_layer_mut(&mut self) -> &mut FinalLayer {
        &mut self.final_layer
    }

    pub fn final_layer(&self) -> &FinalLayer {
        &self.final_layer
    }

    pub fn context_embedder_mut(&mut self) -> &mut Linear {
        &mut self.context_embedder
    }

    pub fn context_embedder(&self) -> &Linear {
        &self.context_embedder
    }

    pub fn timestep_embedder_mut(&mut self) -> &mut TimestepEmbedder {
        &mut self.t_embedder
    }

    pub fn timestep_embedder(&self) -> &TimestepEmbedder {
        &self.t_embedder
    }

    pub fn vector_embedder_mut(&mut self) -> Option<&mut VectorEmbedder> {
        self.y_embedder.as_mut()
    }

    pub fn vector_embedder(&self) -> Option<&VectorEmbedder> {
        self.y_embedder.as_ref()
    }

    pub fn patch_embed_mut(&mut self) -> &mut PatchEmbed {
        &mut self.x_embedder
    }

    pub fn patch_embed(&self) -> &PatchEmbed {
        &self.x_embedder
    }

    pub fn pos_embed_mut(&mut self) -> &mut RoPE2D {
        &mut self.pos_embed
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub fn apply_cpu_snapshot(&mut self, snapshot: &MmditCpuSnapshot) -> Result<()> {
        if snapshot.config.depth != self.config.depth {
            return Err(Error::InvalidInput("MMDiT snapshot depth mismatch".into()));
        }
        self.config = snapshot.config.clone();
        self.x_embedder.apply_cpu_snapshot(&snapshot.patch_embed)?;
        self.t_embedder.apply_cpu_snapshot(&snapshot.timestep_embedder)?;
        match (&mut self.y_embedder, &snapshot.vector_embedder) {
            (Some(embedder), Some(snap)) => embedder.apply_cpu_snapshot(snap)?,
            (None, None) => {}
            _ => {
                return Err(Error::InvalidInput(
                    "MMDiT snapshot vector embedder presence mismatch".into(),
                ));
            }
        }
        snapshot.context_embedder.apply_to(&mut self.context_embedder)?;
        if self.blocks.len() != snapshot.blocks.len() {
            return Err(Error::InvalidInput("MMDiT snapshot block count mismatch".into()));
        }
        for (block, snap) in self.blocks.iter_mut().zip(snapshot.blocks.iter()) {
            block.apply_cpu_snapshot(snap)?;
        }
        self.final_layer.apply_cpu_snapshot(&snapshot.final_layer)?;
        self.pos_embed.apply_cpu_snapshot(&snapshot.pos_frequencies)?;

        Ok(())
    }
}
