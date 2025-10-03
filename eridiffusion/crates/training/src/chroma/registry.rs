use eridiffusion_core::{Error, Result, TensorDebugExt};
use flame_core::{Tensor, DType};
use crate::streaming::DeviceWeights;
use super::lora::LoRALinear;
use super::guards::guard;
use super::keymap::ChromaKeyMap;
use crate::tensor_utils::softmax_stable;
#[cfg(feature = "bf16_u16")]
use flame_core::bf16_elementwise as be;

#[derive(Clone)]
pub struct Cond {
    pub text_hidden: Tensor,
    pub sigma: Tensor,
    // NOTE [Mask dtype]: prefer Bool/u8 for masks; cast to FP32 only at the op that consumes it
    // (e.g., add ~-1e4 before softmax). Avoid storing masks as FP32.
    pub mask_lat: Option<Tensor>,
}

pub trait Block: Send + Sync {
    fn apply(&self, x: &Tensor, c: &Cond, w: &DeviceWeights, l: &Vec<LoRALinear>) -> Result<Tensor>;
}

pub struct CombinedBlock { pub btd: bool, pub idx: usize, pub tripwires: bool, pub route_out: bool, pub route_in: bool }

fn to_dim(x: i64, rank: usize) -> usize {
    if x < 0 { (rank as i64 + x) as usize } else { x as usize }
}

// stable softmax moved to crate::tensor_utils

fn assert_finite(name: &str, t: &Tensor) -> Result<()> {
    // Gate host check behind TRIPWIRES to avoid staging on hot paths
    let trip = std::env::var("TRIPWIRES").ok().map(|v| v != "0").unwrap_or(false);
    if !trip { return Ok(()); }
    let f = t.to_dtype(DType::F32)?;
    // Debug-only scan
    let v = f.to_vec()?;
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::Training(format!("NaN/Inf detected at {}", name)));
    }
    Ok(())
}

fn clamp_symmetric(x: &Tensor, limit: f32) -> Result<Tensor> {
    // Implement clamp(-limit, limit) using only GPU-friendly ops:
    // min(x, limit) = -max(-x, -limit); then max(., -limit)
    let dev = x.device().clone();
    let nlim = Tensor::from_scalar(-limit, dev.clone())?; // scalar broadcasts
    let x_min = x.neg()?.maximum(&nlim)?.neg()?; // min(x, limit)
    let x_clamped = x_min.maximum(&nlim)?; // >= -limit
    Ok(x_clamped)
}

impl Block for CombinedBlock {
    fn apply(&self, x: &Tensor, _c: &Cond, w: &DeviceWeights, l: &Vec<LoRALinear>) -> Result<Tensor> {
        let (wq, wk, wv, wo) = (&w.tensors[0], &w.tensors[1], &w.tensors[2], &w.tensors[3]);
        let (f1_raw, f2_raw) = (&w.tensors[4], &w.tensors[5]);
        debug_assert_eq!(wq.dtype(), DType::BF16, "q weight must be BF16");
        debug_assert_eq!(wk.dtype(), DType::BF16, "k weight must be BF16");
        debug_assert_eq!(wv.dtype(), DType::BF16, "v weight must be BF16");
        debug_assert_eq!(wo.dtype(), DType::BF16, "o weight must be BF16");
        debug_assert_eq!(f1_raw.dtype(), DType::BF16, "fc1 weight must be BF16");
        debug_assert_eq!(f2_raw.dtype(), DType::BF16, "fc2 weight must be BF16");
        let f1 = f1_raw.transpose()?; // [3072,12288]
        let f2 = f2_raw.transpose()?; // [12288,3072]

        // Optional fine-grained NaN/Inf tracing
        let trace = std::env::var("TRACE_NAN").ok().map(|v| v != "0").unwrap_or(false);
        let check = |name: &str, t: &Tensor| -> Result<()> {
            if trace { t.debug_check(name)?; }
            Ok(())
        };

        // Helper for [B,T,D] x [D,Out] using flatten
        let linear_bt = |bt: &Tensor, wmat: &Tensor| -> Result<Tensor> {
            let dims = bt.shape().dims().to_vec();
            let (b,t,d) = (dims[0], dims[1], dims[2]);
            let flat = bt.reshape(&[b*t, d])?;
            // Prefer BF16-tagged matmul when tensors are BF16 to avoid silent F32 promotions
            let out = if flat.dtype() == DType::BF16 && wmat.dtype() == DType::BF16 {
                debug_assert!(flat.device().ordinal() == wmat.device().ordinal());
                flat.matmul_bf16(&wmat)?
            } else {
                flat.matmul(&wmat)?
            };
            let od = out.shape().dims()[1];
            Ok(out.reshape(&[b, t, od])?)
        };
        // Helper to apply LoRA delta on [B,T,D] by flattening (no FP32 expand)
        let lora_delta_bt = |bt: &Tensor, lora: &LoRALinear| -> Result<Tensor> {
            let dims = bt.shape().dims().to_vec();
            let (b,t,d) = (dims[0], dims[1], dims[2]);
            let flat = bt.reshape(&[b*t, d])?;
            let delta = lora.forward_delta(&flat)?;
            Ok(delta.reshape(&[b, t, delta.shape().dims()[1]])?)
        };

        // Normalize last-dim RMS with epsilon (keeps magnitude in check per token/vector)
        let normalize_last_dim = |t: &Tensor| -> Result<Tensor> {
            let dims = t.shape().dims().to_vec();
            let last = dims.last().copied().unwrap_or(1) as f32;
            let eps = Tensor::from_scalar(1e-6f32, t.device().clone())?;
            let t32 = t.to_dtype(DType::F32)?;
            let rms = t32.square()?.sum_dim_keepdim(dims.len()-1)?.div_scalar(last)?.sqrt()?.maximum(&eps)?;
            Ok(t32.div(&rms)?)
        };

        // BF16 elementwise helpers (broadcast-safe) where applicable, fall back to engine ops
        #[inline]
        fn add_tensors(x: &Tensor, y: &Tensor) -> Result<Tensor> {
            #[cfg(feature = "bf16_u16")]
            { if x.dtype()==DType::BF16 && y.dtype()==DType::BF16 { return be::add_bf16(x,y).map_err(Error::from); } }
            x.add(y).map_err(Error::from)
        }
        #[inline]
        fn add_broadcast(x: &Tensor, y: &Tensor) -> Result<Tensor> {
            if x.shape().dims() == y.shape().dims() {
                return add_tensors(x, y);
            }
            let y_b = y.broadcast_to(&x.shape())?;
            add_tensors(x, &y_b)
        }
        #[inline]
        fn mul_tensors(x: &Tensor, y: &Tensor) -> Result<Tensor> {
            #[cfg(feature = "bf16_u16")]
            { if x.dtype()==DType::BF16 && y.dtype()==DType::BF16 { return be::mul_bf16(x,y).map_err(Error::from); } }
            x.mul(y).map_err(Error::from)
        }

        if self.btd && x.shape().dims().len() == 3 {
            // Proper [B,T,D] attention path
            let xn = normalize_last_dim(x)?;
            debug_assert_eq!(xn.dtype(), DType::F32, "normalize_last_dim returns F32 for stable math");
            let q0 = linear_bt(&xn, wq)?; check(&format!("blk{}.q0", self.idx), &q0)?; let q = q0.add(&lora_delta_bt(&xn, &l[0])?)?; check(&format!("blk{}.q", self.idx), &q)?; // [B,T,D]
            let k0 = linear_bt(&xn, wk)?; check(&format!("blk{}.k0", self.idx), &k0)?; let k = k0.add(&lora_delta_bt(&xn, &l[1])?)?; check(&format!("blk{}.k", self.idx), &k)?;
            let v0 = linear_bt(&xn, wv)?; check(&format!("blk{}.v0", self.idx), &v0)?; let v = v0.add(&lora_delta_bt(&xn, &l[2])?)?; check(&format!("blk{}.v", self.idx), &v)?;
            assert_finite("Q", &q)?; assert_finite("K", &k)?; assert_finite("V", &v)?;
            // Compute attention scores (keep current dtype; Flame storage is F32-backed)
            let q32 = q.clone();
            let k32 = k.clone();
            // Normalize Q/K per-token RMS to prevent overflow in dot product
            let d_last = q32.shape().dims()[2] as f32;
            let eps = Tensor::from_scalar(1e-6f32, q32.device().clone())?;
            let q_rms = q32.square()?.sum_dim_keepdim(2)?.div_scalar(d_last)?.sqrt()?.maximum(&eps)?; // [B,T,1]
            let k_rms = k32.square()?.sum_dim_keepdim(2)?.div_scalar(d_last)?.sqrt()?.maximum(&eps)?; // [B,T,1]
            let qn = q32.div(&q_rms)?;
            let kn = k32.div(&k_rms)?;
            let scale = (3072f32).sqrt().recip();
            let kt = kn.transpose_dims(1, 2)?; // [B,D,T]
            let mut scores = qn.bmm(&kt)?; check(&format!("blk{}.scores_raw", self.idx), &scores)?;    // [B,T,T]
            scores = scores.mul_scalar(scale)?;
            scores = clamp_symmetric(&scores, 30.0f32)?;
            assert_finite("scores_pre_softmax", &scores)?;
            let attn = softmax_stable(&scores, -1)?;   // [B,T,T]
            attn.debug_check("attn.probs")?;
            if self.tripwires { guard(&format!("blk{}.attn.probs", self.idx), &attn)?; }
            let v32 = v.to_dtype(DType::F32)?;
            let ctx = attn.to_dtype(DType::F32)?.bmm(&v32)?; check(&format!("blk{}.ctx", self.idx), &ctx)?;         // [B,T,D]
            ctx.debug_check("attn.ctx")?;
            let o0 = linear_bt(&ctx, wo)?; check(&format!("blk{}.o0", self.idx), &o0)?;
            let o_pre  = add_broadcast(&o0, &lora_delta_bt(&ctx, &l[3])?)?; // [B,T,D]
            // Clamp attn output to prevent MLP blow-up
            let o = clamp_symmetric(&o_pre, 1.0e3f32)?; check(&format!("blk{}.o", self.idx), &o)?;
            o.debug_check("attn.out")?;
            if self.tripwires { guard(&format!("blk{}.attn_out", self.idx), &o)?; }
            let h1_pre = linear_bt(&o, &f1)?; check(&format!("blk{}.h1", self.idx), &h1_pre)?;    // [B,T,12288]
            let h1 = {
                #[cfg(feature = "bf16_u16")]
                { use flame_core::bf16_clamp::clamp_bf16; clamp_bf16(&h1_pre, -1.0e3f32, 1.0e3f32)? }
                #[cfg(not(feature = "bf16_u16"))]
                { clamp_symmetric(&h1_pre, 1.0e3f32)? }
            };
            // GELU with dtype-preserving path; use BF16 helper when enabled
            #[cfg(feature = "bf16_u16")]
            let a1 = { use flame_core::bf16_ops::gelu_bf16; gelu_bf16(&h1)? };
            #[cfg(not(feature = "bf16_u16"))]
            let a1 = h1.gelu()?;
            check(&format!("blk{}.gelu", self.idx), &a1)?;
            a1.debug_check("mlp.gelu")?;
            let out0_pre = linear_bt(&a1, &f2)?; check(&format!("blk{}.out0", self.idx), &out0_pre)?; // [B,T,3072]
            let out0 = {
                #[cfg(feature = "bf16_u16")]
                { use flame_core::bf16_clamp::clamp_bf16; clamp_bf16(&out0_pre, -1.0e3f32, 1.0e3f32)? }
                #[cfg(not(feature = "bf16_u16"))]
                { clamp_symmetric(&out0_pre, 1.0e3f32)? }
            };
            let out_pre  = add_broadcast(&out0, &lora_delta_bt(&a1, &l[5])?)?;
            let out = {
                #[cfg(feature = "bf16_u16")]
                { use flame_core::bf16_clamp::clamp_bf16; clamp_bf16(&out_pre, -1.0e3f32, 1.0e3f32)? }
                #[cfg(not(feature = "bf16_u16"))]
                { clamp_symmetric(&out_pre, 1.0e3f32)? }
            }; check(&format!("blk{}.out", self.idx), &out)?;
            out.debug_check("mlp.out")?;
            if self.tripwires { guard(&format!("blk{}.mlp_out", self.idx), &out)?; }
            Ok(out)
        } else {
            // 2D-safe approximation path (input shape [*,3072])
            let xn = normalize_last_dim(&x)?;
            let q = add_broadcast(&xn.matmul(&wq)?, &l[0].forward_delta(&xn)?)?; check(&format!("blk{}.2d.q", self.idx), &q)?;
            let k = add_broadcast(&xn.matmul(&wk)?, &l[1].forward_delta(&xn)?)?; check(&format!("blk{}.2d.k", self.idx), &k)?;
            let v = add_broadcast(&xn.matmul(&wv)?, &l[2].forward_delta(&xn)?)?; check(&format!("blk{}.2d.v", self.idx), &v)?;
            assert_finite("Q2D", &q)?; assert_finite("K2D", &k)?;
            // Normalize per-token vectors to tame magnitude before elementwise product
            let qf = q.clone();
            let kf = k.clone();
            let d_last = qf.shape().dims()[1] as f32;
            let eps = Tensor::from_scalar(1e-6f32, qf.device().clone())?;
            let q_rms = qf.square()?.sum_dim_keepdim(1)?.div_scalar(d_last)?.sqrt()?.maximum(&eps)?; // [*,1]
            let k_rms = kf.square()?.sum_dim_keepdim(1)?.div_scalar(d_last)?.sqrt()?.maximum(&eps)?; // [*,1]
            let qn = qf.div(&q_rms)?;
            let kn = kf.div(&k_rms)?;
            let mut score = mul_tensors(&qn, &kn)?;
            // Clamp before softmax along feature dim=1 for 2D path
            score = clamp_symmetric(&score, 30.0f32)?;
            let att   = softmax_stable(&score, 1)?; check(&format!("blk{}.2d.att", self.idx), &att)?;
            att.debug_check("attn.probs")?;
            if self.tripwires { guard(&format!("blk{}.attn.probs", self.idx), &att)?; }
            let ctx_pre   = mul_tensors(&att, &v)?;
            let ctx = clamp_symmetric(&ctx_pre, 1.0e3f32)?; check(&format!("blk{}.2d.ctx", self.idx), &ctx)?;
            ctx.debug_check("attn.ctx")?;
            let attn_out_pre = add_broadcast(&ctx.matmul(&wo)?, &l[3].forward_delta(&ctx)?)?;
            let attn_out = clamp_symmetric(&attn_out_pre, 1.0e3f32)?; check(&format!("blk{}.2d.attn_out", self.idx), &attn_out)?;
            attn_out.debug_check("attn.out")?;
            if self.tripwires { guard(&format!("blk{}.attn_out", self.idx), &attn_out)?; }
            // MLP: BF16 storage for intermediates, FP32 math inside kernels per mixed-precision policy
            let f1_raw = {
                let mm = if attn_out.dtype() == DType::BF16 && f1.dtype() == DType::BF16 { attn_out.matmul_bf16(&f1)? } else { attn_out.matmul(&f1)? };
                add_broadcast(&mm, &l[4].forward_delta(&attn_out)?)?
            }; check(&format!("blk{}.2d.f1_pre", self.idx), &f1_raw)?;
            let f1_pre = {
                #[cfg(feature = "bf16_u16")]
                { use flame_core::bf16_clamp::clamp_bf16; clamp_bf16(&f1_raw, -1.0e3f32, 1.0e3f32)? }
                #[cfg(not(feature = "bf16_u16"))]
                { clamp_symmetric(&f1_raw, 1.0e3f32)? }
            };
            #[cfg(feature = "bf16_u16")]
            let act32 = { use flame_core::bf16_ops::gelu_bf16; gelu_bf16(&f1_pre)? };
            #[cfg(not(feature = "bf16_u16"))]
            let act32 = f1_pre.gelu()?;
            act32.debug_check("mlp.gelu")?;
            let mlp_out_pre = {
                let mm = if act32.dtype() == DType::BF16 && f2.dtype() == DType::BF16 { act32.matmul_bf16(&f2)? } else { act32.matmul(&f2)? };
                add_broadcast(&mm, &l[5].forward_delta(&act32)?)?
            };
            let mlp_out = {
                #[cfg(feature = "bf16_u16")]
                { use flame_core::bf16_clamp::clamp_bf16; clamp_bf16(&mlp_out_pre, -1.0e3f32, 1.0e3f32)? }
                #[cfg(not(feature = "bf16_u16"))]
                { clamp_symmetric(&mlp_out_pre, 1.0e3f32)? }
            }; check(&format!("blk{}.2d.mlp_out", self.idx), &mlp_out)?;
            mlp_out.debug_check("mlp.out")?;
            if self.tripwires { guard(&format!("blk{}.mlp_out", self.idx), &mlp_out)?; }
            Ok(mlp_out)
        }
    }
}

pub struct LayerRegistry { pub blocks: Vec<Box<dyn Block>> }
pub type ChromaRegistry = LayerRegistry;
impl LayerRegistry {
    pub fn new() -> Self {
        // One combined block per keymap block to align streaming: load(i) → apply(i) using all tensors
        let mut blocks: Vec<Box<dyn Block>> = Vec::with_capacity(<ChromaKeyMap as crate::streaming::KeyMap>::block_count());
        let mode = std::env::var("CHROMA_ATTENTION_MODE").unwrap_or_else(|_| "btd".to_string());
        let btd = mode.to_lowercase() == "btd";
        let tripwires = std::env::var("TRIPWIRES").ok().map(|v| v != "0").unwrap_or(false);
        for i in 0.. <ChromaKeyMap as crate::streaming::KeyMap>::block_count() {
            blocks.push(Box::new(CombinedBlock { btd, idx: i, tripwires, route_out: false, route_in: false }));
        }
        Self { blocks }
    }
    pub fn forward_ids(&self) -> impl Iterator<Item=usize> { 0..self.blocks.len() }
}
