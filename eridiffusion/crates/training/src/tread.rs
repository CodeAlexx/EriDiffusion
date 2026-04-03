use crate::chroma::lora::LoRALinear;
use anyhow::Result;
use flame_core::{DType, Shape, Tensor};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub enum MaskType {
    TopKNorm,
    AttnTopK,
    Random,
    Uniform,
}

#[derive(Debug, Clone, Copy)]
pub enum ReinjectMode {
    Residual,
    Concat,
}

#[derive(Debug, Clone)]
pub struct TreadCfg {
    pub enabled: bool,
    pub mask_type: MaskType,
    pub k: Option<usize>,
    pub k_frac: Option<f32>,
    pub schedule: Vec<(usize, usize)>, // (out_layer, in_layer)
    pub reinject_mode: ReinjectMode,
    pub lambda: f32,
}

impl Default for TreadCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            mask_type: MaskType::TopKNorm,
            k: None,
            k_frac: Some(0.25),
            schedule: vec![],
            reinject_mode: ReinjectMode::Residual,
            lambda: 0.0,
        }
    }
}

/// Per-step routing state stored on device (BF16 shelves + metadata)
pub struct TreadState {
    pub shelves: BTreeMap<usize, Tensor>,
    pub kept_tokens_sum: usize,
    pub kept_tokens_cnt: usize,
    pub kept_frac_per_layer: BTreeMap<usize, f32>,
    pub route_miss: u32,
    pub route_lambda: f32,
}

impl TreadState {
    pub fn new() -> Self {
        Self {
            shelves: BTreeMap::new(),
            kept_tokens_sum: 0,
            kept_tokens_cnt: 0,
            kept_frac_per_layer: BTreeMap::new(),
            route_miss: 0,
            route_lambda: 0.0,
        }
    }
    pub fn log_kept(&mut self, kept: usize) {
        self.kept_tokens_sum += kept;
        self.kept_tokens_cnt += 1;
    }
}

/// Runtime routing context built from config/env for quick checks in hot paths
#[derive(Debug, Clone)]
pub struct TreadCtx {
    pub enabled: bool,
    pub mask_type: MaskType,
    pub k_abs: Option<usize>,
    pub k_frac: Option<f32>,
    pub schedule: BTreeMap<usize, Vec<usize>>, // out -> [in]
    pub reinject_mode: ReinjectMode,
    pub lambda: f32,
}

impl TreadCtx {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            mask_type: MaskType::TopKNorm,
            k_abs: None,
            k_frac: Some(0.25),
            schedule: BTreeMap::new(),
            reinject_mode: ReinjectMode::Residual,
            lambda: 0.0,
        }
    }

    /// Build from environment variables (compat shim for current trainers)
    pub fn from_env() -> Self {
        let enabled = std::env::var("TREAD_ENABLED").ok().map(|v| v == "1").unwrap_or(false);
        let k_abs = std::env::var("TREAD_K").ok().and_then(|s| s.parse::<usize>().ok());
        let k_frac = std::env::var("TREAD_K_FRAC").ok().and_then(|s| s.parse::<f32>().ok());
        let mask_type = match std::env::var("TREAD_MASK_TYPE").ok().as_deref() {
            Some("attn_topk") => MaskType::AttnTopK,
            Some("random") => MaskType::Random,
            Some("uniform") => MaskType::Uniform,
            _ => MaskType::TopKNorm,
        };
        let reinject_mode = match std::env::var("TREAD_REINJECT_MODE").ok().as_deref() {
            Some("concat") => ReinjectMode::Concat,
            _ => ReinjectMode::Residual,
        };
        let lambda =
            std::env::var("TREAD_LAMBDA").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
        let mut schedule: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        if let Ok(env) = std::env::var("TREAD_SCHEDULE") {
            for pair in env.split(',') {
                if let Some((a, b)) = pair.split_once(':') {
                    if let (Ok(oi), Ok(ii)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>())
                    {
                        schedule.entry(oi).or_default().push(ii);
                    }
                }
            }
        }
        Self { enabled, mask_type, k_abs, k_frac, schedule, reinject_mode, lambda }
    }
}

/// Resolve k given absolute or fractional request; clamps to [1, T]
pub fn resolve_k(k_abs: Option<usize>, k_frac: Option<f32>, t: usize) -> usize {
    if let Some(k) = k_abs {
        return k.clamp(1, t.max(1));
    }
    let frac = k_frac.unwrap_or(0.25).clamp(0.0, 1.0);
    let k = ((t as f32) * frac).round() as usize;
    k.clamp(1, t.max(1))
}

/// Build boolean mask over tokens [B,T] (u8 Bool). Simple deterministic implementation.
pub fn build_mask(x: &Tensor, mask_type: MaskType, k: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, t, _d) = (dims[0], dims[1], dims[2]);
    let dev = x.device().clone();
    match mask_type {
        MaskType::Random | MaskType::Uniform => {
            // Fallback deterministic evenly spaced selection (no RNG in hot path)
            let stride = ((t as f32) / (k as f32)).ceil() as usize;
            let mut out: Vec<f32> = Vec::with_capacity(b * t);
            for _ in 0..b {
                let mut count = 0usize;
                for j in 0..t {
                    if count < k && j % stride == 0 {
                        out.push(1.0);
                        count += 1;
                    } else {
                        out.push(0.0);
                    }
                }
            }
            Ok(Tensor::from_vec(out, Shape::from_dims(&[b, t]), dev)?)
        }
        MaskType::TopKNorm | MaskType::AttnTopK => {
            // Compute per-token score [B,T]; for AttnTopK, approximate with L2 norm unless attn is supplied upstream (future API)
            let x32 = x.to_dtype(DType::F32)?;
            let sq = x32.square()?;
            let scores = sq.sum_dim_keepdim(2)?.squeeze(None)?; // [B,T]
            let sc = scores.to_dtype(DType::F32)?.to_vec()?; // host thresholding per batch
            let mut mask_host: Vec<f32> = Vec::with_capacity(b * t);
            for bi in 0..b {
                let row = &sc[bi * t..(bi + 1) * t];
                let mut idxs: Vec<usize> = (0..t).collect();
                idxs.sort_unstable_by(|&i, &j| {
                    row[j].partial_cmp(&row[i]).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut row_mask = vec![0.0f32; t];
                for &ix in idxs.iter().take(k) {
                    row_mask[ix] = 1.0f32;
                }
                mask_host.extend(row_mask);
            }
            Ok(Tensor::from_vec(mask_host, Shape::from_dims(&[b, t]), dev)?)
        }
    }
}

/// Gather routed tokens into a compact BF16 shelf [B,R,D].
pub fn gather_bf16(x: &Tensor, mask_bt: &Tensor) -> Result<Tensor> {
    // Minimal implementation: take first R=k tokens along T (consistent with build_mask) and return view
    let dims = x.shape().dims().to_vec();
    let (_b, _t, _d) = (dims[0], dims[1], dims[2]);
    // Infer R from mask sum on first row (host-safe scalar)
    let r = {
        let m0 = mask_bt.narrow(0, 0, 1)?; // [1,T]
        let sum = m0.to_dtype(DType::F32)?.sum()?.to_scalar::<f32>()?;
        sum as usize
    };
    let shelf = x.narrow(1, 0, r)?; // [B,R,D]
    Ok(shelf.to_dtype(DType::BF16)?)
}

/// Reinjection: Residual add or Concat(+proj) of shelf `r` into `x` at layer `in_layer`.
pub fn reinject(
    x: &Tensor,
    r: &Tensor,
    mode: ReinjectMode,
    proj_w: Option<&Tensor>,
) -> Result<Tensor> {
    match mode {
        ReinjectMode::Residual => {
            // Add back BF16 shelf after padding to match T if needed (pad/truncate to T along dim=1)
            let dims = x.shape().dims().to_vec();
            let (b, t, d) = (dims[0], dims[1], dims[2]);
            let r_dims = r.shape().dims().to_vec();
            let rr = if r_dims[1] == t {
                r.clone()
            } else if r_dims[1] > t {
                r.narrow(1, 0, t)?
            } else {
                // pad along T with zeros
                let need = t - r_dims[1];
                let pad = Tensor::zeros_dtype(
                    Shape::from_dims(&[b, need, d]),
                    x.dtype(),
                    x.device().clone(),
                )?;
                Tensor::cat(&[r, &pad], 1)?
            };
            Ok(x.add(&rr)?)
        }
        ReinjectMode::Concat => {
            let concat = Tensor::cat(&[x, r], 1)?; // concat along T
            if let Some(w) = proj_w {
                // [B,T2,D] x [D,Do] → [B,T2,Do]
                let dims = concat.shape().dims().to_vec();
                let (b, tt, dd) = (dims[0], dims[1], dims[2]);
                let flat = concat.reshape(&[b * tt, dd])?;
                let out = if flat.dtype() == DType::BF16 && w.dtype() == DType::BF16 {
                    flat.matmul_bf16(w)?
                } else {
                    flat.matmul(w)?
                };
                Ok(out.reshape(&[b, tt, out.shape().dims()[1]])?)
            } else {
                Ok(concat)
            }
        }
    }
}

/// Auxiliary route loss: MSE(pred_routed, target) → FP32 scalar
pub fn route_loss(pred_main: &Tensor, pred_routed: &Tensor) -> Result<Tensor> {
    let diff = pred_main.sub(pred_routed)?;
    let mse = diff.square()?.mean()?;
    Ok(mse.to_dtype(DType::F32)?)
}

/// LoRA delta helper for [B,T,D] inputs using existing forward_delta
pub fn apply_lora_delta_bt(x_bt: &Tensor, l: &LoRALinear) -> Result<Tensor> {
    let dims = x_bt.shape().dims();
    debug_assert!(dims.len() == 3, "expected [B,T,D_in]");
    let (b, t, d_in) = (dims[0], dims[1], dims[2]);
    let flat = x_bt.reshape(&[b * t, d_in])?;
    let delta = l.forward_delta(&flat)?; // [B*T, D_out]
    let d_out = delta.shape().dims()[1];
    Ok(delta.reshape(&[b, t, d_out])?)
}

/// Route-out at this layer if scheduled: builds mask (optionally attention-aware), stores shelf and stats.
pub fn route_out_if_scheduled(
    x: &Tensor,
    attn_opt: Option<&Tensor>,
    layer_idx: usize,
    ctx: &TreadCtx,
    state: &mut TreadState,
) -> Result<()> {
    if !ctx.enabled {
        return Ok(());
    }
    if !ctx.schedule.contains_key(&layer_idx) {
        return Ok(());
    }
    let t = x.shape().dims()[1];
    let k = resolve_k(ctx.k_abs, ctx.k_frac, t);
    // build mask; AttnTopK uses attention mass if provided
    let mask = if matches!(ctx.mask_type, MaskType::AttnTopK) {
        if let Some(attn) = attn_opt {
            // scores: sum over last dim: [B,T,T] -> [B,T]
            let sc = attn
                .to_dtype(DType::F32)?
                .sum_dim_keepdim(attn.shape().dims().len() - 1)?
                .squeeze(None)?;
            // host select exact top-k per batch
            let b = sc.shape().dims()[0];
            let tlen = sc.shape().dims()[1];
            let host = sc.to_vec()?;
            let mut mask_host: Vec<f32> = Vec::with_capacity(b * tlen);
            for bi in 0..b {
                let row = &host[bi * tlen..(bi + 1) * tlen];
                let mut idxs: Vec<usize> = (0..tlen).collect();
                idxs.sort_unstable_by(|&i, &j| {
                    row[j].partial_cmp(&row[i]).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut row_mask = vec![0.0f32; tlen];
                for &ix in idxs.iter().take(k) {
                    row_mask[ix] = 1.0f32;
                }
                mask_host.extend(row_mask);
            }
            Tensor::from_vec(mask_host, Shape::from_dims(&[b, tlen]), x.device().clone())?
        } else {
            build_mask(x, MaskType::TopKNorm, k)?
        }
    } else {
        build_mask(x, ctx.mask_type, k)?
    };
    let r = gather_bf16(x, &mask)?; // [B,R,D]
    let kept = k * x.shape().dims()[0];
    state.log_kept(kept);
    // record kept fraction for this layer
    let t = x.shape().dims()[1] as f32;
    if t > 0.0 {
        state.kept_frac_per_layer.insert(layer_idx, (k as f32) / t);
    }
    // echo current lambda for metrics
    state.route_lambda = ctx.lambda;
    state.shelves.insert(layer_idx, r);
    Ok(())
}

/// Reinjection at this layer if any shelves scheduled for here. Concats shelves and reinjects.
pub fn reinject_if_scheduled(
    x: &Tensor,
    layer_idx: usize,
    ctx: &TreadCtx,
    state: &mut TreadState,
) -> Result<Tensor> {
    if !ctx.enabled {
        return Ok(x.clone());
    }
    // Collect all shelves whose out layer maps to current layer_idx
    let mut cats: Vec<Tensor> = Vec::new();
    let mut expected: usize = 0;
    let mut found: usize = 0;
    for (out_idx, tgt_list) in ctx.schedule.iter() {
        if tgt_list.iter().any(|&tgt| tgt == layer_idx) {
            expected += 1;
            if let Some(sh) = state.shelves.get(out_idx) {
                cats.push(sh.clone());
                found += 1;
            }
        }
    }
    if expected > found {
        state.route_miss = state.route_miss.saturating_add((expected - found) as u32);
    }
    if cats.is_empty() {
        return Ok(x.clone());
    }
    let r_all = if cats.len() == 1 {
        cats.remove(0)
    } else {
        Tensor::cat(&cats.iter().collect::<Vec<_>>(), 1)?
    };
    reinject(x, &r_all, ctx.reinject_mode, None)
}

#[derive(Clone, Default)]
pub struct TreadMetrics {
    pub kept_avg: f32,
    pub kept_frac: Vec<f32>, // per-layer kept fraction in [0,1]
    pub shelves_mb: f32,
    pub route_miss: u32,
    pub route_lambda: f32,
}

impl TreadState {
    pub fn metrics(&self) -> TreadMetrics {
        // average kept fraction across recorded layers for this step
        let kept_avg = if self.kept_frac_per_layer.is_empty() {
            0.0
        } else {
            let s: f32 = self.kept_frac_per_layer.values().copied().sum();
            s / (self.kept_frac_per_layer.len() as f32)
        };
        // ordered per-layer fractions
        let kept_frac: Vec<f32> = self.kept_frac_per_layer.iter().map(|(_k, v)| *v).collect();
        // sum shelves bytes
        let mut bytes: usize = 0;
        for (_k, t) in self.shelves.iter() {
            let ne = t.shape().elem_count();
            let bpe = match t.dtype() {
                DType::BF16 | DType::F16 => 2usize,
                _ => 4usize,
            };
            bytes = bytes.saturating_add(ne * bpe);
        }
        let shelves_mb = (bytes as f32) / (1024.0 * 1024.0);
        TreadMetrics {
            kept_avg,
            kept_frac,
            shelves_mb,
            route_miss: self.route_miss,
            route_lambda: self.route_lambda,
        }
    }
}
