use anyhow::Result;
use flame_core::{
    ops::{gemm_bf16::bmm_bf16_fp32acc_out, reduce::sum_dim_keepdim_as},
    DType, Shape, Tensor,
};

use super::dtype_policy::{MatmulDTypePolicy, resolve_compute_dtype};
use super::op_dtype::align_for_matmul;

macro_rules! probe {
    ($name:expr, $tensor:expr, $step:expr, $blk:expr) => {{
        Ok::<(), anyhow::Error>(())
    }};
}

fn safe_softmax(logits: &Tensor, dim: usize) -> Result<Tensor> {
    let max_vals = logits.max_dim(dim, true)?;
    let shifted = logits.sub(&max_vals)?;
    let shifted = shifted.maximum_scalar(-20.0f32)?.minimum_scalar(20.0f32)?;
    let exp = shifted.exp()?;
    let exp_bf16 = exp.to_dtype(DType::BF16)?;
    let denom_bf16 = sum_dim_keepdim_as(&exp_bf16, dim, DType::BF16)?;
    let exp = exp_bf16.to_dtype(DType::F32)?;
    let denom = denom_bf16.to_dtype(DType::F32)?;
    Ok(exp.div(&denom)?)
}

/// Real DiT block math with provided weights.
pub struct WeightPack {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub fc1: Tensor,
    pub fc2: Tensor,
}

pub struct LinearWeights {
    w_param: Tensor,
    w_fp32: Option<Tensor>,
}

impl LinearWeights {
    pub fn new(mut weight: Tensor, param_dtype: DType, precompute_fp32: bool) -> Result<Self> {
        if weight.dtype() != param_dtype {
            weight = weight.to_dtype(param_dtype)?;
        }
        let w_fp32 = if precompute_fp32 {
            Some(weight.to_dtype(DType::F32)?)
        } else {
            None
        };
        Ok(Self {
            w_param: weight,
            w_fp32,
        })
    }

    pub fn weight_for_compute(&self, compute: DType) -> Result<Tensor> {
        match compute {
            DType::BF16 => Ok(self.w_param.clone()),
            DType::F32 => {
                if let Some(w) = &self.w_fp32 {
                    Ok(w.clone())
                } else {
                    Ok(self.w_param.to_dtype(DType::F32)?)
                }
            }
            other => Err(anyhow::anyhow!("unsupported compute dtype {:?}", other)),
        }
    }

}

pub struct DiTBlock {
    pub hidden: usize,
    pub heads: usize,
    param_dtype: DType,
    matmul_policy: MatmulDTypePolicy,
    wq: LinearWeights,
    wk: LinearWeights,
    wv: LinearWeights,
    wo: LinearWeights,
    fc1: LinearWeights,
    fc2: LinearWeights,
}

impl DiTBlock {
    pub fn new(
        hidden: usize,
        heads: usize,
        weights: WeightPack,
        param_dtype: DType,
        matmul_policy: MatmulDTypePolicy,
    ) -> Result<Self> {
        let precompute_fp32 = matches!(matmul_policy, MatmulDTypePolicy::ForceFP32);
        Ok(Self {
            hidden,
            heads,
            param_dtype,
            matmul_policy,
            wq: LinearWeights::new(weights.wq, param_dtype, precompute_fp32)?,
            wk: LinearWeights::new(weights.wk, param_dtype, precompute_fp32)?,
            wv: LinearWeights::new(weights.wv, param_dtype, precompute_fp32)?,
            wo: LinearWeights::new(weights.wo, param_dtype, precompute_fp32)?,
            fc1: LinearWeights::new(weights.fc1, param_dtype, precompute_fp32)?,
            fc2: LinearWeights::new(weights.fc2, param_dtype, precompute_fp32)?,
        })
    }
}

impl DiTBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_probe(x, usize::MAX, usize::MAX)
    }

    fn to_param_dtype(&self, tensor: Tensor) -> Result<Tensor> {
        if tensor.dtype() == self.param_dtype {
            Ok(tensor)
        } else {
            Ok(tensor.to_dtype(self.param_dtype)?)
        }
    }

    fn linear_project(
        &self,
        input: &Tensor,
        weights: &LinearWeights,
        compute: DType,
        ctx: &str,
    ) -> Result<Tensor> {
        let dims = input.shape().dims().to_vec();
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let prepared = if input.dtype() != compute {
            input.to_dtype(compute)?
        } else {
            input.clone()
        };
        let flat = prepared.reshape(&[b * s, d])?;
        let weight = weights.weight_for_compute(compute)?;
        let out_dim = weight.shape().dims()[1];
        let (flat_aligned, weight_aligned) = align_for_matmul(&flat, &weight, compute)?;
        let flat_ready = flat_aligned.reshape(&[b * s, d])?;
        debug_assert_eq!(flat_ready.dtype(), weight_aligned.dtype(), "{}", ctx);
        let prod = if compute == DType::BF16 {
            flat_ready.matmul_bf16(&weight_aligned)?
        } else {
            flat_ready.matmul(&weight_aligned)?
        };
        let prod = prod.reshape(&[b, s, out_dim])?;
        Ok(prod)
    }

    pub fn forward_with_probe(&self, x: &Tensor, step: usize, blk_idx: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, s, h) = (dims[0], dims[1], dims[2]);
        assert_eq!(h, self.hidden, "hidden mismatch");
        let compute = resolve_compute_dtype(self.param_dtype, self.matmul_policy);

        // RMSNorm last dim: BF16 sums with FP32 math, BF16 storage at the boundary
        let last = h as f32;
        let x_bf16 = x.to_dtype(DType::BF16)?;
        let sum_bf16 = sum_dim_keepdim_as(&x_bf16, 2, DType::BF16)?;
        let sum_f32 = sum_bf16.to_dtype(DType::F32)?;
        let mean = sum_f32.div_scalar(last)?;
        let x32 = x.to_dtype(DType::F32)?;
        let xc = x32.sub(&mean)?;
        let xc_sq = xc.square()?;
        let xc_sq_bf16 = xc_sq.to_dtype(DType::BF16)?;
        let var_bf16 = sum_dim_keepdim_as(&xc_sq_bf16, 2, DType::BF16)?;
        let var = var_bf16.to_dtype(DType::F32)?.div_scalar(last)?;
        let inv = var.add_scalar(1e-5)?.rsqrt()?;
        let xn_f32 = xc.mul(&inv)?;
        probe!("blk0.pre_ln", &xn_f32, step, blk_idx)?;
        let xn = self.to_param_dtype(xn_f32.to_dtype(self.param_dtype)?)?;

        // QKV projections (compute dtype)
        let q = self.linear_project(&xn, &self.wq, compute, "attn.q")?;
        let k = self.linear_project(&xn, &self.wk, compute, "attn.k")?;
        let v = self.linear_project(&xn, &self.wv, compute, "attn.v")?;
        probe!("blk0.q", &q, step, blk_idx)?;
        probe!("blk0.k", &k, step, blk_idx)?;
        probe!("blk0.v", &v, step, blk_idx)?;

        // Attention logits in FP32
        let head_dim = self.hidden / self.heads;
        let scale = (head_dim as f32).sqrt().recip();
        let bh = b * self.heads;
        let q_f32 = if q.dtype() == DType::F32 { q.clone() } else { q.to_dtype(DType::F32)? };
        let k_f32 = if k.dtype() == DType::F32 { k.clone() } else { k.to_dtype(DType::F32)? };
        let v_f32 = if v.dtype() == DType::F32 { v.clone() } else { v.to_dtype(DType::F32)? };

        let reshape_q = q_f32
            .reshape(&[b, s, self.heads, head_dim])?
            .permute(&[0, 2, 1, 3])?
            .reshape(&[bh, s, head_dim])?;
        let reshape_k = k_f32
            .reshape(&[b, s, self.heads, head_dim])?
            .permute(&[0, 2, 1, 3])?
            .reshape(&[bh, s, head_dim])?;
        let reshape_v = v_f32
            .reshape(&[b, s, self.heads, head_dim])?
            .permute(&[0, 2, 1, 3])?
            .reshape(&[bh, s, head_dim])?;

        let q_bf16 = reshape_q.to_dtype(DType::BF16)?;
        let k_bf16 = reshape_k.to_dtype(DType::BF16)?;
        let k_t = k_bf16.transpose_dims(1, 2)?.clone_result()?;
        let mut logits_bf16 = Tensor::zeros_dtype(
            Shape::from_dims(&[bh, s, s]),
            DType::BF16,
            q_bf16.device().clone(),
        )?;
        bmm_bf16_fp32acc_out(&q_bf16, &k_t, &mut logits_bf16, false, false)?;
        let logits = logits_bf16.to_dtype(DType::F32)?.mul_scalar(scale)?;
        probe!("blk0.attn_logits", &logits, step, blk_idx)?;
        let probs = safe_softmax(&logits, 2)?;
        probe!("blk0.attn_probs", &probs, step, blk_idx)?;
        let probs_bf16 = probs.to_dtype(DType::BF16)?;
        let v_bf16 = reshape_v.to_dtype(DType::BF16)?;
        let mut ctx_bf16 = Tensor::zeros_dtype(
            Shape::from_dims(&[bh, s, head_dim]),
            DType::BF16,
            probs_bf16.device().clone(),
        )?;
        bmm_bf16_fp32acc_out(&probs_bf16, &v_bf16, &mut ctx_bf16, false, false)?;
        let ctx = ctx_bf16.to_dtype(DType::F32)?;
        probe!("blk0.attn_out", &ctx, step, blk_idx)?;
        let ctx = ctx
            .reshape(&[b, self.heads, s, head_dim])?
            .permute(&[0, 2, 1, 3])?
            .reshape(&[b, s, self.hidden])?;
        let ctx = if ctx.dtype() != compute {
            ctx.to_dtype(compute)?
        } else {
            ctx
        };

        let attn_out_compute = self.linear_project(&ctx, &self.wo, compute, "attn.out")?;
        let attn_out = self.to_param_dtype(attn_out_compute)?;
        let x1 = x.add(&attn_out)?;

        // MLP path
        let h1 = self.linear_project(&x1, &self.fc1, compute, "mlp.fc1")?;
        let h1 = if blk_idx == 0 && step < 8 {
            h1.maximum_scalar(-50.0f32)?.minimum_scalar(50.0f32)?
        } else {
            h1
        };
        let h1_gelu = if h1.dtype() != DType::F32 {
            h1.to_dtype(DType::F32)?.gelu()?
        } else {
            h1.gelu()?
        };
        let h1_gelu = if compute != DType::F32 {
            h1_gelu.to_dtype(compute)?
        } else {
            h1_gelu
        };
        let mlp_out_compute = self.linear_project(&h1_gelu, &self.fc2, compute, "mlp.fc2")?;
        probe!("blk0.mlp_out", &mlp_out_compute, step, blk_idx)?;
        let mlp_out = self.to_param_dtype(mlp_out_compute)?;
        let x2 = x1.add(&mlp_out)?;
        Ok(x2)
    }
}
