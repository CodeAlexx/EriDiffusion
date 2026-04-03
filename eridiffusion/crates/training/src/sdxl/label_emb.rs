//! SDXL label embedding MLP (expects 2816-d concatenated conditioning).

use anyhow::{bail, Result};
use flame_core::{DType, Tensor};

/// Wrapper around `model.diffusion_model.label_emb.0.{0,2}` weights.
/// We store Linear weights in `[IN, OUT]` layout; transpose at load time if needed.
pub struct LabelEmbedding {
    w0: Tensor, // [2816, 1280]
    b0: Tensor, // [1280]
    w2: Tensor, // [1280, 1280]
    b2: Tensor, // [1280]
}

impl LabelEmbedding {
    pub fn new(w0: Tensor, b0: Tensor, w2: Tensor, b2: Tensor) -> Result<Self> {
        let w0_dims = w0.shape().dims().to_vec();
        if w0_dims != vec![2816, 1280] {
            bail!(
                "label_emb.0.0.weight must be [2816,1280] (IN,OUT), got {:?}. Ensure the strict loader transposes PyTorch [OUT,IN] weights.",
                w0_dims
            );
        }
        let w0 = ensure_bf16(w0, "label_emb.w0.weight")?;
        let b0 = ensure_bf16(b0, "label_emb.w0.bias")?;
        let w2 = ensure_bf16(w2, "label_emb.w2.weight")?;
        let b2 = ensure_bf16(b2, "label_emb.w2.bias")?;
        Ok(Self { w0, b0, w2, b2 })
    }

    fn linear_in_out(x: &Tensor, w: &Tensor, b: &Tensor, name: &str) -> Result<Tensor> {
        let in_dim = x.shape().dims().last().copied().unwrap_or_default();
        let w_dims = w.shape().dims();
        if w_dims.len() != 2 || w_dims[0] != in_dim {
            bail!("{name}: expected weight [IN,OUT] with IN={}, got {:?}", in_dim, w_dims);
        }
        let bias = b.reshape(&[1, w_dims[1]])?;
        let y = x.matmul_bf16(w)?;
        Ok(crate::tensor_utils::broadcast_add(&y, &bias)?)
    }

    pub fn forward_from_2816(&self, cond2816: &Tensor) -> Result<Tensor> {
        let dims = cond2816.shape().dims();
        if dims.len() != 2 || dims[1] != 2816 {
            bail!("forward_from_2816: expected [N,2816], got {:?}", dims);
        }
        let cond_bf16 = if cond2816.dtype() == DType::BF16 {
            cond2816.clone_result()?
        } else {
            cond2816.to_dtype(DType::BF16)?
        };
        let mut x = Self::linear_in_out(&cond_bf16, &self.w0, &self.b0, "label_emb.w0")?;
        x = x
            .to_dtype(DType::F32)?
            .gelu()? // compute in FP32
            .to_dtype(DType::BF16)?;
        let x = Self::linear_in_out(&x, &self.w2, &self.b2, "label_emb.w2")?;
        Ok(x)
    }
}

fn ensure_bf16(t: Tensor, tag: &str) -> Result<Tensor> {
    if t.dtype() == DType::BF16 {
        Ok(t)
    } else {
        eprintln!("[label_emb] casting {tag} from {:?} to BF16", t.dtype());
        t.to_dtype(DType::BF16).map_err(Into::into)
    }
}
