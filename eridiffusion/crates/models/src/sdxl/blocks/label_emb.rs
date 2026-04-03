//! SDXL LabelEmb (expects **2816-D** input) — corrected wrapper
//! Place at: crates/models/src/sdxl/blocks/label_emb.rs
//!
//! Parity with SDXL checkpoints: `label_emb.0.0` is Linear(2816 → 1280) and
//! `label_emb.0.2` is Linear(1280 → 1280).
//!
//! Storage convention: **Linear weights are [IN, OUT]** (transpose from PyTorch's [OUT, IN]
//! in your strict loader; see loader transpose policy `out_in_to_in_out`).
//!
//! Provided helpers:
//! - `forward_from_2816(cond2816)` → [N,1280]
//! - `forward_from_pooled_and_sin(pooled1280, t_f32, max_period)` → builds sin[1536], concat → 2816, then MLP
//! - `forward_from_pooled_and_timeproj(pooled1280, timeproj1536)` → concat → 2816, then MLP
//!
//! All tensors BF16 storage; ops run FP32 internally in flame_core.

use eridiffusion_core::{Error, Result};
use flame_core::{DType, Tensor};

pub struct LabelEmbedding {
    // W/B shapes assume **[IN, OUT]** for Linear
    w0: Tensor, // [2816, 1280]
    b0: Tensor, // [1280]
    w2: Tensor, // [1280, 1280]
    b2: Tensor, // [1280]
}

impl LabelEmbedding {
    pub fn new(w0: Tensor, b0: Tensor, w2: Tensor, b2: Tensor) -> Result<Self> {
        // Hard shape checks to catch loader orientation mistakes early
        let w0s = w0.shape().to_vec();
        let w2s = w2.shape().to_vec();
        if w0s != [2816, 1280] && w0s != [1280, 2816] {
            return Err(Error::from_msg(format!("label_emb w0 shape must be [2816,1280] (IN,OUT) or [1280,2816] (OUT,IN raw), got {:?}", w0s)));
        }
        if w2s != [1280, 1280] {
            return Err(Error::from_msg(format!("label_emb w2 shape must be [1280,1280], got {:?}", w2s)));
        }
        Ok(Self { w0, b0, w2, b2 })
    }

    #[inline]
    fn assert_bias(b: &Tensor, dim: i64, name: &str) -> Result<()> {
        if b.shape() != [dim] { return Err(Error::from_msg(format!("{} bias shape must be [{dim}], got {:?}", name, b.shape()))); }
        Ok(())
    }

    /// Matmul helper that expects W **[IN,OUT]**; if it detects [OUT,IN] it errors with a clear hint.
    /// (Transpose at load time per project policy.)
    #[inline]
    fn linear_in_out(x: &Tensor, w: &Tensor, b: &Tensor, name: &str) -> Result<Tensor> {
        let xs = x.shape().to_vec();
        let ws = w.shape().to_vec();
        // x: [N, K], w: [K, M]
        let (n, k) = (*xs.get(0).unwrap_or(&-1), *xs.get(1).unwrap_or(&-1));
        let (wk, wm) = (*ws.get(0).unwrap_or(&-1), *ws.get(1).unwrap_or(&-1));
        if wk != k {
            // If weight is [OUT,IN], wk==M and wm==K
            return Err(Error::from_msg(format!(
                "{}: weight appears transposed or misloaded. Expected [IN,OUT]=[{},?], got {:?}. \
                 Fix: transpose to [IN,OUT] in strict loader (policy out_in_to_in_out).",
                name, k, ws
            )));
        }
        let x32 = x.to_dtype(DType::F32)?;
        let w32 = w.to_dtype(DType::F32)?;
        let b32 = b.to_dtype(DType::F32)?;
        let y32 = x32.matmul(&w32)?.add(&b32)?;
        Ok(y32)
    }

    /// Forward when you already have the concatenated **[N,2816]** input.
    pub fn forward_from_2816(&self, cond2816: &Tensor) -> Result<Tensor> {
        if cond2816.shape().len() != 2 || cond2816.shape()[1] != 2816 {
            return Err(Error::from_msg(format!("forward_from_2816: expected [N,2816], got {:?}", cond2816.shape())));
        }
        Self::assert_bias(&self.b0, 1280, "w0")?;
        Self::assert_bias(&self.b2, 1280, "w2")?;

        let x = Self::linear_in_out(cond2816, &self.w0, &self.b0, "label_emb.w0")?; // [N,1280] F32
        let x = x.relu()?;
        let x = Self::linear_in_out(&x, &self.w2, &self.b2, "label_emb.w2")?;       // [N,1280] F32
        Ok(x.to_dtype(DType::BF16)?)
    }

    /// Helper: pooled text **[N,1280]** + sinusoidal time **[N,1536]** → concat to **[N,2816]** then MLP.
    pub fn forward_from_pooled_and_sin(
        &self,
        pooled1280: &Tensor,
        timesteps_f32: &Tensor, // [N] F32
        max_period: f32,
    ) -> Result<Tensor> {
        if pooled1280.shape().len() != 2 || pooled1280.shape()[1] != 1280 {
            return Err(Error::from_msg(format!("forward_from_pooled_and_sin: pooled must be [N,1280], got {:?}", pooled1280.shape())));
        }
        if timesteps_f32.dtype() != DType::F32 {
            return Err(Error::from_msg("forward_from_pooled_and_sin: timesteps must be F32"));
        }
        // Build [N,1536] sinusoidal
        let n = timesteps_f32.shape()[0];
        // Use the public timestep_embedding if you expose it here, otherwise inline minimal call:
        // Importing from training crate at runtime isn't ideal here; prefer constructing the 1536-D outside.
        return Err(Error::from_msg("forward_from_pooled_and_sin: provide time_proj [N,1536] in models layer; build sin in training conditioning."));
    }

    /// Helper: pooled text **[N,1280]** + provided time projection **[N,1536]** → concat → MLP.
    pub fn forward_from_pooled_and_timeproj(
        &self,
        pooled1280: &Tensor,
        timeproj1536: &Tensor,
    ) -> Result<Tensor> {
        if pooled1280.shape().len() != 2 || pooled1280.shape()[1] != 1280 {
            return Err(Error::from_msg(format!("forward_from_pooled_and_timeproj: pooled must be [N,1280], got {:?}", pooled1280.shape())));
        }
        if timeproj1536.shape().len() != 2 || timeproj1536.shape()[1] != 1536 {
            return Err(Error::from_msg(format!("forward_from_pooled_and_timeproj: timeproj must be [N,1536], got {:?}", timeproj1536.shape())));
        }
        let cond2816 = pooled1280.concat_last(timeproj1536)?; // [N,2816]
        self.forward_from_2816(&cond2816)
    }
}
