//! LoRA (Low-Rank Adaptation) adapter based on flame-diffusion's LoRALinear.
//!
//! Design:
//! - Weights stored in F32 (what AdamW expects).
//! - Forward path casts to BF16 for compute, uses autograd-aware path.
//! - Kaiming uniform init on A, zero init on B.
//! - Supports save/load with standard lora_A / lora_B key naming.
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};

use crate::Result;

#[derive(Clone)]
pub struct LoRALinear {
    pub lora_a: Parameter,
    pub lora_b: Parameter,
    pub rank: usize,
    pub alpha: f32,
    in_features: usize,
    out_features: usize,
}

impl LoRALinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        device: Arc<CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        // Kaiming uniform — matches torch nn.Linear default
        let bound = 1.0 / (in_features as f32).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        let a_data: Vec<f32> = (0..rank * in_features)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * bound)
            .collect();

        let lora_a = Parameter::new(
            Tensor::from_vec(a_data, Shape::from_dims(&[rank, in_features]), device.clone())?
                .requires_grad_(true),
        );
        let lora_b = Parameter::new(
            Tensor::zeros_dtype(Shape::from_dims(&[out_features, rank]), DType::F32, device)?
                .requires_grad_(true),
        );

        Ok(Self { lora_a, lora_b, rank, alpha, in_features, out_features })
    }

    /// Compute LoRA delta: scale * (input @ A^T @ B^T).
    ///
    /// **Currently BF16.** F32 LoRA branch (matching inference-flame /
    /// ai-toolkit) was attempted 2026-05-04 at rank 16/8/4 — all OOM ERNIE
    /// training at step 0 even with FLAME_ALLOC_POOL=0. Each `to_dtype(F32)`
    /// is autograd-retained for backward; 252 modules × multiple F32
    /// intermediates per call exceeds 24 GB regardless of rank. Real fix
    /// requires gradient checkpointing on the full model forward, not just
    /// the LoRA branch. Until then, BF16 LoRA produces soft / identity-less
    /// renders (the inference-flame "featureless mush" failure mode).
    pub fn forward_delta(&self, input: &Tensor) -> Result<Tensor> {
        let ndim = input.shape().dims().len();
        let leading: usize = input.shape().dims()[..ndim - 1].iter().product();
        let input_2d = input.reshape(&[leading, self.in_features])?;

        let a = self.lora_a.tensor()?.to_dtype(DType::BF16)?;
        let b = self.lora_b.tensor()?.to_dtype(DType::BF16)?;
        let a_t = a.transpose()?.contiguous()?;
        let b_t = b.transpose()?.contiguous()?;

        let mut delta = input_2d.matmul(&a_t)?.matmul(&b_t)?.contiguous()?;
        let scale = self.alpha / self.rank as f32;
        if (scale - 1.0).abs() > f32::EPSILON {
            delta = delta.mul_scalar(scale)?;
        }

        let mut out_shape: Vec<usize> = input.shape().dims().to_vec();
        *out_shape.last_mut().unwrap() = self.out_features;
        delta.reshape(&out_shape).map_err(Into::into)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    /// Save in diffusers / PEFT / ai-toolkit convention:
    ///   `<prefix>.lora_A.weight` and `<prefix>.lora_B.weight`
    /// — the `.weight` suffix is what the broader ecosystem expects (HF PEFT,
    /// diffusers `load_lora_weights`, ai-toolkit, etc.). Without it, every
    /// inference-side loader has to special-case "bare suffix" trainers.
    pub fn save_tensors(&self, prefix: &str, out: &mut HashMap<String, Tensor>) -> Result<()> {
        out.insert(format!("{prefix}.lora_A.weight"), self.lora_a.tensor()?);
        out.insert(format!("{prefix}.lora_B.weight"), self.lora_b.tensor()?);
        Ok(())
    }

    /// Loads either the new diffusers convention (`<prefix>.lora_A.weight`)
    /// or the legacy bare-suffix convention (`<prefix>.lora_A`) for
    /// back-compat with checkpoints from before the format change.
    pub fn load_tensors(&self, prefix: &str, source: &HashMap<String, Tensor>) -> Result<()> {
        let a_new = format!("{prefix}.lora_A.weight");
        let b_new = format!("{prefix}.lora_B.weight");
        let a_legacy = format!("{prefix}.lora_A");
        let b_legacy = format!("{prefix}.lora_B");
        let a = source.get(&a_new).or_else(|| source.get(&a_legacy))
            .ok_or_else(|| crate::EriDiffusionError::Lora(format!("missing {a_new} (or legacy {a_legacy})")))?;
        let b = source.get(&b_new).or_else(|| source.get(&b_legacy))
            .ok_or_else(|| crate::EriDiffusionError::Lora(format!("missing {b_new} (or legacy {b_legacy})")))?;
        self.lora_a.set_data(a.to_dtype(DType::F32)?.requires_grad_(true))?;
        self.lora_b.set_data(b.to_dtype(DType::F32)?.requires_grad_(true))?;
        Ok(())
    }

    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }
    pub fn rank_val(&self) -> usize { self.rank }
    /// Accessor for ports that expect a method-style API (e.g. zimage model from flame-diffusion).
    pub fn lora_a(&self) -> &Parameter { &self.lora_a }
    /// Accessor for ports that expect a method-style API.
    pub fn lora_b(&self) -> &Parameter { &self.lora_b }

    /// Clear any cached tensors — call after optimizer.step()
    pub fn refresh_cache(&self) {}
}
