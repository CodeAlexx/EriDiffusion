//! flame-core side of the kernel-count diff bench.
//!
//! Runs N forward+backward steps on a single transformer block, designed
//! to be wrapped in `nsys profile --trace=cuda`. Mirrors `pytorch_block.py`
//! exactly: same input bytes, same forward graph, same backward call.
//!
//! Build (from `flame-core/`):
//!   cargo build --release --features cuda --bin block_count_bench
//! Run:
//!   FLAME_ALLOC_POOL=0 ./target/release/block_count_bench \
//!     --inputs /home/alex/EriDiffusion/EriDiffusion-v2/benches/kernel_count_diff/inputs.safetensors \
//!     --steps 6 --warmup 2
//!
//! Forward layout (matches block_spec.md exactly):
//!   h     = rms_norm(x, w_norm1)
//!   qkv   = linear(h, w_qkv)
//!   q,k,v = qkv.chunk(3, -1)  → permute to [B,heads,seq,dim]
//!   attn  = sdpa(q, k, v)
//!   attn  = permute back + reshape + linear(., w_o)
//!   x1    = x + attn
//!   h2    = rms_norm(x1, w_norm2)
//!   m     = linear(h2, w_mlp1).silu()
//!   m     = linear(m, w_mlp2)
//!   out   = x1 + m
//!   loss  = mse_loss(out, y)
//!
//! Backward: loss.backward()
//!
//! We do **not** apply LoRA, AdaLN, or RoPE. This is the pure foundation:
//! norms + linears + sdpa + residuals + silu + mse.

#![cfg(feature = "cuda")]

use flame_core::{
    autograd::AutogradContext,
    global_cuda_device,
    serialization::load_file,
    DType, Result, Tensor,
};
use std::collections::HashMap;
use std::path::PathBuf;

const B: usize = 1;
const SEQ: usize = 4096;
const HIDDEN: usize = 2560;
const HEADS: usize = 20;
const HEAD_DIM: usize = 128;
const EPS: f32 = 1e-6;
// MLP intermediate is 4*HIDDEN=10240; the weight shapes encode this.

extern "C" {
    fn cudaDeviceSynchronize() -> i32;
}
fn cuda_sync() {
    let s = unsafe { cudaDeviceSynchronize() };
    assert_eq!(s, 0, "cudaDeviceSynchronize failed: {s}");
}

/// `F.linear`-equivalent: out = x @ w.T (+ optional bias).
///
/// Uses `fused_linear3d_native` which records `Op::Linear` for autograd
/// and does the transpose inside cuBLASLt (TRANSA=T) — one GEMM call,
/// no separate transpose kernel. Matches the production zimage path.
fn linear3d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    let x3 = if dims.len() == 2 {
        x.reshape(&[1, dims[0], dims[1]])?
    } else {
        x.clone()
    };
    flame_core::ops::fused_inference::fused_linear3d_native(&x3, w, None)
}

fn rms_norm(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    // Single canonical path. Recorded for autograd.
    let last = w.shape().dims().last().copied().unwrap();
    flame_core::norm::rms_norm(x, &[last], Some(w), EPS)
}

fn block_forward(
    x: &Tensor,
    y: &Tensor,
    weights: &HashMap<String, Tensor>,
) -> Result<Tensor> {
    let w_norm1 = &weights["w_norm1"];
    let w_norm2 = &weights["w_norm2"];
    let w_qkv = &weights["w_qkv"];
    let w_o = &weights["w_o"];
    let w_mlp1 = &weights["w_mlp1"];
    let w_mlp2 = &weights["w_mlp2"];

    // --- attention ---
    let h = rms_norm(x, w_norm1)?;
    let qkv = linear3d(&h, w_qkv)?;            // [B, seq, 3H]
    let qkv_chunks = qkv.chunk(3, 2)?;
    let q = qkv_chunks[0].clone();
    let k = qkv_chunks[1].clone();
    let v = qkv_chunks[2].clone();

    let q = q.reshape(&[B, SEQ, HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let k = k.reshape(&[B, SEQ, HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let v = v.reshape(&[B, SEQ, HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

    let attn = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[B, SEQ, HIDDEN])?;
    let attn = linear3d(&attn, w_o)?;
    let x1 = x.add(&attn)?;

    // --- mlp ---
    let h2 = rms_norm(&x1, w_norm2)?;
    let m = linear3d(&h2, w_mlp1)?;
    let m = m.silu()?;
    let m = linear3d(&m, w_mlp2)?;
    let out = x1.add(&m)?;

    flame_core::loss::mse_loss(&out, y)
}

fn parse_args() -> (PathBuf, usize, usize) {
    let mut inputs = PathBuf::from(
        "/home/alex/EriDiffusion/EriDiffusion-v2/benches/kernel_count_diff/inputs.safetensors",
    );
    let mut steps = 6usize;
    let mut warmup = 2usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--inputs" => { inputs = PathBuf::from(&args[i + 1]); i += 2; }
            "--steps" => { steps = args[i + 1].parse().expect("--steps"); i += 2; }
            "--warmup" => { warmup = args[i + 1].parse().expect("--warmup"); i += 2; }
            _ => i += 1,
        }
    }
    (inputs, steps, warmup)
}

fn main() -> Result<()> {
    // Recording IS the target. We measure forward+backward.
    AutogradContext::set_enabled(true);

    let (inputs_path, steps, warmup) = parse_args();
    let device = global_cuda_device();

    // Warm device
    {
        let _ = Tensor::zeros_dtype(
            flame_core::Shape::from_dims(&[64, 64]),
            DType::BF16,
            device.clone(),
        )?;
        cuda_sync();
    }

    println!("Loading {}...", inputs_path.display());
    let raw: HashMap<String, Tensor> = load_file(&inputs_path, &device)?;
    cuda_sync();

    // Build a fresh weight set with requires_grad. The loaded tensors are
    // leaves with requires_grad=false; we promote them.
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for k in ["w_norm1", "w_norm2", "w_qkv", "w_o", "w_mlp1", "w_mlp2"] {
        let t = raw.get(k).unwrap_or_else(|| panic!("missing weight {k}")).clone();
        weights.insert(k.to_string(), t.requires_grad_(true));
    }
    let x = raw.get("x").unwrap().clone().requires_grad_(true);
    let y = raw.get("y").unwrap().clone(); // target, no grad

    println!("loaded; running {} steps ({} warmup)", steps, warmup);

    let mut wall_ms = Vec::new();
    for i in 0..steps {
        // Clear tape from previous step (drops the entire forward graph +
        // any dangling intermediate refs). Mirrors PT's "set grad to None".
        AutogradContext::clear();
        cuda_sync();

        let t0 = std::time::Instant::now();
        let loss = block_forward(&x, &y, &weights)?;
        let _grads = loss.backward()?;
        cuda_sync();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        wall_ms.push(dt);

        // Convert loss to F32 scalar for printing
        let loss_f32 = if loss.dtype() == DType::F32 {
            loss.clone()
        } else {
            loss.to_dtype(DType::F32)?
        };
        let v = loss_f32.to_vec()?;
        let lv = v.first().copied().unwrap_or(f32::NAN);
        println!("step {}: loss={:.4}  wall={:.2} ms", i, lv, dt);
    }

    let measured: Vec<f64> = wall_ms.iter().copied().skip(warmup).collect();
    if !measured.is_empty() {
        let mut sorted = measured.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = sorted[sorted.len() / 2];
        println!("\nmedian measured step (steps {}..{}): {:.2} ms", warmup, steps - 1, med);
    }
    println!("total measured steps: {}", measured.len());
    println!("warmup steps: {}", warmup);

    Ok(())
}
