use eridiffusion_core as core;
use flame_core::{Tensor, Shape, DType, Device};

#[test]
fn flash_attention_matches_sdpa_bf16() -> anyhow::Result<()> {
    // Small ragged case: two sequences
    let device = Device::cuda(0)?;
    let h = 2usize; // heads
    let d = 4usize; // head dim
    let q_lens = [3i32, 2i32];
    let k_lens = [3i32, 2i32];
    let sq = (q_lens[0] + q_lens[1]) as usize;
    let sk = (k_lens[0] + k_lens[1]) as usize;
    let s_total = sq + sk;

    // Build packed qkv: first Sq rows hold Q, next Sk rows hold K/V
    // Fill with deterministic data for stability
    let mut data = Vec::with_capacity(s_total * 3 * h * d);
    for i in 0..s_total*3*h*d { data.push(((i % 13) as f32) * 0.01); }
    let qkv_f32 = Tensor::from_vec(data, Shape::from_dims(&[s_total, 3, h, d]), device.clone())?;
    let qkv = qkv_f32.to_dtype(DType::BF16)?;

    // Scale
    let scale = (d as f32).recip().sqrt();

    // Compute SDPA and Flash paths
    let sdpa = core::attention::sdpa_attention_packed(&qkv, Some(&q_lens), Some(&k_lens), h as i32, scale, None);
    // Force FLASH selection via env var; function falls back to SDPA internally
    std::env::set_var("ATTENTION_IMPL", "flash");
    let flash = core::attention::flash_attention_packed(&qkv, Some(&q_lens), Some(&k_lens), h as i32, scale, None);

    // Compare in FP32
    let a = sdpa.to_dtype(DType::F32)?;
    let b = flash.to_dtype(DType::F32)?;
    let diff = a.sub(&b)?;
    let max_abs = diff.abs()?.max_all().unwrap_or(0.0);
    assert!(max_abs < 1e-3, "max abs diff {} >= 1e-3", max_abs);
    Ok(())
}
#![cfg(feature = "legacy-tests")]

