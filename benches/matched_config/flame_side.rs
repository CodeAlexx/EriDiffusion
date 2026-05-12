//! flame-core side of the matched-config benchmark.
//!
//! Loads `inputs.safetensors` produced by `gen_inputs.py`, runs each op
//! WARMUP iterations + ITERS timed iterations using CUDA events, writes
//! `flame_results.json`.
//!
//! Hard rules:
//!   - Same input bytes as PyTorch (loaded from the same .safetensors).
//!   - No autograd: AutogradContext::set_enabled(false) at startup.
//!   - FLAME_ALLOC_POOL is whatever the user set (we recommend =0 to match
//!     the production cold-allocator path; see README).
//!   - Forward only. We measure the kernel call; we re-time per iteration
//!     via cudaEventRecord + cudaEventSynchronize.
//!
//! Build (from `flame-core/`):
//!   cargo build --release --features cuda --bin matched_bench
//! Run:
//!   FLAME_ALLOC_POOL=0 ./target/release/matched_bench \
//!     --inputs ../EriDiffusion-v2/benches/matched_config/inputs.safetensors \
//!     --out    ../EriDiffusion-v2/benches/matched_config/flame_results.json

#![cfg(feature = "cuda")]

use flame_core::{
    autograd::AutogradContext,
    global_cuda_device,
    serialization::load_file,
    DType, Result, Tensor,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;

const WARMUP: usize = 20;
const ITERS: usize = 100;

// ── CUDA event timing (mirrors PyTorch's torch.cuda.Event) ────────────────
extern "C" {
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

struct CudaEvent(*mut c_void);
impl CudaEvent {
    fn new() -> Self {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let s = unsafe { cudaEventCreate(&mut raw) };
        assert_eq!(s, 0, "cudaEventCreate failed: {s}");
        Self(raw)
    }
    fn record(&self) {
        let s = unsafe { cudaEventRecord(self.0, std::ptr::null_mut()) };
        assert_eq!(s, 0, "cudaEventRecord failed: {s}");
    }
    fn sync(&self) {
        let s = unsafe { cudaEventSynchronize(self.0) };
        assert_eq!(s, 0, "cudaEventSynchronize failed: {s}");
    }
    fn elapsed_us(&self, start: &CudaEvent) -> f64 {
        let mut ms: f32 = 0.0;
        let s = unsafe { cudaEventElapsedTime(&mut ms, start.0, self.0) };
        assert_eq!(s, 0, "cudaEventElapsedTime failed: {s}");
        ms as f64 * 1000.0
    }
}
impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.0) };
    }
}

fn cuda_sync() {
    let s = unsafe { cudaDeviceSynchronize() };
    assert_eq!(s, 0, "cudaDeviceSynchronize failed: {s}");
}

fn percentile_sorted(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let k = ((p * (v.len() as f64 - 1.0)).floor()) as usize;
    v[k.min(v.len() - 1)]
}

fn median_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

#[derive(Debug)]
struct OpResult {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    median_us: f64,
    p90_us: f64,
}

/// Time a closure that produces a tensor. We drop the output (it can be
/// freed asynchronously) and re-sync per iteration like PyTorch does.
fn time_op<F>(label: &str, mut f: F) -> (f64, f64)
where
    F: FnMut() -> Result<Tensor>,
{
    // Warmup
    for _ in 0..WARMUP {
        let _ = f().expect(label);
    }
    cuda_sync();

    let mut times = Vec::with_capacity(ITERS);
    let start = CudaEvent::new();
    let end = CudaEvent::new();
    for _ in 0..ITERS {
        cuda_sync();
        start.record();
        let _ = f().expect(label);
        end.record();
        end.sync();
        times.push(end.elapsed_us(&start));
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = median_sorted(&times);
    let p90 = percentile_sorted(&times, 0.9);
    (med, p90)
}

fn record_op(
    results: &mut Vec<OpResult>,
    name: &str,
    shape: &[usize],
    dtype: &str,
    times: (f64, f64),
) {
    println!(
        "  {:35} {:30} median={:8.2}us  p90={:8.2}us",
        name,
        format!("{:?}", shape),
        times.0,
        times.1
    );
    results.push(OpResult {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
        median_us: times.0,
        p90_us: times.1,
    });
}

fn write_json(results: &[OpResult], path: &PathBuf) -> std::io::Result<()> {
    use std::fmt::Write;
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(s, "  \"flame_core\": \"matched_bench\",").unwrap();
    writeln!(s, "  \"iters\": {},", ITERS).unwrap();
    writeln!(s, "  \"warmup\": {},", WARMUP).unwrap();
    writeln!(s, "  \"ops\": [").unwrap();
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 == results.len() { "" } else { "," };
        write!(
            s,
            "    {{\"name\": \"{}\", \"shape\": {:?}, \"dtype\": \"{}\", \"median_us\": {:.4}, \"p90_us\": {:.4}, \"iters\": {}}}{}\n",
            r.name, r.shape, r.dtype, r.median_us, r.p90_us, ITERS, comma
        )
        .unwrap();
    }
    writeln!(s, "  ]").unwrap();
    writeln!(s, "}}").unwrap();
    std::fs::write(path, s)
}

fn parse_args() -> (PathBuf, PathBuf) {
    let mut inputs = PathBuf::from(
        "/home/alex/EriDiffusion/EriDiffusion-v2/benches/matched_config/inputs.safetensors",
    );
    let mut out = PathBuf::from(
        "/home/alex/EriDiffusion/EriDiffusion-v2/benches/matched_config/flame_results.json",
    );
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--inputs" => {
                inputs = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            _ => i += 1,
        }
    }
    (inputs, out)
}

fn main() -> Result<()> {
    // Disable autograd for the entire bench — this is forward-only kernel
    // timing. (Most ops we call don't record anyway since loaded tensors
    // are leaves without requires_grad, but be explicit.)
    AutogradContext::set_enabled(false);

    let (inputs_path, out_path) = parse_args();
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
    let t0 = std::time::Instant::now();
    let tensors: HashMap<String, Tensor> = load_file(&inputs_path, &device)?;
    cuda_sync();
    println!(
        "Loaded {} tensors in {:.1}s",
        tensors.len(),
        t0.elapsed().as_secs_f64()
    );

    // Ensure all are contiguous BF16/F32 as loaded
    let get = |k: &str| -> Tensor {
        tensors
            .get(k)
            .unwrap_or_else(|| panic!("missing tensor `{k}` in safetensors"))
            .clone()
    };

    let mut results: Vec<OpResult> = Vec::new();

    // ── Tier 1: elementwise ───────────────────────────────────────────────
    println!("\n[Tier 1] Elementwise");
    for label in ["small", "hot", "large"] {
        let x = get(&format!("x_bf16_{label}"));
        let y = get(&format!("y_bf16_{label}"));
        let shape = x.shape().dims().to_vec();
        record_op(&mut results, &format!("silu/{label}"), &shape, "bf16",
            time_op("silu", || x.silu()));
        record_op(&mut results, &format!("gelu/{label}"), &shape, "bf16",
            time_op("gelu", || x.gelu()));
        record_op(&mut results, &format!("add/{label}"),  &shape, "bf16",
            time_op("add",  || x.add(&y)));
        record_op(&mut results, &format!("mul/{label}"),  &shape, "bf16",
            time_op("mul",  || x.mul(&y)));
        record_op(&mut results, &format!("mul_scalar/{label}"), &shape, "bf16",
            time_op("mul_scalar", || x.mul_scalar(0.5)));
    }

    // ── Tier 2: reductions ────────────────────────────────────────────────
    println!("\n[Tier 2] Reductions");
    for label in ["small", "hot", "large"] {
        let x = get(&format!("x_bf16_{label}"));
        let shape = x.shape().dims().to_vec();
        record_op(&mut results, &format!("sum_full/{label}"), &shape, "bf16",
            time_op("sum_full", || x.sum()));
        let last_dim = shape.len() - 1;
        record_op(&mut results, &format!("sum_dim_last/{label}"), &shape, "bf16",
            time_op("sum_dim_last", || x.sum_dim(last_dim)));
        record_op(&mut results, &format!("mean_full/{label}"), &shape, "bf16",
            time_op("mean_full", || x.mean()));
    }

    // ── Tier 3: norms ─────────────────────────────────────────────────────
    println!("\n[Tier 3] Norms");
    for (label, last_dim) in [("zimage", 2560usize), ("klein", 4096usize)] {
        let x = get(&format!("norm_x_{label}"));
        let w = get(&format!("norm_w_{label}"));
        let b = get(&format!("norm_b_{label}"));
        let shape = x.shape().dims().to_vec();
        let ld = last_dim;
        record_op(&mut results, &format!("rms_norm/{label}"), &shape, "bf16",
            time_op("rms_norm", || {
                flame_core::norm::rms_norm(&x, &[ld], Some(&w), 1e-6)
            }));
        record_op(&mut results, &format!("layer_norm/{label}"), &shape, "bf16",
            time_op("layer_norm", || {
                flame_core::layer_norm::layer_norm(&x, &[ld], Some(&w), Some(&b), 1e-5)
            }));
    }

    // ── Tier 4: GEMM ──────────────────────────────────────────────────────
    println!("\n[Tier 4] GEMM");
    // Square 4096×4096: a @ b
    let a = get("gemm_sq_a");
    let bsq = get("gemm_sq_b");
    let shape = a.shape().dims().to_vec();
    record_op(&mut results, "matmul/sq_4096", &shape, "bf16",
        time_op("matmul_sq", || a.matmul(&bsq)));

    // Rect: x @ w.T   (PyTorch F.linear: out = x @ w.T + b, w is [out,in])
    // flame matmul takes [..., M, K] × [..., K, N] → [..., M, N]. We have
    // w as [out=2560, in=2560]; we transpose it to [in, out] then matmul.
    let x = get("gemm_rect_x");
    let w_oi = get("gemm_rect_w_oi");
    let bias = get("gemm_rect_bias");
    let shape = x.shape().dims().to_vec();
    record_op(&mut results, "matmul/rect_4096x2560", &shape, "bf16",
        time_op("matmul_rect", || {
            let wt = w_oi.transpose()?.contiguous()?;
            x.matmul(&wt)
        }));
    // Linear-with-bias via fused_linear3d_native (the actual flame call site
    // for 3D linear in inference paths).
    record_op(&mut results, "linear_bias/rect_4096x2560", &shape, "bf16",
        time_op("linear_bias", || {
            flame_core::ops::fused_inference::fused_linear3d_native(&x, &w_oi, Some(&bias))
        }));

    // ── Tier 5: SDPA ──────────────────────────────────────────────────────
    println!("\n[Tier 5] SDPA");
    let q = get("attn_q");
    let k = get("attn_k");
    let v = get("attn_v");
    let shape = q.shape().dims().to_vec();
    record_op(&mut results, "sdpa/qkv_4096_h24_d128", &shape, "bf16",
        time_op("sdpa", || flame_core::attention::sdpa(&q, &k, &v, None)));

    // ── Tier 6: layout (materialised) ─────────────────────────────────────
    println!("\n[Tier 6] Layout (materialised)");
    let l3 = get("layout_r3");      // [1, 4096, 2560]
    let l4 = get("layout_r4");      // [1, 24, 4096, 128]
    let shape = l3.shape().dims().to_vec();
    record_op(&mut results, "transpose_contig/r3", &shape, "bf16",
        time_op("transpose_contig_r3", || {
            // Transpose last-2 dims of a rank-3 tensor — flame doesn't have
            // .transpose(-1,-2) directly so we use transpose_dims(1,2)
            l3.transpose_dims(1, 2)?.contiguous()
        }));
    let shape = l4.shape().dims().to_vec();
    record_op(&mut results, "permute_reshape/r4", &shape, "bf16",
        time_op("permute_reshape_r4", || {
            // permute([2,0,1,3]) on [1,24,4096,128] → [4096,1,24,128]
            // then materialise + reshape to [4096, 24*128].
            l4.permute(&[2, 0, 1, 3])?.contiguous()?.reshape(&[4096, 24 * 128])
        }));

    // ── Tier 7: cast ──────────────────────────────────────────────────────
    println!("\n[Tier 7] Cast");
    for label in ["small", "hot", "large"] {
        let x_bf = get(&format!("x_bf16_{label}"));
        let x_f32 = get(&format!("x_f32_{label}"));
        let shape = x_bf.shape().dims().to_vec();
        record_op(&mut results, &format!("cast_bf16_to_f32/{label}"), &shape, "bf16->f32",
            time_op("cast_bf16_f32", || x_bf.to_dtype(DType::F32)));
        record_op(&mut results, &format!("cast_f32_to_bf16/{label}"), &shape, "f32->bf16",
            time_op("cast_f32_bf16", || x_f32.to_dtype(DType::BF16)));
    }

    println!("\nWriting {}...", out_path.display());
    write_json(&results, &out_path).expect("write json");
    println!("Done. {} ops.", results.len());
    Ok(())
}
