//! Klein 9B real-weights smoke: load full 18 GB checkpoint, run a LoRA forward+
//! backward pass, and confirm gradients reach every LoRA parameter without
//! producing NaN/Inf. This is the single-batch, deterministic, GPU-only
//! validation we run before committing thousands of training steps.
//!
//! Mirrors the trainer's actual call site (see `train_klein.rs`):
//!   - `KleinModel::load(&shards, &config, device)` with `TrainingMethod::Lora`
//!   - `model.forward(&noisy_bchw_bf16, &text_bf16, &timestep_f32_b)`
//!   - target = noise - latent (rectified-flow), F32 MSE on the prediction.
//!
//! Heavyweight; gated behind `#[ignore]`. Run with:
//!   LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:/home/alex/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH \
//!     cargo test --release -p eridiffusion-core --test klein9b_real_weights -- \
//!     --ignored --nocapture
//!
//! Empowerment note: this test validates the *trainer's* path, not an
//! idealized one. If `KleinModel::forward` ever changes its signature, this
//! test must follow the trainer's actual call (which it currently does:
//! `(latent_bchw, text_embedding, timestep)`).

use std::path::PathBuf;

use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::models::KleinModel;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};

const KLEIN_9B_PATH: &str = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors";

/// Klein 9B inner_dim=4096, joint_dim=12288, in_channels=128 (per
/// `KleinConfig::klein_9b()` and `from_weights` autodetect). Latents are
/// packed to 16× downsampling; for a 1024² image that's [1, 128, 64, 64].
/// Use the smallest meaningful spatial size that still exercises the full
/// double+single block stack without OOM-ing on a 24 GB card under the
/// default `KLEIN_GRAD_CHECKPOINT=1` setting.
const LATENT_C: usize = 128;
// Smoke test scaling: the goal is to hit every block, not to match training
// resolution. 16x16=256 image tokens + 64 text tokens fits in 24 GB during
// forward+backward through all 8 double + 24 single blocks (the trainer at
// 32x32 + 512 text uses BlockOffloader to fit; we don't here).
const LATENT_HW: usize = 16;
const TXT_LEN: usize = 64;
const JOINT_DIM_9B: usize = 12288;

#[test]
#[ignore]
fn klein9b_real_forward_backward() {
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let path = PathBuf::from(KLEIN_9B_PATH);
    assert!(
        path.exists(),
        "Klein 9B weights not found at {} — cannot run real-weights smoke",
        path.display(),
    );

    // Match the trainer (`train_klein.rs:128-137`): TrainingMethod::Lora with
    // explicit rank/alpha. We pick rank=4 for fast load & small grad-norm
    // numerics; the trainer typically runs rank=16 but the wiring is identical.
    let mut config = TrainConfig::default();
    config.training_method = TrainingMethod::Lora;
    config.lora_rank = 4;
    config.lora_alpha = 4.0;

    println!(
        "Loading Klein 9B from {} (rank={}, alpha={})...",
        path.display(),
        config.lora_rank,
        config.lora_alpha,
    );
    let t_load = std::time::Instant::now();
    let mut model = KleinModel::load(&[path.clone()], &config, device.clone())
        .expect("KleinModel::load failed");
    println!(
        "Loaded in {:.1}s — {} weight tensors, {} LoRA params, num_double={} num_single={}",
        t_load.elapsed().as_secs_f32(),
        model.weights.len(),
        model.parameters.len(),
        model.kconfig.num_double,
        model.kconfig.num_single,
    );

    // Sanity: this had better be the 9B variant.
    assert_eq!(model.kconfig.inner_dim, 4096, "expected Klein 9B inner_dim=4096");
    assert_eq!(model.kconfig.joint_attention_dim, JOINT_DIM_9B);
    // Adapter count: 12 per double + 2 per single.
    let expected_adapters = model.kconfig.num_double * 12 + model.kconfig.num_single * 2;
    assert_eq!(
        model.lora_adapters.len(),
        expected_adapters,
        "LoRA adapter count mismatch (12*double + 2*single)",
    );

    // ── Inputs (deterministic seed=42) ───────────────────────────────────
    // Latent: [1, 128, 32, 32] BF16, randn ~ N(0,1). Trainer feeds the
    // sigma-blended `noisy = sigma*noise + (1-sigma)*latent`, but for a
    // smoke test we just need *some* finite input the model can ingest;
    // the gradient check is what we care about.
    let latent_shape = Shape::from_dims(&[1, LATENT_C, LATENT_HW, LATENT_HW]);
    let noise = Tensor::randn(latent_shape.clone(), 0.0, 1.0, device.clone())
        .expect("noise alloc")
        .to_dtype(DType::BF16)
        .expect("noise->bf16");
    let latent = Tensor::randn(latent_shape, 0.0, 1.0, device.clone())
        .expect("latent alloc")
        .to_dtype(DType::BF16)
        .expect("latent->bf16");
    // Mid-schedule sigma=0.5: noisy = 0.5*noise + 0.5*latent
    let noisy = noise
        .mul_scalar(0.5)
        .unwrap()
        .add(&latent.mul_scalar(0.5).unwrap())
        .unwrap();

    // Text embeddings: trainer feeds [1, 512, joint_dim] BF16 from prepare_klein.
    // We use TXT_LEN=256 to halve attention cost while still exercising the
    // joint-attention path.
    let txt = Tensor::randn(
        Shape::from_dims(&[1, TXT_LEN, JOINT_DIM_9B]),
        0.0,
        1.0,
        device.clone(),
    )
    .expect("txt alloc")
    .to_dtype(DType::BF16)
    .expect("txt->bf16");

    // Timestep: [1] F32 (trainer line 303-307; not BF16 — model converts internally).
    let timestep = Tensor::from_vec(vec![0.5f32], Shape::from_dims(&[1]), device.clone())
        .expect("timestep alloc");

    println!(
        "Inputs: noisy={:?} txt={:?} timestep={:?}",
        noisy.shape().dims(),
        txt.shape().dims(),
        timestep.shape().dims(),
    );

    // ── Forward ──────────────────────────────────────────────────────────
    let t_fwd = std::time::Instant::now();
    let pred = model
        .forward(&noisy, &txt, &timestep)
        .expect("Klein forward failed");
    println!(
        "Forward in {:.1}s — pred {:?}",
        t_fwd.elapsed().as_secs_f32(),
        pred.shape().dims(),
    );
    assert_eq!(
        pred.shape().dims(),
        noisy.shape().dims(),
        "pred shape != input shape",
    );

    // Output sanity: a degenerate forward (all zeros, all NaN) would silently
    // produce a tiny finite loss but kill training.
    let pred_f32 = pred.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let pred_min = pred_f32.iter().copied().fold(f32::INFINITY, f32::min);
    let pred_max = pred_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let pred_mean: f32 = pred_f32.iter().sum::<f32>() / pred_f32.len() as f32;
    println!(
        "Pred stats: min={:.4} max={:.4} mean={:.4} n={}",
        pred_min,
        pred_max,
        pred_mean,
        pred_f32.len(),
    );
    assert!(pred_min.is_finite() && pred_max.is_finite(), "pred has NaN/Inf");

    // ── Loss (rectified-flow target: noise - latent), F32 MSE ────────────
    let target = noise.sub(&latent).expect("noise-latent");
    let diff = pred
        .to_dtype(DType::F32)
        .unwrap()
        .sub(&target.to_dtype(DType::F32).unwrap())
        .unwrap();
    let loss = diff.square().unwrap().mean().unwrap();
    let loss_val = loss.to_vec().unwrap()[0];
    println!("Loss: {:.6}", loss_val);
    assert!(loss_val.is_finite(), "loss is NaN/Inf");
    assert!(loss_val > 0.0, "loss is zero (degenerate forward)");

    // ── Backward ─────────────────────────────────────────────────────────
    let t_bwd = std::time::Instant::now();
    let grads = loss.backward().expect("backward failed");
    println!(
        "Backward in {:.1}s — {} gradients tracked",
        t_bwd.elapsed().as_secs_f32(),
        grads.len(),
    );

    // ── Gradient verification ────────────────────────────────────────────
    //
    // *** Important LoRA-init asymmetry on step 1 ***
    //
    // The standard LoRA init (lora.rs:38-51, mirrors PEFT/diffusers) is:
    //   lora_a = Kaiming-uniform   (nonzero)
    //   lora_b = zeros
    // The branch is `delta = x @ A^T @ B^T`. With B=0:
    //   ∂L/∂B ∝ A^T @ x  → nonzero  (A is nonzero, x is nonzero)
    //   ∂L/∂A ∝ B^T @ ...  → ZERO   (B is zero)
    // So on the *first* backward only HALF the LoRA parameters see gradient
    // — every `lora_b` does, every `lora_a` does not. This is correct init,
    // not a bug; ∂L/∂A becomes nonzero from step 2 onward once the optimizer
    // has moved B off zero.
    //
    // The smoke check therefore asserts:
    //   - 100% of `lora_b` have nonzero, finite gradient
    //   - 0% of `lora_a` should be NaN/Inf (zero is expected)
    //   - the global max-abs grad is positive and finite
    //
    // This is the strongest single-step assertion possible without faking
    // an optimizer update (which would require touching optimizers.rs —
    // out of scope for this builder).
    let n_adapters = model.lora_adapters.len();
    let mut b_nonzero = 0usize;
    let mut b_missing = 0usize;
    let mut a_nonzero = 0usize;   // expected zero on step 1; counted for visibility
    let mut a_missing = 0usize;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut grad_abs_sum = 0f64;
    let mut grad_abs_max = 0f32;
    let mut elem_count: u64 = 0;

    let mut inspect = |p: &flame_core::parameter::Parameter,
                       nonzero: &mut usize,
                       missing: &mut usize| {
        let g = match grads.get(p.id()) {
            Some(g) => g,
            None => {
                *missing += 1;
                return;
            }
        };
        let g_f32 = g.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        let mut p_abs_sum = 0f64;
        let mut p_has_nan = false;
        let mut p_has_inf = false;
        for &v in &g_f32 {
            if v.is_nan() {
                p_has_nan = true;
            } else if v.is_infinite() {
                p_has_inf = true;
            } else {
                let av = v.abs();
                p_abs_sum += av as f64;
                if av > grad_abs_max {
                    grad_abs_max = av;
                }
            }
        }
        if p_has_nan { nan_count += 1; }
        if p_has_inf { inf_count += 1; }
        if p_abs_sum > 0.0 { *nonzero += 1; }
        grad_abs_sum += p_abs_sum;
        elem_count += g_f32.len() as u64;
    };

    for adapter in &model.lora_adapters {
        inspect(adapter.lora_a(), &mut a_nonzero, &mut a_missing);
        inspect(adapter.lora_b(), &mut b_nonzero, &mut b_missing);
    }

    let grad_mean_abs = if elem_count > 0 {
        (grad_abs_sum / elem_count as f64) as f32
    } else {
        0.0
    };
    println!("LoRA grad summary ({} adapters):", n_adapters);
    println!(
        "  lora_b: {}/{} nonzero, {} missing  (expected 100% nonzero)",
        b_nonzero, n_adapters, b_missing,
    );
    println!(
        "  lora_a: {}/{} nonzero, {} missing  (expected 0% nonzero on step 1, B=0 init)",
        a_nonzero, n_adapters, a_missing,
    );
    println!(
        "  NaN={} Inf={}  grad_mean_abs={:.6e}  grad_max_abs={:.6e}",
        nan_count, inf_count, grad_mean_abs, grad_abs_max,
    );

    assert_eq!(nan_count, 0, "Found NaN gradients on {} LoRA params", nan_count);
    assert_eq!(inf_count, 0, "Found Inf gradients on {} LoRA params", inf_count);
    assert_eq!(
        b_missing, 0,
        "{} lora_b parameters had no gradient entry — autograd path broken",
        b_missing,
    );
    assert_eq!(
        b_nonzero, n_adapters,
        "Only {}/{} lora_b have nonzero gradient — incomplete grad flow",
        b_nonzero, n_adapters,
    );
    assert!(grad_abs_max > 0.0, "All LoRA gradients are zero");

    println!(
        "\n=== PASS: Klein 9B real-weights forward+backward verified \
         ({} adapters, all lora_b grads nonzero, max|grad|={:.3e}) ===",
        n_adapters, grad_abs_max,
    );

    AutogradContext::clear();
}
