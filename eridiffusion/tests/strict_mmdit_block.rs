use std::sync::{Arc, Once};

use eridiffusion::models::mmdit_blocks::{JointTransformerBlock, QkNormKind};
use flame_core::{
    autograd::AutogradContext,
    config,
    device::Device,
    strict::{self, GuardMode},
    CudaDevice, DType, Result as FlameResult, Shape, Tensor,
};

fn enable_strict_mode() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        std::env::set_var("STRICT_BF16", "1");
        std::env::set_var("STRICT_BF16_MODE", "panic");
        strict::force_runtime_enforced();
    });
}

fn bf16_rand(shape: &[usize], device: &Arc<CudaDevice>) -> FlameResult<Tensor> {
    Tensor::randn(Shape::from_dims(shape), 0.0, 0.02, device.clone())?
        .to_dtype(DType::BF16)
        .map(|t| t.requires_grad_(false))
}

#[test]
fn strict_mmdit_block_respects_arena() -> FlameResult<()> {
    enable_strict_mode();
    let Ok(device) = Device::cuda(0) else {
        return Ok(());
    };
    let cuda = device.cuda_device().clone();

    config::set_default_dtype(DType::BF16);
    strict::reset_counters();

    let hidden = 64usize;
    let heads = 8usize;
    let mlp_ratio = 4.0f32;
    let cond_dim = 32usize;

    let block = JointTransformerBlock::new(
        hidden,
        heads,
        mlp_ratio,
        false,
        QkNormKind::Layer,
        cond_dim,
        &device,
    )?;

    let batch = 2usize;
    let n_img = 16usize;
    let n_txt = 8usize;

    let _no_grad = AutogradContext::no_grad();

    let x = bf16_rand(&[batch, n_img, hidden], &cuda)?;
    let context = bf16_rand(&[batch, n_txt, hidden], &cuda)?;
    let cond = bf16_rand(&[batch, cond_dim], &cuda)?;

    let result = strict::scope("strict.mmdit_block", GuardMode::Panic, || {
        let _no_grad_scope = AutogradContext::no_grad();
        let (x_out, context_out) = block.forward(&x, &context, &cond)?;
        assert_eq!(x_out.dtype(), DType::BF16);
        assert_eq!(context_out.dtype(), DType::BF16);
        Ok(())
    });

    let telemetry = strict::telemetry_snapshot();
    if let Err(err) = result {
        strict::reset_counters();
        return Err(err);
    }

    assert!(telemetry.strict_bf16, "STRICT_BF16 runtime must be enabled for strict harness tests");
    assert_eq!(telemetry.clone_allocs, 0, "unexpected clone allocations under STRICT_BF16");
    assert_eq!(telemetry.f32_graph_casts, 0, "unexpected f32 graph casts under STRICT_BF16");
    assert_eq!(telemetry.layout_fixes, 0, "unexpected layout fixes under STRICT_BF16");

    strict::reset_counters();
    Ok(())
}
