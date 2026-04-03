use anyhow::Result;
use flame_core::{Device, Tensor, Shape, DType};
use eridiffusion_common_weights::ParamRegistry;
use eridiffusion_common_optim::{adamw::AdamW, grad_store::GradStore};

#[test]
fn adamw_updates_param_with_synthetic_grad() -> Result<()> {
    let dev = Device::cuda(0)?;
    let mut reg = ParamRegistry::new();

    // Param [2,2] initialized to 1.0 (BF16)
    let p0 = Tensor::full(Shape::from(vec![2,2]), 1.0_f32, dev.clone())?.to_dtype(DType::BF16)?;
    let pid = reg.insert("test.weight", p0.clone());

    // Grad = ones (FP32)
    let g = Tensor::full(Shape::from(vec![2,2]), 1.0_f32, dev.clone())?;
    let mut gs = GradStore::new();
    gs.set(pid, g);

    // Step AdamW
    let mut opt = AdamW::new(&[pid], 1e-2, (0.9, 0.999), 1e-8, 0.0);
    opt.step_with_grads(&mut reg, &gs, &[pid], 1e-2)?;

    // Diff must be > 0
    let p_new = reg.get_by_id(pid).unwrap().to_dtype(DType::F32)?;
    let p_old = p0.to_dtype(DType::F32)?;
    let diff = p_new.sub(&p_old)?.abs()?.sum_all()?.item()?;
    assert!(diff > 0.0, "expected non-zero update, got diff={}", diff);
    Ok(())
}
