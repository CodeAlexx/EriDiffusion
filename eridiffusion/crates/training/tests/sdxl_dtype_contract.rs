#![cfg(all(feature = "flux_trainer", feature = "sdxl"))]

use anyhow::Result;
use eridiffusion_training::sdxl::{
    registry::ConditioningBundle,
    runtime::{ExecutableBlock, SdxlBlockRuntime},
};
use flame_core::{
    device::Device as FlameDevice,
    CudaDevice,
    DType,
    Shape,
    Tensor,
};
use std::sync::Arc;

const DRIVER_DIM: usize = 1280;
const TIME_PROJ_DIM: usize = 1536;

#[test]
fn sdxl_block_dtype_contract_bf16_nhwc() -> Result<()> {
    let Some(flame_device) = cuda_device_or_skip()? else {
        return Ok(());
    };
    let cuda = flame_device.cuda_device_arc();
    let hidden = 1280;
    let context = 2048;
    let mut block = synthetic_block(&cuda, hidden, context)?;

    let batch = 2;
    let tokens = 77;
    let h_lat = 4;
    let w_lat = 4;
    let bundle = synthetic_bundle(&cuda, batch)?;
    let ctx = randn_bf16(&cuda, &[batch, tokens, context], 0.02)?;
    let sample = randn_bf16(&cuda, &[batch, h_lat, w_lat, hidden], 0.02)?;

    let out = block.forward_with_cond(
        &sample,
        &ctx,
        &bundle.driver_1280,
        Some(&bundle.time_proj_1536),
    )?;
    assert_eq!(out.dtype(), DType::BF16, "block output must remain BF16");
    assert_eq!(
        out.storage_dtype(),
        DType::BF16,
        "block output storage dtype must remain BF16"
    );
    assert_eq!(
        out.shape().dims(),
        &[batch, h_lat, w_lat, hidden],
        "output must stay NHWC"
    );
    assert_eq!(sample.dtype(), DType::BF16, "input sample mutated");
    assert_eq!(ctx.dtype(), DType::BF16, "context tensor mutated");

    Ok(())
}

#[test]
fn sdxl_block_requires_reuse_guard() -> Result<()> {
    let Some(flame_device) = cuda_device_or_skip()? else {
        return Ok(());
    };
    let cuda = flame_device.cuda_device_arc();
    let hidden = 1280;
    let context = 2048;
    let mut block = synthetic_block(&cuda, hidden, context)?;
    let bundle = synthetic_bundle(&cuda, 1)?;

    let tokens = 96;
    let h_lat = 8;
    let w_lat = 8; // seq = 64 > chunk
    block.apply_attn_env(16, false, 64, 0);

    let ctx = randn_bf16(&cuda, &[1, tokens, context], 0.02)?;
    let sample = randn_bf16(&cuda, &[1, h_lat, w_lat, hidden], 0.02)?;
    let err = block
        .forward_with_cond(
            &sample,
            &ctx,
            &bundle.driver_1280,
            Some(&bundle.time_proj_1536),
        )
        .expect_err("reuse guard must trip when ATTN_CHUNK_REUSE=0");
    let msg = err.to_string();
    assert!(
        msg.contains("ATTN_CHUNK_REUSE"),
        "expected reuse guard message, got {msg}"
    );

    Ok(())
}

#[test]
fn sdxl_block_respects_workspace_cap() -> Result<()> {
    let Some(flame_device) = cuda_device_or_skip()? else {
        return Ok(());
    };
    let cuda = flame_device.cuda_device_arc();
    let hidden = 1280;
    let context = 2048;
    let mut block = synthetic_block(&cuda, hidden, context)?;
    let bundle = synthetic_bundle(&cuda, 1)?;

    // Force large chunks so the guard triggers.
    let tokens = 128;
    let h_lat = 32;
    let w_lat = 32; // seq = 1024
    block.apply_attn_env(512, true, 1, 0);

    let ctx = randn_bf16(&cuda, &[1, tokens, context], 0.02)?;
    let sample = randn_bf16(&cuda, &[1, h_lat, w_lat, hidden], 0.02)?;
    let err = block
        .forward_with_cond(
            &sample,
            &ctx,
            &bundle.driver_1280,
            Some(&bundle.time_proj_1536),
        )
        .expect_err("workspace guard must trigger when cap is too small");
    let msg = err.to_string();
    assert!(
        msg.contains("workspace"),
        "expected workspace guard message, got {msg}"
    );

    Ok(())
}

fn synthetic_block(cuda: &Arc<CudaDevice>, hidden: usize, context: usize) -> Result<SdxlBlockRuntime> {
    let ff_dim = hidden * 4;
    let mut tensors = Vec::with_capacity(26);

    tensors.push(ones_bf16(cuda, &[hidden])?);
    tensors.push(zeros_bf16(cuda, &[hidden])?);
    tensors.push(randn_bf16(cuda, &[hidden, hidden], 0.02)?);
    tensors.push(zeros_bf16(cuda, &[hidden])?);

    tensors.push(ones_bf16(cuda, &[hidden])?);
    tensors.push(zeros_bf16(cuda, &[hidden])?);
    tensors.push(randn_bf16(cuda, &[hidden, hidden], 0.02)?);
    tensors.push(randn_bf16(cuda, &[hidden, hidden], 0.02)?);
    tensors.push(randn_bf16(cuda, &[hidden, hidden], 0.02)?);
    tensors.push(randn_bf16(cuda, &[hidden, hidden], 0.02)?);
    tensors.push(zeros_bf16(cuda, &[hidden])?);

    tensors.push(ones_bf16(cuda, &[hidden])?);
    tensors.push(zeros_bf16(cuda, &[hidden])?);
    tensors.push(randn_bf16(cuda, &[hidden, hidden], 0.02)?);
    tensors.push(randn_bf16(cuda, &[hidden, context], 0.02)?);
    tensors.push(randn_bf16(cuda, &[hidden, context], 0.02)?);
    tensors.push(randn_bf16(cuda, &[hidden, hidden], 0.02)?);
    tensors.push(zeros_bf16(cuda, &[hidden])?);

    tensors.push(ones_bf16(cuda, &[hidden])?);
    tensors.push(zeros_bf16(cuda, &[hidden])?);
    tensors.push(randn_bf16(cuda, &[ff_dim * 2, hidden], 0.02)?);
    tensors.push(zeros_bf16(cuda, &[ff_dim * 2])?);
    tensors.push(randn_bf16(cuda, &[hidden, ff_dim], 0.02)?);
    tensors.push(zeros_bf16(cuda, &[hidden])?);

    tensors.push(randn_bf16(cuda, &[hidden, hidden], 0.02)?);
    tensors.push(zeros_bf16(cuda, &[hidden])?);

    let mut block = SdxlBlockRuntime::from_mmap(0, tensors)?;
    block.mod1.weight =
        randn_bf16(cuda, &[block.mod1.cond_dim, 3 * hidden], 0.02)?;
    block.mod1.bias = randn_bf16(cuda, &[3 * hidden], 0.01)?;
    block.mod2.weight =
        randn_bf16(cuda, &[block.mod2.cond_dim, 3 * hidden], 0.02)?;
    block.mod2.bias = randn_bf16(cuda, &[3 * hidden], 0.01)?;
    block.mod3.weight =
        randn_bf16(cuda, &[block.mod3.cond_dim, 3 * hidden], 0.02)?;
    block.mod3.bias = randn_bf16(cuda, &[3 * hidden], 0.01)?;
    Ok(block)
}

fn synthetic_bundle(cuda: &Arc<CudaDevice>, batch: usize) -> Result<ConditioningBundle> {
    Ok(ConditioningBundle {
        driver_1280: randn_bf16(cuda, &[batch, DRIVER_DIM], 0.02)?,
        time_proj_1536: randn_bf16(cuda, &[batch, TIME_PROJ_DIM], 0.02)?,
    })
}

fn cuda_device_or_skip() -> Result<Option<FlameDevice>> {
    match FlameDevice::cuda(0) {
        Ok(device) => Ok(Some(device)),
        Err(err) => {
            eprintln!("[sdxl-dtype] skipping test: {err}");
            Ok(None)
        }
    }
}

fn randn_bf16(cuda: &Arc<CudaDevice>, shape: &[usize], std: f32) -> Result<Tensor> {
    let tensor = Tensor::randn(Shape::from_dims(shape), 0.0, std, cuda.clone())
        .map_err(anyhow::Error::from)?;
    tensor.to_dtype(DType::BF16).map_err(anyhow::Error::from)
}

fn ones_bf16(cuda: &Arc<CudaDevice>, shape: &[usize]) -> Result<Tensor> {
    Tensor::ones(Shape::from_dims(shape), cuda.clone())
        .map_err(anyhow::Error::from)?
        .to_dtype(DType::BF16)
        .map_err(anyhow::Error::from)
}

fn zeros_bf16(cuda: &Arc<CudaDevice>, shape: &[usize]) -> Result<Tensor> {
    Tensor::zeros_dtype(Shape::from_dims(shape), DType::BF16, cuda.clone())
        .map_err(anyhow::Error::from)
}
