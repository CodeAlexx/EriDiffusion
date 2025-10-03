//! SDXL layer registry (Phase-4).
//! Adapts the legacy EriDiffusion SDXL registry into the manifest-friendly runtime.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock};

use once_cell::sync::OnceCell;

use crate::tensor_utils::broadcast_to_as;
use anyhow::{anyhow, bail, ensure, Context, Result};
#[cfg(not(feature = "bf16_conv"))]
use flame_core::cuda_conv2d_fast::CudaConv2dFast;
use flame_core::debug_device::assert_cuda;
#[cfg(feature = "bf16_conv")]
use flame_core::{
    device::Device as FlameDevice,
    ops::conv2d_bf16::{Conv2dBF16, Conv2dBF16Cfg},
};
use flame_core::{DType, Tensor};

use super::RuntimeMode;
use super::{
    label_emb::LabelEmbedding,
    runtime::{with_attn_chunks, AttnChunkConfig, ExecutableBlock, SdxlBlockRuntime},
    weights::SdxlWeightProvider,
};
use crate::conditioning::make_conditioning::make_conditioning_with_sinusoidal;
#[cfg(feature = "cond_time_ids_mlp")]
use crate::conditioning::make_conditioning::make_conditioning_with_time_ids;
use crate::streaming::WeightProvider;
#[cfg(feature = "cond_time_ids_mlp")]
use eridiffusion_models::sdxl::blocks::timeids_mlp::TimeIdsMLP;

pub struct SdxlLayerRegistry {
    provider: Arc<SdxlWeightProvider>,
    mode: RuntimeMode,
    blocks: Vec<OnceCell<Arc<SdxlBlockRuntime>>>,
    block_count: usize,
    hidden_dim: usize,
    label_emb: LabelEmbedding,
    lora_keys: BTreeMap<String, (String, String)>,
    expected_checkpoint: BTreeSet<String>,
    stage_transitions: Vec<Option<StageTransition>>,
    #[cfg(feature = "cond_time_ids_mlp")]
    time_ids_mlp: Option<Arc<TimeIdsMLP>>,
}

fn trace_verbose() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("SDXL_TRACE_VERBOSE").ok().as_deref() == Some("1"))
}

pub struct ConditioningBundle {
    pub driver_1280: Tensor,
    pub time_proj_1536: Tensor,
}

#[derive(Clone)]
pub struct StageTransition {
    kind: TransitionKind,
    down_aux: Arc<Mutex<Option<Tensor>>>,
}

#[derive(Clone, Copy)]
enum FuseStrategy {
    None,
    Add,
    Cat,
    CatThenProject,
}

#[derive(Clone)]
enum TransitionKind {
    DownExpand {
        down_weight: Tensor,
        down_bias: Option<Tensor>,
        expand_weight: Tensor,
        expand_bias: Option<Tensor>,
    },
    Up {
        upsample: Option<UpSampleMode>,
        align_weight: Option<Tensor>,
        align_bias: Option<Tensor>,
        fuse: FuseStrategy,
        needs_skip: bool,
    },
}

#[derive(Clone)]
enum UpSampleMode {
    Nearest2xConv { weight: Tensor, bias: Option<Tensor> },
}

impl StageTransition {
    const UP_TRANSITION_BLOCK_INDICES: &[usize] = &[64, 68];

    fn build(provider: &SdxlWeightProvider, block_count: usize) -> Result<Vec<Option<Self>>> {
        let mut transitions: Vec<Option<StageTransition>> = Vec::with_capacity(block_count);
        transitions.resize_with(block_count, || None);

        if block_count > 0 {
            transitions[0] = Some(Self::build_down(
                provider,
                "model.diffusion_model.input_blocks.3.0",
                "model.diffusion_model.input_blocks.4.0",
            )?);
        }

        if block_count > 4 {
            transitions[4] = Some(Self::build_down(
                provider,
                "model.diffusion_model.input_blocks.6.0",
                "model.diffusion_model.input_blocks.7.0",
            )?);
        }

        for &idx in Self::UP_TRANSITION_BLOCK_INDICES {
            if block_count > idx {
                transitions[idx] = Some(Self::build_up(provider, idx)?);
            }
        }

        Ok(transitions)
    }

    fn build_down(
        provider: &SdxlWeightProvider,
        down_base: &str,
        expand_base: &str,
    ) -> Result<Self> {
        let down_w =
            provider.load_tensor(&format!("{down_base}.op.weight"))?.to_dtype(DType::F32)?;
        let down_b =
            Some(provider.load_tensor(&format!("{down_base}.op.bias"))?.to_dtype(DType::F32)?);
        let expand_w = provider
            .load_tensor(&format!("{expand_base}.skip_connection.weight"))?
            .to_dtype(DType::F32)?;
        let expand_b = Some(
            provider
                .load_tensor(&format!("{expand_base}.skip_connection.bias"))?
                .to_dtype(DType::F32)?,
        );

        Ok(Self {
            kind: TransitionKind::DownExpand {
                down_weight: down_w,
                down_bias: down_b,
                expand_weight: expand_w,
                expand_bias: expand_b,
            },
            down_aux: Arc::new(Mutex::new(None)),
        })
    }

    fn build_up(provider: &SdxlWeightProvider, block_index: usize) -> Result<Self> {
        let upsample = match block_index {
            64 => Some(Self::load_upsample(provider, "model.diffusion_model.output_blocks.2.2")?),
            _ => None,
        };

        let base = super::keymap::BASES
            .get(block_index)
            .ok_or_else(|| anyhow!("up transition block index {block_index} out of range"))?;

        let align_key = match base.rsplit_once(".transformer_blocks.") {
            Some((prefix, _)) => {
                let parent_up = prefix.rsplit_once('.').map(|(p, _)| p).unwrap_or(prefix);
                format!("{parent_up}.0.skip_connection.weight")
            }
            None => "model.diffusion_model.output_blocks.3.0.skip_connection.weight".to_string(),
        };
        let align_bias_key = align_key.replace("weight", "bias");
        let align_weight = Some(provider.load_tensor(&align_key)?.to_dtype(DType::BF16)?);
        let align_bias = Some(provider.load_tensor(&align_bias_key)?.to_dtype(DType::BF16)?);
        if let Some(ref w) = align_weight {
            let align_shape = w.shape().dims().to_vec();
            if trace_verbose() {
                eprintln!("[stage-up] align weight {:?}", align_shape);
            }
            ensure!(
                align_shape.len() == 4,
                "align weight must be rank-4 [oc,ic,1,1], got {:?}",
                align_shape
            );
            ensure!(
                align_shape[2] == 1 && align_shape[3] == 1,
                "align conv must be 1x1, got {:?}",
                align_shape
            );
        }

        let needs_skip = matches!(block_index, 64 | 68);

        if std::env::var_os("SDXL_STAGE_DEBUG").is_some() {
            eprintln!("[stage-up:build] idx={} needs_skip={}", block_index, needs_skip);
        }

        Ok(Self {
            kind: TransitionKind::Up {
                upsample,
                align_weight,
                align_bias,
                fuse: FuseStrategy::CatThenProject,
                needs_skip,
            },
            down_aux: Arc::new(Mutex::new(None)),
        })
    }

    fn load_upsample(provider: &SdxlWeightProvider, base: &str) -> Result<UpSampleMode> {
        let weight_key = format!("{base}.conv.weight");
        let bias_key = format!("{base}.conv.bias");
        let up_weight_torch =
            provider.load_tensor(&weight_key)?.to_dtype(DType::F32)?.clone_result()?;
        let up_bias_tensor = provider.load_tensor(&bias_key)?.to_dtype(DType::F32)?;

        // Permute to the layout expected by conv_transpose2d: [kh, kw, ic, oc]
        let up_weight_perm = up_weight_torch.permute(&[2, 3, 1, 0])?.clone_result()?;
        let shape_perm = up_weight_perm.shape().dims().to_vec();
        ensure!(
            shape_perm.len() == 4,
            "upsample weight must be rank-4 [kh,kw,ic,oc], got {:?}",
            shape_perm
        );
        let kernel_h = shape_perm[0];
        let kernel_w = shape_perm[1];
        let ic = shape_perm[2];
        let oc = shape_perm[3];
        ensure!(ic > 0 && oc > 0, "upsample weight must have valid channels ic={} oc={}", ic, oc);

        let up_bias_clone = up_bias_tensor.clone();
        let mode = match (kernel_h, kernel_w) {
            (4, 4) => {
                let converted = Self::convert_conv_transpose4_to_conv3(&up_weight_perm)?;
                if trace_verbose() {
                    eprintln!(
                        "[sdxl::registry] converted conv_transpose 4x4 -> nearest+conv3 (ic={} oc={})",
                        converted.shape().dims()[1],
                        converted.shape().dims()[0]
                    );
                }
                UpSampleMode::Nearest2xConv { weight: converted, bias: Some(up_bias_clone) }
            }
            (3, 3) => {
                if trace_verbose() {
                    eprintln!(
                        "[sdxl::registry] using nearest+conv3 (ic={} oc={})",
                        up_weight_torch.shape().dims()[1],
                        up_weight_torch.shape().dims()[0]
                    );
                }
                UpSampleMode::Nearest2xConv { weight: up_weight_torch, bias: Some(up_bias_clone) }
            }
            other => {
                bail!("unsupported upsample kernel {:?} for {base}", other)
            }
        };
        Ok(mode)
    }

    pub fn consumes_skip(&self) -> bool {
        matches!(self.kind, TransitionKind::Up { needs_skip: true, .. })
    }

    pub fn produces_skip(&self) -> bool {
        matches!(self.kind, TransitionKind::DownExpand { .. })
    }

    fn set_down_aux(&self, tensor: Tensor) {
        if let Ok(mut slot) = self.down_aux.lock() {
            *slot = Some(tensor);
        }
    }

    pub fn take_down_aux(&self) -> Option<Tensor> {
        match self.down_aux.lock() {
            Ok(mut slot) => slot.take(),
            Err(_) => None,
        }
    }

    pub fn apply(
        &self,
        sample: &Tensor,
        skip_in: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        use anyhow::{anyhow, bail, Context};

        if sample.shape().dims().len() != 4 {
            bail!("stage transition expects NHWC rank-4 tensor, got {:?}", sample.shape().dims());
        }

        let debug = std::env::var_os("SDXL_STAGE_DEBUG").is_some();

        match &self.kind {
            TransitionKind::DownExpand { down_weight, down_bias, expand_weight, expand_bias } => {
                let skip_for_stack = sample.clone_result()?;
                #[cfg(feature = "bf16_conv")]
                {
                    if debug {
                        eprintln!("[stage-down] in {:?}", sample.shape().dims(),);
                    }
                    let flame_device = FlameDevice::from(sample.device().clone());
                    let down_weight_bf16 = if down_weight.dtype() == DType::BF16 {
                        down_weight.clone_result()?
                    } else {
                        down_weight.to_dtype(DType::BF16)?
                    };
                    let down_bias_bf16 = match down_bias {
                        Some(b) if b.dtype() == DType::BF16 => Some(b.clone_result()?),
                        Some(b) => Some(b.to_dtype(DType::BF16)?),
                        None => None,
                    };
                    let mut down_conv = Conv2dBF16::new(
                        &flame_device,
                        Conv2dBF16Cfg { stride: (2, 2), pad: (1, 1), dil: (1, 1), groups: 1 },
                    )?;
                    let down =
                        down_conv.forward(sample, &down_weight_bf16, down_bias_bf16.as_ref())?;
                    if debug {
                        eprintln!("[stage-down] after stride2 {:?}", down.shape().dims());
                    }

                    let aux = down.clone_result()?;
                    if debug {
                        eprintln!("[stage-down] captured down aux {:?}", aux.shape().dims());
                    }
                    self.set_down_aux(aux);

                    let expand_weight_bf16 = if expand_weight.dtype() == DType::BF16 {
                        expand_weight.clone_result()?
                    } else {
                        expand_weight.to_dtype(DType::BF16)?
                    };
                    let expand_bias_bf16 = match expand_bias {
                        Some(b) if b.dtype() == DType::BF16 => Some(b.clone_result()?),
                        Some(b) => Some(b.to_dtype(DType::BF16)?),
                        None => None,
                    };
                    let mut expand_conv = Conv2dBF16::new(
                        &flame_device,
                        Conv2dBF16Cfg { stride: (1, 1), pad: (0, 0), dil: (1, 1), groups: 1 },
                    )?;
                    let expanded = expand_conv.forward(
                        &down,
                        &expand_weight_bf16,
                        expand_bias_bf16.as_ref(),
                    )?;
                    if debug {
                        eprintln!("[stage-down] expanded {:?}", expanded.shape().dims());
                    }
                    return Ok((expanded, Some(skip_for_stack)));
                }

                #[cfg(not(feature = "bf16_conv"))]
                {
                    let x_nchw = sample
                        .permute(&[0, 3, 1, 2])
                        .context("permute NHWC->NCHW failed for down transition")?;
                    let x_f32 = x_nchw.to_dtype(DType::F32).context("down transition: to F32")?;

                    let down = x_f32
                        .conv2d(down_weight, down_bias.as_ref(), 2, 1)
                        .context("down transition: stride-2 conv")?;
                    drop(x_f32);
                    let down_bf16 = down
                        .to_dtype(DType::BF16)
                        .context("down transition: to BF16 after stride-2 conv")?;
                    drop(down);
                    if debug {
                        eprintln!(
                            "[stage-down:compat] after stride2 {:?}",
                            down_bf16.shape().dims()
                        );
                    }

                    let skip_nhwc = down_bf16
                        .permute(&[0, 2, 3, 1])
                        .context("down transition: permute stride-2 result to NHWC")?;
                    let aux =
                        skip_nhwc.clone_result().context("down transition: own stride-2 skip")?;
                    if debug {
                        eprintln!("[stage-down:compat] captured down aux {:?}", aux.shape().dims());
                    }
                    self.set_down_aux(aux);

                    let down_f32 = down_bf16
                        .to_dtype(DType::F32)
                        .context("down transition: to F32 before expand conv")?;
                    let expanded = down_f32
                        .conv2d(expand_weight, expand_bias.as_ref(), 1, 0)
                        .context("down transition: expand conv")?;
                    drop(down_f32);
                    let expanded_bf16 = expanded
                        .to_dtype(DType::BF16)
                        .context("down transition: to BF16 after expand conv")?;
                    drop(expanded);
                    if debug {
                        eprintln!(
                            "[stage-down:compat] expanded {:?}",
                            expanded_bf16.shape().dims()
                        );
                    }
                    let expanded_nhwc = expanded_bf16
                        .permute(&[0, 2, 3, 1])
                        .context("down transition: permute back to NHWC")?;
                    if debug {
                        eprintln!(
                            "[stage-down:compat] expanded_nhwc {:?}",
                            expanded_nhwc.shape().dims()
                        );
                    }

                    Ok((expanded_nhwc, Some(skip_for_stack)))
                }
            }
            TransitionKind::Up { upsample, align_weight, align_bias, fuse, .. } => {
                let skip_tensor = skip_in;
                let mut returned_skip: Option<Tensor> = None;
                let main_view = match upsample {
                    Some(UpSampleMode::Nearest2xConv { weight, bias }) => {
                        Self::upsample_nearest_then_conv_nhwc(sample, weight, bias.as_ref())?
                    }
                    None => {
                        if sample.dtype() == DType::BF16 {
                            sample.clone_result()?
                        } else {
                            sample
                                .to_dtype(DType::BF16)
                                .context("up transition: main to BF16 (no upsample)")?
                        }
                    }
                };
                if debug {
                    let skip_dims = skip_tensor.map(|t| t.shape().dims().to_vec());
                    eprintln!(
                        "[stage-up] main {:?} skip {:?}",
                        main_view.shape().dims(),
                        skip_dims
                    );
                }
                assert_cuda("up transition:main_view", &main_view)?;

                let main_nhwc = main_view.clone_result().context("up transition: own main")?;
                assert_cuda("up transition:main", &main_nhwc)?;

                let fused = match fuse {
                    FuseStrategy::None => main_nhwc,
                    FuseStrategy::Add | FuseStrategy::Cat | FuseStrategy::CatThenProject => {
                        let main_shape = main_nhwc.shape().dims().to_vec();
                        ensure!(main_shape.len() == 4, "main tensor must be NHWC");
                        let main_channels = main_shape[3] as usize;

                        match fuse {
                            FuseStrategy::Add => {
                                let skip_ref = skip_tensor.ok_or_else(|| {
                                    anyhow!("skip stack underflow during up transition")
                                })?;
                                let skip_bf16 = if skip_ref.dtype() != DType::BF16 {
                                    skip_ref
                                        .to_dtype(DType::BF16)
                                        .context("up transition: skip to BF16")?
                                } else {
                                    skip_ref.clone_result()?
                                };
                                returned_skip = Some(skip_bf16.clone_result()?);
                                let skip_shape = skip_bf16.shape().dims().to_vec();
                                ensure!(skip_shape.len() == 4, "skip tensor must be NHWC");
                                ensure!(
                                    main_shape[0] == skip_shape[0]
                                        && main_shape[1] == skip_shape[1]
                                        && main_shape[2] == skip_shape[2],
                                    "up transition: spatial dims mismatch main={:?} skip={:?}",
                                    main_shape,
                                    skip_shape
                                );
                                ensure!(
                                    main_shape[3] == skip_shape[3],
                                    "up transition: add fuse requires matching channels, main={}, skip={}",
                                    main_shape[3],
                                    skip_shape[3]
                                );
                                let sum =
                                    main_nhwc.add(&skip_bf16).context("up transition: add fuse")?;
                                assert_cuda("up transition:add", &sum)?;
                                let sum_owned =
                                    sum.clone_result().context("up transition: own add result")?;
                                assert_cuda("up transition:add_owned", &sum_owned)?;
                                sum_owned
                            }
                            FuseStrategy::Cat => {
                                let skip_ref = skip_tensor.ok_or_else(|| {
                                    anyhow!("skip stack underflow during up transition")
                                })?;
                                let skip_bf16 = if skip_ref.dtype() != DType::BF16 {
                                    skip_ref
                                        .to_dtype(DType::BF16)
                                        .context("up transition: skip to BF16")?
                                } else {
                                    skip_ref.clone_result()?
                                };
                                returned_skip = Some(skip_bf16.clone_result()?);
                                let skip_shape = skip_bf16.shape().dims().to_vec();
                                ensure!(skip_shape.len() == 4, "skip tensor must be NHWC");
                                ensure!(
                                    main_shape[0] == skip_shape[0]
                                        && main_shape[1] == skip_shape[1]
                                        && main_shape[2] == skip_shape[2],
                                    "up transition: spatial dims mismatch main={:?} skip={:?}",
                                    main_shape,
                                    skip_shape
                                );
                                let cat = Tensor::cat(&[&main_nhwc, &skip_bf16], 3)
                                    .context("up transition: cat fuse")?;
                                assert_cuda("up transition:cat", &cat)?;
                                let cat_owned =
                                    cat.clone_result().context("up transition: own cat result")?;
                                assert_cuda("up transition:cat_owned", &cat_owned)?;
                                ensure!(
                                    cat_owned.shape().dims()[3] == main_shape[3] + skip_shape[3],
                                    "up transition: cat channels mismatch"
                                );
                                cat_owned
                            }
                            FuseStrategy::CatThenProject => {
                                let align_w = align_weight.as_ref().ok_or_else(|| {
                                    anyhow!("missing align weight for CatThenProject fuse")
                                })?;
                                let align_b = align_bias.as_ref();

                                let align_w_f32 = if align_w.dtype() == DType::F32 {
                                    align_w.clone_result()?
                                } else {
                                    align_w
                                        .to_dtype(DType::F32)
                                        .context("up transition: align weight to F32")?
                                };
                                let align_b_f32 = match align_b {
                                    Some(b) if b.dtype() == DType::F32 => Some(b.clone_result()?),
                                    Some(b) => Some(
                                        b.to_dtype(DType::F32)
                                            .context("up transition: align bias to F32")?,
                                    ),
                                    None => None,
                                };

                                if trace_verbose() {
                                    eprintln!(
                                        "[stage-up] align weight {:?}",
                                        align_w_f32.shape().dims()
                                    );
                                }
                                let align_shape = align_w_f32.shape().dims().to_vec();
                                ensure!(
                                    align_shape.len() == 4,
                                    "align conv weight must be rank-4, got {:?}",
                                    align_shape
                                );
                                let (out_c, in_c) =
                                    (align_shape[0] as usize, align_shape[1] as usize);

                                let mut owned_parts: Vec<Tensor> = Vec::new();
                                let main_cat = main_nhwc
                                    .clone_result()
                                    .context("up transition: own main (align)")?;
                                owned_parts.push(main_cat);
                                let mut total_channels = main_channels;

                                if let Some(skip_ref) = skip_tensor {
                                    let skip_bf16 = if skip_ref.dtype() != DType::BF16 {
                                        skip_ref
                                            .to_dtype(DType::BF16)
                                            .context("up transition: skip to BF16")?
                                    } else {
                                        skip_ref.clone_result()?
                                    };
                                    returned_skip = Some(skip_bf16.clone_result()?);
                                    let skip_shape = skip_bf16.shape().dims().to_vec();
                                    ensure!(skip_shape.len() == 4, "skip tensor must be NHWC");
                                    ensure!(
                                        main_shape[0] == skip_shape[0]
                                            && main_shape[1] == skip_shape[1]
                                            && main_shape[2] == skip_shape[2],
                                        "up transition: spatial dims mismatch main={:?} skip={:?}",
                                        main_shape,
                                        skip_shape
                                    );
                                    total_channels += skip_shape[3] as usize;
                                    owned_parts.push(skip_bf16);
                                }

                                ensure!(
                                    total_channels == in_c,
                                    "align conv input channels mismatch: weight expects {}, got {}",
                                    in_c,
                                    total_channels
                                );

                                let cat_refs: Vec<&Tensor> = owned_parts.iter().collect();
                                let cat =
                                    Tensor::cat(&cat_refs, 3).context("up transition: cat fuse")?;
                                assert_cuda("up transition:cat_pre_proj", &cat)?;
                                let cat =
                                    cat.clone_result().context("up transition: own cat result")?;
                                assert_cuda("up transition:cat_owned", &cat)?;
                                let cat_nchw = cat
                                    .permute(&[0, 3, 1, 2])
                                    .context("up transition: cat NHWC->NCHW")?;
                                assert_cuda("up transition:cat_nchw", &cat_nchw)?;
                                let cat_f32 = cat_nchw
                                    .to_dtype(DType::F32)
                                    .context("up transition: cat to F32")?;
                                let proj = cat_f32
                                    .conv2d(&align_w_f32, align_b_f32.as_ref(), 1, 0)
                                    .context("up transition: align conv")?;
                                drop(cat_f32);
                                assert_cuda("up transition:align_conv", &proj)?;
                                let proj_bf16 = proj
                                    .to_dtype(DType::BF16)
                                    .context("up transition: align conv to BF16")?;
                                drop(proj);
                                let proj_nhwc = proj_bf16
                                    .permute(&[0, 2, 3, 1])
                                    .context("up transition: align conv NCHW->NHWC")?;
                                assert_cuda("up transition:proj_nhwc", &proj_nhwc)?;
                                ensure!(
                                    proj_nhwc.shape().dims()[3] as usize == out_c,
                                    "align conv output channels mismatch: expected {}, got {}",
                                    out_c,
                                    proj_nhwc.shape().dims()[3]
                                );
                                let proj_out = proj_nhwc
                                    .clone_result()
                                    .context("up transition: own align result")?;
                                assert_cuda("up transition:proj_out", &proj_out)?;
                                proj_out
                            }
                            FuseStrategy::None => unreachable!(),
                        }
                    }
                };

                Ok((fused, returned_skip))
            }
        }
    }

    fn upsample_nearest2x_nhwc(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        ensure!(dims.len() == 4, "upsample expects NHWC tensor, got {:?}", dims);
        let (b, h, w, c) = (dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize);
        ensure!(h > 0 && w > 0 && c > 0, "upsample expects positive spatial dims, got {:?}", dims);

        assert_cuda("upsample_nearest2x:input", x)?;

        let x_bf16 =
            if x.dtype() == DType::BF16 { x.clone_result()? } else { x.to_dtype(DType::BF16)? };

        let reshaped = x_bf16.reshape(&[b, h, 1, w, 1, c])?;
        let broadcast = broadcast_to_as(&reshaped, &[b, h, 2, w, 2, c], DType::BF16)?;
        let view = broadcast.reshape(&[b, h * 2, w * 2, c])?;
        let owning = view.clone_result().context("upsample_nearest2x_nhwc: own result")?;
        assert_cuda("upsample_nearest2x:owning", &owning)?;

        debug_assert_eq!(owning.dtype(), DType::BF16);

        Ok(owning)
    }

    fn upsample_nearest_then_conv_nhwc(
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        assert_cuda("nearest_conv:input", x)?;
        let upsampled = Self::upsample_nearest2x_nhwc(x)?;
        assert_cuda("nearest_conv:upsampled", &upsampled)?;
        let dims = upsampled.shape().dims().to_vec();
        ensure!(dims.len() == 4, "nearest2x conv expects NHWC tensor, got {:?}", dims);
        let (_batch, h, w, c_in) = (dims[0], dims[1], dims[2], dims[3]);

        #[cfg(feature = "bf16_conv")]
        {
            let _ = (h, w);
            let flame_device = FlameDevice::from(upsampled.device().clone());
            let weight_bf16 = if weight.dtype() == DType::BF16 {
                weight.clone_result()?
            } else {
                weight.to_dtype(DType::BF16)?
            };
            let bias_bf16 = match bias {
                Some(b) if b.dtype() == DType::BF16 => Some(b.clone_result()?),
                Some(b) => Some(b.to_dtype(DType::BF16)?),
                None => None,
            };

            let mut conv = Conv2dBF16::new(
                &flame_device,
                Conv2dBF16Cfg { stride: (1, 1), pad: (1, 1), dil: (1, 1), groups: 1 },
            )?;
            return Ok(conv.forward(&upsampled, &weight_bf16, bias_bf16.as_ref())?);
        }

        #[cfg(not(feature = "bf16_conv"))]
        {
            let weight_shape = weight.shape().dims().to_vec();
            ensure!(
                weight_shape.len() == 4,
                "conv weight must be rank-4 [oc,ic,kh,kw], got {:?}",
                weight_shape
            );
            let (out_c, in_c, kh, kw) =
                (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]);
            ensure!(in_c == c_in, "conv weight expects {} channels, got {}", in_c, c_in);
            ensure!(kh == 3 && kw == 3, "nearest2x conv expects 3x3 kernel, got {:?}", (kh, kw));
            if let Some(b) = bias {
                let bias_shape = b.shape().dims().to_vec();
                ensure!(
                    bias_shape.len() == 1 && bias_shape[0] == out_c,
                    "conv bias must match output channels {}",
                    out_c
                );
            }

            let up_nchw = upsampled.permute(&[0, 3, 1, 2]).context("nearest2x conv: NHWC->NCHW")?;
            assert_cuda("nearest_conv:permute", &up_nchw)?;
            let up_f32 = up_nchw.to_dtype(DType::F32).context("nearest2x conv: input to F32")?;
            let conv = CudaConv2dFast::conv2d_forward(&up_f32, weight, bias, (1, 1), (1, 1), 1)
                .context("nearest2x conv: conv2d_fast")?;
            drop(up_f32);
            assert_cuda("nearest_conv:conv_out", &conv)?;

            if trace_verbose() {
                eprintln!(
                    "[sdxl::upsample] nearest2x conv3x3 using conv2d_fast (batch={} in={} out={})",
                    dims[0], c_in, out_c
                );
            }

            let conv_bf16 = conv.to_dtype(DType::BF16).context("nearest2x conv: to BF16")?;
            drop(conv);
            let conv_nhwc =
                conv_bf16.permute(&[0, 2, 3, 1]).context("nearest2x conv: NCHW->NHWC")?;
            assert_cuda("nearest_conv:permute_back", &conv_nhwc)?;
            ensure!(
                conv_nhwc.shape().dims()[1] == h && conv_nhwc.shape().dims()[2] == w,
                "nearest2x conv spatial mismatch"
            );

            let out = conv_nhwc.clone_result().context("nearest2x conv: own result")?;
            assert_cuda("nearest_conv:owning", &out)?;
            Ok(out)
        }
    }

    fn convert_conv_transpose4_to_conv3(weight: &Tensor) -> Result<Tensor> {
        ensure!(
            weight.dtype() == DType::F32,
            "expected F32 weight for conv transpose, got {:?}",
            weight.dtype()
        );
        let dims = weight.shape().dims().to_vec();
        ensure!(dims.len() == 4, "conv transpose weight must be [kh,kw,ic,oc], got {:?}", dims);
        let (kh, kw, in_c, out_c) = (dims[0], dims[1], dims[2], dims[3]);
        ensure!(kh == 4 && kw == 4, "expected 4x4 kernel, got {}x{}", kh, kw);

        let host = weight.to_vec()?;
        let mut out = vec![0f32; out_c * in_c * 3 * 3];
        let kw4 = kw;
        let eps = 1e-4f32;

        for ic_idx in 0..in_c {
            for oc_idx in 0..out_c {
                let mut w4 = [[0f32; 4]; 4];
                for p in 0..4 {
                    for q in 0..4 {
                        let idx = (((p * kw4 + q) * in_c + ic_idx) * out_c) + oc_idx;
                        w4[p][q] = host[idx];
                    }
                }

                let k00 = w4[0][0];
                let k01 = w4[0][1] - k00;
                let k02 = w4[0][2] - k01;
                debug_assert!((w4[0][3] - k02).abs() < eps);

                let k10 = w4[1][0] - k00;
                let k11 = w4[1][1] - (k10 + k01 + k00);
                let k12 = w4[1][2] - (k11 + k02 + k01);
                debug_assert!((w4[1][3] - (k12 + k02)).abs() < eps);

                let k20 = w4[2][0] - k10;
                let k21 = w4[2][1] - (k20 + k11 + k10);
                let k22 = w4[2][2] - (k21 + k12 + k11);
                debug_assert!((w4[2][3] - (k22 + k12)).abs() < eps);
                debug_assert!((w4[3][0] - k20).abs() < eps);
                debug_assert!((w4[3][1] - (k21 + k20)).abs() < eps);
                debug_assert!((w4[3][2] - (k22 + k21)).abs() < eps);
                debug_assert!((w4[3][3] - k22).abs() < eps);

                let k = [[k00, k01, k02], [k10, k11, k12], [k20, k21, k22]];
                for (r, row) in k.iter().enumerate() {
                    for (c, &val) in row.iter().enumerate() {
                        let idx = ((((oc_idx * in_c) + ic_idx) * 3 + r) * 3) + c;
                        out[idx] = val;
                    }
                }
            }
        }

        let shape = flame_core::Shape::from_dims(&[out_c, in_c, 3, 3]);
        Tensor::from_vec(out, shape, weight.device().clone())
            .context("convert_conv_transpose4_to_conv3: create tensor")
    }
}

impl SdxlLayerRegistry {
    fn infer_hidden_dim(provider: &SdxlWeightProvider) -> Result<usize> {
        let base = super::keymap::BASES
            .first()
            .ok_or_else(|| anyhow!("SDXL registry requires at least one block"))?;
        let key = format!("{base}.attn1.to_q.weight");
        let shape = provider.tensor_shape(&key)?;
        ensure!(shape.len() == 2, "attn1.to_q.weight expected rank 2, got {:?}", shape);
        Ok(shape[1])
    }

    fn parity_trace_enabled() -> bool {
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("SDXL_PARITY_TRACE").ok().as_deref() == Some("1"))
    }

    fn parity_check(name: &str, tensor: &Tensor) -> anyhow::Result<()> {
        let f32 = tensor.to_dtype(DType::F32)?;
        let data = f32.to_vec_f32()?;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut count = 0usize;
        let mut nan = 0usize;
        for &v in &data {
            if v.is_finite() {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
                sum += v as f64;
                count += 1;
            } else {
                nan += 1;
            }
        }
        let mean = if count > 0 { (sum / count as f64) as f32 } else { f32::NAN };
        eprintln!("[parity] block={name} mean={mean:.6} min={:.6} max={:.6} nan={nan}", min, max);
        if nan > 0 {
            return Err(anyhow!(
                "parity detected non-finite values in block {name} (nan count={nan})"
            ));
        }
        Ok(())
    }

    pub fn new(
        provider: Arc<SdxlWeightProvider>,
        mode: RuntimeMode,
        blocks: Vec<OnceCell<Arc<SdxlBlockRuntime>>>,
        block_count: usize,
        hidden_dim: usize,
        label_emb: LabelEmbedding,
        lora_keys: BTreeMap<String, (String, String)>,
        stage_transitions: Vec<Option<StageTransition>>,
    ) -> Self {
        let expected = lora_keys.values().flat_map(|(a, b)| [a.clone(), b.clone()]).collect();
        Self {
            provider,
            mode,
            blocks,
            block_count,
            hidden_dim,
            label_emb,
            lora_keys,
            expected_checkpoint: expected,
            stage_transitions,
            #[cfg(feature = "cond_time_ids_mlp")]
            time_ids_mlp: None,
        }
    }

    pub fn mode(&self) -> RuntimeMode {
        self.mode
    }

    pub fn block_count(&self) -> usize {
        self.block_count
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn block(&self, index: usize) -> anyhow::Result<Arc<SdxlBlockRuntime>> {
        self.materialize_block(index)
    }

    pub fn load_block_ephemeral(&self, index: usize) -> anyhow::Result<SdxlBlockRuntime> {
        let tensors = self.provider.load_block_to_gpu(index)?.tensors;
        SdxlBlockRuntime::from_mmap(index, tensors)
    }

    fn materialize_block(&self, index: usize) -> anyhow::Result<Arc<SdxlBlockRuntime>> {
        let cell =
            self.blocks.get(index).ok_or_else(|| anyhow!("block index {index} out of range"))?;
        let provider = self.provider.clone();
        let idx = index;
        let block = cell.get_or_try_init(|| {
            let tensors = provider.load_block_to_gpu(idx)?.tensors;
            SdxlBlockRuntime::from_mmap(idx, tensors).map(Arc::new)
        })?;
        Ok(block.clone())
    }

    pub fn forward_blocks(
        &self,
        sample: Tensor,
        ctx: &Tensor,
        cond: &ConditioningBundle,
        attn_chunks: AttnChunkConfig,
    ) -> anyhow::Result<Tensor> {
        with_attn_chunks(attn_chunks, move || {
            let mut skip_stack: Vec<Tensor> = Vec::new();
            let mut current = sample;
            let stage_debug = std::env::var_os("SDXL_STAGE_DEBUG").is_some();
            for idx in 0..self.block_count() {
                if let Some(trans) = self.transition_for(idx) {
                    if stage_debug {
                        eprintln!(
                            "[stage] before transition idx={} dims={:?}",
                            idx,
                            current.shape().dims()
                        );
                    }
                    let owned_skip = if trans.consumes_skip() {
                        Some(skip_stack.pop().ok_or_else(|| {
                            anyhow!("skip stack underflow at transition index {idx}")
                        })?)
                    } else {
                        None
                    };
                    let skip_ref = owned_skip.as_ref();
                    let (next, maybe_skip) = trans.apply(&current, skip_ref)?;
                    if let Some(s) = maybe_skip {
                        skip_stack.push(s);
                    }
                    if stage_debug {
                        eprintln!(
                            "[stage] after transition idx={} dims={:?}",
                            idx,
                            next.shape().dims()
                        );
                    }
                    current = next;
                }

                let block = self.materialize_block(idx)?;
                current = block.forward_with_cond(
                    &current,
                    ctx,
                    &cond.driver_1280,
                    Some(&cond.time_proj_1536),
                )?;
                if Self::parity_trace_enabled() {
                    Self::parity_check(block.name(), &current)?;
                }
            }
            Ok(current)
        })
    }

    pub fn transition_for(&self, index: usize) -> Option<&StageTransition> {
        self.stage_transitions.get(index).and_then(|slot| slot.as_ref())
    }

    pub fn label_emb(&self) -> &LabelEmbedding {
        &self.label_emb
    }

    pub fn make_conditioning(
        &self,
        pooled: &Tensor,
        raw_timesteps: &Tensor,
        time_ids: &Tensor,
    ) -> anyhow::Result<ConditioningBundle> {
        #[cfg(not(feature = "cond_time_ids_mlp"))]
        let _ = time_ids;
        #[cfg(feature = "cond_time_ids_mlp")]
        if let Some(time_mlp) = self.time_ids_mlp.as_ref() {
            let cond = make_conditioning_with_time_ids(pooled, pooled, time_ids, &|ids| {
                time_mlp.forward(ids)
            })?;
            return self.build_bundle(cond);
        }

        #[allow(unused_mut)]
        let cond = make_conditioning_with_sinusoidal(pooled, raw_timesteps, 0.0, 1000.0, 10_000.0)?;
        self.build_bundle(cond)
    }

    fn build_bundle(
        &self,
        cond: crate::conditioning::make_conditioning::SdxlCond,
    ) -> anyhow::Result<ConditioningBundle> {
        let driver = self.label_emb.forward_from_2816(&cond.cond_2816)?;
        let time_proj = if cond.time_proj_1536.dtype() == DType::BF16 {
            cond.time_proj_1536.clone_result()?
        } else {
            cond.time_proj_1536.to_dtype(DType::BF16)?
        };
        Ok(ConditioningBundle { driver_1280: driver, time_proj_1536: time_proj })
    }

    pub fn lora_keys(&self) -> &BTreeMap<String, (String, String)> {
        &self.lora_keys
    }

    pub fn expected_checkpoint_keys(&self) -> &BTreeSet<String> {
        &self.expected_checkpoint
    }

    pub fn build(provider: Arc<SdxlWeightProvider>, mode: RuntimeMode) -> Result<Self> {
        let head = provider.load_head_to_gpu()?.tensors;
        anyhow::ensure!(head.len() == 4, "SDXL head expected 4 tensors, got {}", head.len());
        let mut iter = head.into_iter();
        let w0 = iter.next().unwrap();
        let b0 = iter.next().unwrap();
        let w2 = iter.next().unwrap();
        let b2 = iter.next().unwrap();
        let label_emb = LabelEmbedding::new(w0, b0, w2, b2)?;

        let block_count = super::keymap::BASES.len();
        let blocks: Vec<OnceCell<Arc<SdxlBlockRuntime>>> =
            (0..block_count).map(|_| OnceCell::new()).collect();

        let hidden_dim =
            if block_count > 0 { Self::infer_hidden_dim(provider.as_ref())? } else { 4 };

        let mut lora_map = BTreeMap::new();
        let lora_sites = [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ];
        for (i, _) in super::keymap::BASES.iter().enumerate() {
            for site in &lora_sites {
                lora_map.insert(
                    format!("blocks.{i:02}.{site}"),
                    (format!("blocks.{i:02}.{site}.A"), format!("blocks.{i:02}.{site}.B")),
                );
            }
        }

        #[allow(unused_mut)]
        let transitions = StageTransition::build(provider.as_ref(), block_count)?;

        #[allow(unused_mut)]
        let mut registry = Self::new(
            provider,
            mode,
            blocks,
            block_count,
            hidden_dim,
            label_emb,
            lora_map,
            transitions,
        );
        #[cfg(feature = "cond_time_ids_mlp")]
        {
            let ordinal = provider.device().ordinal();
            let device = eridiffusion_core::Device::Cuda(ordinal);
            registry.time_ids_mlp = Some(Arc::new(TimeIdsMLP::new(device)?));
        }
        Ok(registry)
    }
}
