use std::sync::Arc;

use anyhow::{anyhow, Result};
use eridiffusion_common_vae::{decode as vae_decode, VaePolicy};
use eridiffusion_core::Device as ModelDevice;
use eridiffusion_models::devtensor::randn_on;
use eridiffusion_training::conditioning::{
    time_ids::build_time_ids, timestep_embedding::timestep_embedding,
};
use eridiffusion_training::sdxl::inference_runtime::SdxlInferenceRuntime;
use eridiffusion_training::sdxl::registry::SdxlLayerRegistry;
use eridiffusion_training::sdxl::weights::SdxlWeightProvider;
use eridiffusion_training::sdxl::{
    scheduler::{make_sigmas, sigma_to_timestep, ScheduleKind, SchedulerCfg},
    RuntimeMode,
};
use flame_core::{cuda_ops::GpuOps, group_norm, DType, Shape, Tensor};

use crate::sdxl::{
    prompt::{PromptEmbeddings, PromptEncoder},
    weights::SdxlResources,
};

struct Conv2dParams {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    padding: usize,
}

struct ResnetParams {
    in_norm_weight: Tensor,
    in_norm_bias: Tensor,
    in_conv: Conv2dParams,
    emb_proj_weight: Tensor,
    emb_proj_bias: Tensor,
    out_norm_weight: Tensor,
    out_norm_bias: Tensor,
    out_conv: Conv2dParams,
    skip_conv: Option<Conv2dParams>,
}

pub struct SdxlNativePipeline {
    prompt: PromptEncoder,
    registry: Arc<SdxlLayerRegistry>,
    infer_runtime: SdxlInferenceRuntime,
    vae_spec: eridiffusion_common_vae::VaeSpec,
    model_device: ModelDevice,
    time_fc1_weight: Tensor,
    time_fc1_bias: Tensor,
    time_fc2_weight: Tensor,
    time_fc2_bias: Tensor,
    conv_in: Conv2dParams,
    res_block1: ResnetParams,
    res_block2: ResnetParams,
    downsample: Conv2dParams,
    res_block4: ResnetParams,
    up_res_block6: ResnetParams,
    up_res_block7: ResnetParams,
    up_res_block8: ResnetParams,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    head_conv: Conv2dParams,
}

impl SdxlNativePipeline {
    pub fn new(
        resources: SdxlResources,
        runtime_mode: RuntimeMode,
        model_device: ModelDevice,
    ) -> Result<Self> {
        let flame_device = resources.prompt_encoder.device_clone();
        let mmap = resources.unet_mmap.clone();
        let provider = Arc::new(SdxlWeightProvider::new(mmap, flame_device.clone()));
        let registry = Arc::new(SdxlLayerRegistry::build(provider.clone(), runtime_mode)?);
        let infer_runtime = SdxlInferenceRuntime::new(registry.clone(), None, None, false)?;

        let time_fc1_weight =
            to_f32(provider.load_tensor("model.diffusion_model.time_embed.0.weight")?)?;
        let time_fc1_bias =
            to_f32(provider.load_tensor("model.diffusion_model.time_embed.0.bias")?)?;
        let time_fc2_weight =
            to_f32(provider.load_tensor("model.diffusion_model.time_embed.2.weight")?)?;
        let time_fc2_bias =
            to_f32(provider.load_tensor("model.diffusion_model.time_embed.2.bias")?)?;

        let conv_in = load_conv_params(
            provider.as_ref(),
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.input_blocks.0.0.bias",
            1,
            1,
        )?;
        let res_block1 =
            load_resnet_params(provider.as_ref(), "model.diffusion_model.input_blocks.1.0")?;
        let res_block2 =
            load_resnet_params(provider.as_ref(), "model.diffusion_model.input_blocks.2.0")?;
        let downsample = load_conv_params(
            provider.as_ref(),
            "model.diffusion_model.input_blocks.3.0.op.weight",
            "model.diffusion_model.input_blocks.3.0.op.bias",
            2,
            1,
        )?;
        let res_block4 =
            load_resnet_params(provider.as_ref(), "model.diffusion_model.input_blocks.4.0")?;
        let up_res_block6 =
            load_resnet_params(provider.as_ref(), "model.diffusion_model.output_blocks.6.0")?;
        let up_res_block7 =
            load_resnet_params(provider.as_ref(), "model.diffusion_model.output_blocks.7.0")?;
        let up_res_block8 =
            load_resnet_params(provider.as_ref(), "model.diffusion_model.output_blocks.8.0")?;

        let norm_out_weight = provider.load_tensor("model.diffusion_model.out.0.weight")?;
        let norm_out_bias = provider.load_tensor("model.diffusion_model.out.0.bias")?;
        let head_conv = load_conv_params(
            provider.as_ref(),
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.out.2.bias",
            1,
            1,
        )?;

        Ok(Self {
            prompt: resources.prompt_encoder,
            registry,
            infer_runtime,
            vae_spec: resources.vae_spec,
            model_device,
            time_fc1_weight,
            time_fc1_bias,
            time_fc2_weight,
            time_fc2_bias,
            conv_in,
            res_block1,
            res_block2,
            downsample,
            res_block4,
            up_res_block6,
            up_res_block7,
            up_res_block8,
            norm_out_weight,
            norm_out_bias,
            head_conv,
        })
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<PromptEmbeddings> {
        self.prompt.encode_single(prompt)
    }

    pub fn encode_pair(
        &self,
        prompt: &str,
        negative: &str,
    ) -> Result<(PromptEmbeddings, PromptEmbeddings)> {
        self.prompt.encode_pair(prompt, negative)
    }

    pub fn sample(
        &self,
        prompt: &str,
        negative_prompt: &str,
        steps: usize,
        guidance_scale: f32,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        let (cond, uncond) = self.encode_pair(prompt, negative_prompt)?;
        self.sample_with_embeddings(&cond, &uncond, steps, guidance_scale, height, width, seed)
    }

    pub fn decode_latents(&self, latents: &Tensor) -> Result<Tensor> {
        vae_decode(&self.vae_spec, &latents.to_dtype(DType::BF16)?, VaePolicy::GpuFirst)
    }

    pub fn make_time_ids(&self, batch_size: usize, height: usize, width: usize) -> Result<Tensor> {
        Ok(build_time_ids(
            batch_size,
            height as f32,
            width as f32,
            0.0,
            0.0,
            height as f32,
            width as f32,
            &self.model_device,
        )?)
    }

    pub fn denoise(
        &self,
        latents: &Tensor,
        timesteps: &Tensor,
        ctx: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Tensor> {
        let timesteps_f32 = timesteps.to_dtype(DType::F32)?;
        let time_emb = timestep_embedding(&timesteps_f32, 320, 10000.0, false, Some(1.0))?;
        let time_emb = linear(&time_emb, &self.time_fc1_weight, &self.time_fc1_bias)?.silu()?;
        let time_emb = linear(&time_emb, &self.time_fc2_weight, &self.time_fc2_bias)?;

        let mut h = latents.to_dtype(DType::BF16)?;
        h = conv2d_nhwc(&h, &self.conv_in)?;
        h = apply_resnet(&self.res_block1, &h, &time_emb)?;
        h = apply_resnet(&self.res_block2, &h, &time_emb)?;
        h = conv2d_nhwc(&h, &self.downsample)?;
        let lowres_skip = h.clone_result()?;

        h = apply_resnet(&self.res_block4, &h, &time_emb)?;

        let pooled_f32 = pooled.to_dtype(DType::F32)?;
        let time_ids_f32 = time_ids.to_dtype(DType::F32)?;
        let cond = self.registry.make_conditioning(&pooled_f32, timesteps, &time_ids_f32)?;
        let prefilled_skips = [lowres_skip.clone_result()?];
        let (mut h, mut skip_bank) = self.infer_runtime.forward(h, ctx, &cond, &prefilled_skips)?;

        let stage_debug = std::env::var_os("SDXL_STAGE_DEBUG").is_some();
        if stage_debug {
            eprintln!("[native] runtime out {:?} skip_bank={}", h.shape().dims(), skip_bank.len());
            for (i, res) in skip_bank.iter().enumerate() {
                eprintln!("    skip_bank[{i}] {:?}", res.shape().dims());
            }
        }

        let mut take_skip = |label: &str,
                             expected_channels: usize,
                             target_hw: usize,
                             fallback: Option<&Tensor>|
         -> Result<Tensor> {
            loop {
                let maybe_candidate = skip_bank.pop();
                if maybe_candidate.is_none() {
                    if let Some(extra) = fallback {
                        let mut skip = extra.clone_result()?;
                        while (skip.shape().dims()[1] as usize) < target_hw {
                            skip = upsample_nearest2x_nhwc(&skip)?;
                        }
                        let skip = if skip.dtype() == DType::BF16 {
                            skip
                        } else {
                            skip.to_dtype(DType::BF16)?
                        };
                        if stage_debug {
                            eprintln!(
                                "[native] {label} using fallback skip {:?}",
                                skip.shape().dims()
                            );
                        }
                        return Ok(skip);
                    }
                    return Err(anyhow!(
                        "decoder residual stack underflow before {label} (no fallback)"
                    ));
                }

                let candidate = maybe_candidate.unwrap();
                let dims = candidate.shape().dims().to_vec();
                if dims.len() == 4 {
                    let mut working = candidate.clone_result()?;
                    let mut converted = false;

                    while (working.shape().dims()[1] as usize) < target_hw {
                        working = upsample_nearest2x_nhwc(&working)?;
                        converted = true;
                    }

                    let mut work_dims = working.shape().dims().to_vec();
                    if work_dims[3] as usize == expected_channels * 2 {
                        let offset = work_dims[3] as usize - expected_channels;
                        working = working.narrow(3, offset, expected_channels)?.clone_result()?;
                        converted = true;
                        work_dims = working.shape().dims().to_vec();
                    }

                    if work_dims[3] as usize == expected_channels
                        && work_dims[1] as usize == target_hw
                    {
                        let skip = if working.dtype() == DType::BF16 {
                            working
                        } else {
                            working.to_dtype(DType::BF16)?
                        };
                        if stage_debug {
                            let note = if converted { "(converted)" } else { "" };
                            eprintln!("[native] {label} skip {:?} {note}", skip.shape().dims());
                        }
                        return Ok(skip);
                    }
                }
                if stage_debug {
                    eprintln!(
                        "[native] discarded residual {:?} while searching for {label} (expected C={})",
                        dims,
                        expected_channels
                    );
                }
            }
        };

        let target_hw = h.shape().dims()[1] as usize;
        let skip6 = take_skip("up_res_block6", 320, target_hw, None)?;
        let cat6 = Tensor::cat(&[&h, &skip6], 3)?;
        h = apply_resnet(&self.up_res_block6, &cat6, &time_emb)?;
        if stage_debug {
            eprintln!("[native] after up_res_block6 {:?}", h.shape().dims());
        }

        let target_hw = h.shape().dims()[1] as usize;
        let skip7 = take_skip("up_res_block7", 320, target_hw, None)?;
        let cat7 = Tensor::cat(&[&h, &skip7], 3)?;
        h = apply_resnet(&self.up_res_block7, &cat7, &time_emb)?;
        if stage_debug {
            eprintln!("[native] after up_res_block7 {:?}", h.shape().dims());
        }

        let target_hw = h.shape().dims()[1] as usize;
        let skip8 = take_skip("up_res_block8", 320, target_hw, Some(&lowres_skip))?;
        let cat8 = Tensor::cat(&[&h, &skip8], 3)?;
        h = apply_resnet(&self.up_res_block8, &cat8, &time_emb)?;
        if stage_debug {
            eprintln!(
                "[native] after up_res_block8 {:?} skip_bank={}",
                h.shape().dims(),
                skip_bank.len()
            );
        }

        let mut eps = group_norm_nhwc(&h, &self.norm_out_weight, &self.norm_out_bias, 32)?;
        eps = eps.silu()?;
        eps = conv2d_nhwc(&eps, &self.head_conv)?;

        let target_hw = latents.shape().dims()[1] as usize;
        let mut current_hw = eps.shape().dims()[1] as usize;
        while current_hw < target_hw {
            eps = upsample_nearest2x_nhwc(&eps)?;
            current_hw = eps.shape().dims()[1] as usize;
        }

        Ok(eps.to_dtype(DType::F32)?)
    }

    pub fn sample_with_embeddings(
        &self,
        cond: &PromptEmbeddings,
        uncond: &PromptEmbeddings,
        steps: usize,
        guidance_scale: f32,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        anyhow::ensure!(height % 8 == 0 && width % 8 == 0, "resolution must be divisible by 8");
        let batch = cond.context.shape().dims()[0] as usize;
        anyhow::ensure!(
            batch == uncond.context.shape().dims()[0] as usize,
            "batch mismatch between cond and uncond"
        );

        let latent_shape = Shape::from_dims(&[batch, height / 8, width / 8, 4]);
        let mut latents = randn_on(latent_shape, &self.model_device, DType::F32, seed)
            .map_err(|e| anyhow!("randn_on failed: {e}"))?;

        let scheduler = SchedulerCfg {
            steps,
            sigma_min: 0.029,
            sigma_max: 14.0,
            rho: 7.0,
            kind: ScheduleKind::Karras,
        };
        let sigmas = make_sigmas(&scheduler);
        anyhow::ensure!(sigmas.len() >= 2, "sigma schedule must contain at least two entries");

        let sigma0 = sigmas.get(0);
        latents = latents.mul_scalar(sigma0)?;

        let time_ids = self.make_time_ids(batch, height, width)?;

        for i in 0..(sigmas.len() - 1) {
            let sigma = sigmas.get(i);
            let sigma_next = sigmas.get(i + 1);
            let t = sigma_to_timestep(batch, sigma, &self.model_device)
                .map_err(|e| anyhow!("sigma_to_timestep failed: {e}"))?;

            let eps_cond = self.denoise(&latents, &t, &cond.context, &cond.pooled, &time_ids)?;
            let eps_uncond = if guidance_scale > 1.0 {
                self.denoise(&latents, &t, &uncond.context, &uncond.pooled, &time_ids)?
            } else {
                eps_cond.clone()
            };
            let eps = cfg_mix(&eps_uncond, &eps_cond, guidance_scale)?;

            let delta = sigma_next - sigma;
            latents = latents.add(&eps.mul_scalar(delta)?)?;
        }

        self.decode_latents(&latents)
    }
}

fn to_f32(t: Tensor) -> Result<Tensor> {
    if t.dtype() == DType::F32 {
        Ok(t)
    } else {
        Ok(t.to_dtype(DType::F32)?)
    }
}

fn load_conv_params(
    provider: &SdxlWeightProvider,
    weight_key: &str,
    bias_key: &str,
    stride: usize,
    padding: usize,
) -> Result<Conv2dParams> {
    let weight = to_f32(provider.load_tensor(weight_key)?)?;
    let bias = to_f32(provider.load_tensor(bias_key)?)?;
    Ok(Conv2dParams { weight, bias, stride, padding })
}

fn load_resnet_params(provider: &SdxlWeightProvider, prefix: &str) -> Result<ResnetParams> {
    let in_norm_weight = to_f32(provider.load_tensor(&format!("{prefix}.in_layers.0.weight"))?)?;
    let in_norm_bias = to_f32(provider.load_tensor(&format!("{prefix}.in_layers.0.bias"))?)?;
    let in_conv = load_conv_params(
        provider,
        &format!("{prefix}.in_layers.2.weight"),
        &format!("{prefix}.in_layers.2.bias"),
        1,
        1,
    )?;
    let emb_proj_weight = to_f32(provider.load_tensor(&format!("{prefix}.emb_layers.1.weight"))?)?;
    let emb_proj_bias = to_f32(provider.load_tensor(&format!("{prefix}.emb_layers.1.bias"))?)?;
    let out_norm_weight = to_f32(provider.load_tensor(&format!("{prefix}.out_layers.0.weight"))?)?;
    let out_norm_bias = to_f32(provider.load_tensor(&format!("{prefix}.out_layers.0.bias"))?)?;
    let out_conv = load_conv_params(
        provider,
        &format!("{prefix}.out_layers.3.weight"),
        &format!("{prefix}.out_layers.3.bias"),
        1,
        1,
    )?;
    let skip_conv = if provider.tensor_shape(&format!("{prefix}.skip_connection.weight")).is_ok() {
        Some(load_conv_params(
            provider,
            &format!("{prefix}.skip_connection.weight"),
            &format!("{prefix}.skip_connection.bias"),
            1,
            0,
        )?)
    } else {
        None
    };
    Ok(ResnetParams {
        in_norm_weight,
        in_norm_bias,
        in_conv,
        emb_proj_weight,
        emb_proj_bias,
        out_norm_weight,
        out_norm_bias,
        out_conv,
        skip_conv,
    })
}

fn conv2d_nhwc(input: &Tensor, params: &Conv2dParams) -> Result<Tensor> {
    let x_f32 = input.to_dtype(DType::F32)?;
    let x_nchw = GpuOps::permute_nhwc_to_nchw(&x_f32)?;
    let conv = x_nchw.conv2d(&params.weight, Some(&params.bias), params.stride, params.padding)?;
    let y = GpuOps::permute_nchw_to_nhwc(&conv)?;
    Ok(y.to_dtype(DType::BF16)?)
}

fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let w_t = weight.transpose_dims(0, 1)?;
    let mut y = x_f32.matmul(&w_t)?;
    y = y.add(&bias.to_dtype(DType::F32)?)?;
    Ok(y)
}

fn group_norm_nhwc(x: &Tensor, weight: &Tensor, bias: &Tensor, groups: usize) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_nchw = GpuOps::permute_nhwc_to_nchw(&x_f32)?;
    let y = group_norm(
        &x_nchw,
        groups,
        Some(&weight.to_dtype(DType::F32)?),
        Some(&bias.to_dtype(DType::F32)?),
        1e-6,
    )?;
    let y_nhwc = GpuOps::permute_nchw_to_nhwc(&y)?;
    Ok(y_nhwc.to_dtype(DType::BF16)?)
}

fn apply_resnet(params: &ResnetParams, x: &Tensor, t_emb: &Tensor) -> Result<Tensor> {
    let mut h = group_norm_nhwc(x, &params.in_norm_weight, &params.in_norm_bias, 32)?;
    h = h.silu()?;
    h = conv2d_nhwc(&h, &params.in_conv)?;

    let t_proj = linear(t_emb, &params.emb_proj_weight, &params.emb_proj_bias)?;
    let t_proj = t_proj.reshape(&[t_proj.shape().dims()[0], 1, 1, t_proj.shape().dims()[1]])?;
    h = h.to_dtype(DType::F32)?.add(&t_proj)?.to_dtype(DType::BF16)?;

    h = group_norm_nhwc(&h, &params.out_norm_weight, &params.out_norm_bias, 32)?;
    h = h.silu()?;
    h = conv2d_nhwc(&h, &params.out_conv)?;

    let skip = match &params.skip_conv {
        Some(skip_conv) => conv2d_nhwc(x, skip_conv)?,
        None => x.clone_result()?,
    };
    Ok(skip.add(&h)?)
}

fn upsample_nearest2x_nhwc(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(anyhow!("nearest upsample expects NHWC tensor, got {:?}", dims));
    }
    let (b, h, w, c) = (dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize);
    let reshaped = x.reshape(&[b, h, 1, w, 1, c])?;
    let target = Shape::from_dims(&[b, h, 2, w, 2, c]);
    let broadcast = reshaped.broadcast_to(&target)?;
    let view = broadcast.reshape(&[b, h * 2, w * 2, c])?;
    Ok(view.clone_result()?)
}

fn cfg_mix(uncond: &Tensor, cond: &Tensor, scale: f32) -> Result<Tensor> {
    let dims_uncond = uncond.shape().dims().to_vec();
    let dims_cond = cond.shape().dims().to_vec();
    anyhow::ensure!(dims_uncond.len() == 4 && dims_cond.len() == 4, "cfg_mix expects NHWC tensors");
    let target_hw = usize::max(dims_uncond[1] as usize, dims_cond[1] as usize);

    let align_tensor = |src: &Tensor| -> Result<Tensor> {
        let mut tensor = if src.dtype() == DType::BF16 {
            src.clone_result()?
        } else {
            src.to_dtype(DType::BF16)?
        };
        while (tensor.shape().dims()[1] as usize) < target_hw {
            tensor = upsample_nearest2x_nhwc(&tensor)?;
        }
        Ok(tensor.to_dtype(DType::F32)?)
    };

    let uncond_f32 = align_tensor(uncond)?;
    let cond_f32 = align_tensor(cond)?;

    let diff = cond_f32.sub(&uncond_f32)?;
    let scaled = diff.mul_scalar(scale)?;
    Ok(uncond_f32.add(&scaled)?)
}
