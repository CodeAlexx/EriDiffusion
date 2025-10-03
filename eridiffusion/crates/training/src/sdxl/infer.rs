#![cfg(feature = "sdxl")]

use std::{path::Path, sync::Arc};

use super::scheduler::{make_sigmas, sigma_to_timestep, ScheduleKind, SchedulerCfg};
use super::RuntimeMode;
use crate::conditioning::{time_ids::build_time_ids, timestep_embedding::timestep_embedding};
use crate::tensor_utils::broadcast_to_as;
use anyhow::{anyhow, Result};
use eridiffusion_common_text::{clip_l::ClipL, openclip_g::OpenClipG, HfTokenizer};
use eridiffusion_common_vae::{decode as vae_decode, VaeKind, VaePolicy, VaeSpec};
use eridiffusion_common_weights::strict_loader::StrictMmapLoader;
use eridiffusion_core::Device as EDevice;
use eridiffusion_models::devtensor::{randn_on, zeros_like_on};
use flame_core::{group_norm, DType, Device as FDevice, Shape, Tensor};

use super::{
    data_loader::SdxlBatch, inference_runtime::SdxlInferenceRuntime, registry::SdxlLayerRegistry,
    weights::SdxlWeightProvider,
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

fn to_f32(t: Tensor) -> anyhow::Result<Tensor> {
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
) -> anyhow::Result<Conv2dParams> {
    let weight = to_f32(provider.load_tensor(weight_key)?)?;
    let bias = to_f32(provider.load_tensor(bias_key)?)?;
    Ok(Conv2dParams { weight, bias, stride, padding })
}

fn load_resnet_params(provider: &SdxlWeightProvider, prefix: &str) -> anyhow::Result<ResnetParams> {
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

fn conv2d_nhwc(input: &Tensor, params: &Conv2dParams) -> anyhow::Result<Tensor> {
    let x_f32 = input.to_dtype(DType::F32)?;
    let x_nchw = x_f32.permute(&[0, 3, 1, 2])?;
    let mut y = x_nchw.conv2d(&params.weight, Some(&params.bias), params.stride, params.padding)?;
    y = y.permute(&[0, 2, 3, 1])?;
    Ok(y.to_dtype(DType::BF16)?)
}

fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> anyhow::Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let w_t = weight.transpose_dims(0, 1)?;
    let mut y = x_f32.matmul(&w_t)?;
    y = y.add(&bias.to_dtype(DType::F32)?)?;
    Ok(y)
}

fn group_norm_nhwc(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    groups: usize,
) -> anyhow::Result<Tensor> {
    if std::env::var_os("SDXL_TRACE_VERBOSE").is_some() {
        eprintln!(
            "[group_norm_nhwc] x {:?} weight {:?} bias {:?}",
            x.shape().dims(),
            weight.shape().dims(),
            bias.shape().dims()
        );
    }
    let x_nchw = x.to_dtype(DType::F32)?.permute(&[0, 3, 1, 2])?;
    let y = group_norm(
        &x_nchw,
        groups,
        Some(&weight.to_dtype(DType::F32)?),
        Some(&bias.to_dtype(DType::F32)?),
        1e-6,
    )?;
    Ok(y.permute(&[0, 2, 3, 1])?.to_dtype(DType::BF16)?)
}

fn apply_resnet(params: &ResnetParams, x: &Tensor, t_emb: &Tensor) -> anyhow::Result<Tensor> {
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

/// Configuration for the SDXL inference pipeline.
#[derive(Clone, Debug)]
pub struct SdxlInferConfig {
    pub unet_path: String,
    pub vae_path: Option<String>,
    pub clip_l_path: String,
    pub clip_g_path: String,
    pub tokenizer_path: String,
    pub seq_len: usize,
    pub device: EDevice,
    pub dtype: DType,
    pub schedule: SchedulerCfg,
    pub attn_chunk: Option<usize>,
    pub kv_chunk: Option<usize>,
    pub kernel_telemetry: bool,
    pub unet_tile: Option<usize>,
    pub vae_tile: Option<usize>,
    pub mode: RuntimeMode,
}

impl Default for SdxlInferConfig {
    fn default() -> Self {
        Self {
            unet_path: String::new(),
            vae_path: None,
            clip_l_path: String::new(),
            clip_g_path: String::new(),
            tokenizer_path: String::new(),
            seq_len: 77,
            device: EDevice::Cuda(0),
            dtype: DType::BF16,
            schedule: SchedulerCfg {
                steps: 30,
                sigma_min: 0.029,
                sigma_max: 14.0,
                rho: 7.0,
                kind: ScheduleKind::Karras,
            },
            attn_chunk: None,
            kv_chunk: None,
            kernel_telemetry: false,
            unet_tile: None,
            vae_tile: None,
            mode: RuntimeMode::Resident,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerMode {
    Euler,
    Heun,
}

pub struct SdxlInferencePipeline {
    registry: Arc<SdxlLayerRegistry>,
    infer_runtime: SdxlInferenceRuntime,
    vae_spec: VaeSpec,
    tokenizer: HfTokenizer,
    clip_l: ClipL,
    clip_g: OpenClipG,
    flame_device: FDevice,
    device: EDevice,
    dtype: DType,
    schedule: SchedulerCfg,
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
    unet_tile: Option<usize>,
    vae_tile: Option<usize>,
}

impl SdxlInferencePipeline {
    pub fn new(cfg: SdxlInferConfig) -> Result<Self> {
        let ordinal = match cfg.device {
            EDevice::Cuda(ix) => ix,
            _ => return Err(anyhow!("SDXL inference currently supports CUDA devices only")),
        };
        let flame_device = FDevice::cuda(ordinal)?;
        let mmap = Arc::new(StrictMmapLoader::open(Path::new(&cfg.unet_path))?);
        let provider = Arc::new(SdxlWeightProvider::new(mmap, flame_device.clone()));
        let registry = Arc::new(SdxlLayerRegistry::build(provider.clone(), cfg.mode)?);
        let infer_runtime = SdxlInferenceRuntime::new(
            registry.clone(),
            cfg.attn_chunk,
            cfg.kv_chunk,
            cfg.kernel_telemetry,
        )?;

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

        let norm_out_weight = to_f32(provider.load_tensor("model.diffusion_model.out.0.weight")?)?;
        let norm_out_bias = to_f32(provider.load_tensor("model.diffusion_model.out.0.bias")?)?;
        let head_conv = load_conv_params(
            provider.as_ref(),
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.out.2.bias",
            1,
            1,
        )?;

        if cfg.clip_l_path.is_empty() || cfg.clip_g_path.is_empty() || cfg.tokenizer_path.is_empty()
        {
            return Err(anyhow!("clip_l_path, clip_g_path, and tokenizer_path must be provided"));
        }
        let tokenizer = HfTokenizer::from_path(&cfg.tokenizer_path, cfg.seq_len)?;
        let clip_l = ClipL::from_weights_auto(&cfg.clip_l_path, &flame_device, cfg.seq_len)?;
        let clip_g = OpenClipG::from_weights_auto(&cfg.clip_g_path, &flame_device, cfg.seq_len)?;

        let vae_path =
            cfg.vae_path.clone().ok_or_else(|| anyhow!("SDXL VAE path must be provided"))?;
        if !Path::new(&vae_path).exists() {
            return Err(anyhow!("SDXL VAE weights not found at {}", vae_path));
        }
        let vae_spec = VaeSpec {
            kind: VaeKind::Sdxl,
            path: vae_path,
            latent_div: 8,
            latent_channels: 4,
            latent_scale: 0.18215,
        };

        Ok(Self {
            registry,
            infer_runtime,
            vae_spec,
            tokenizer,
            clip_l,
            clip_g,
            flame_device,
            device: cfg.device,
            dtype: cfg.dtype,
            schedule: cfg.schedule,
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
            unet_tile: cfg.unet_tile,
            vae_tile: cfg.vae_tile,
        })
    }

    /// Sample a batch of images using embeddings provided by the manifest loader.
    pub fn sample_batch(
        &self,
        batch: &SdxlBatch,
        steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        let nhwc_shape = self.latent_shape(batch)?;
        let cond_ctx = Tensor::cat(
            &[&batch.clip_l.to_dtype(DType::BF16)?, &batch.clip_g.to_dtype(DType::BF16)?],
            2,
        )?;
        let uncond_ctx = zeros_like_on(&cond_ctx, &self.device, Some(DType::BF16))?;
        let cond_pooled = batch.pooled.to_dtype(DType::BF16)?;
        let uncond_pooled = zeros_like_on(&cond_pooled, &self.device, Some(DType::BF16))?;
        self.run_sampler(
            nhwc_shape,
            &cond_ctx,
            &uncond_ctx,
            &cond_pooled,
            &uncond_pooled,
            &batch.time_ids,
            steps,
            guidance_scale,
            seed,
            SamplerMode::Heun,
        )
    }

    /// Sample directly from prompts (text-to-image).
    pub fn sample_prompts(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        steps: usize,
        guidance_scale: f32,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        let negative = negative_prompt.unwrap_or("");
        let cond = self.encode_prompt(prompt)?;
        let uncond = self.encode_prompt(negative)?;
        let latents_shape = vec![cond.batch_size, height / 8, width / 8, 4];
        let time_ids = build_time_ids(
            cond.batch_size,
            height as f32,
            width as f32,
            0.0,
            0.0,
            height as f32,
            width as f32,
            &self.device,
        )?;
        self.run_sampler(
            latents_shape,
            &cond.context,
            &uncond.context,
            &cond.pooled,
            &uncond.pooled,
            &time_ids,
            steps,
            guidance_scale,
            seed,
            SamplerMode::Heun,
        )
    }

    /// Sample with explicit sampler mode control (Euler vs Heun).
    pub fn sample_prompts_with_mode(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        steps: usize,
        guidance_scale: f32,
        height: usize,
        width: usize,
        seed: Option<u64>,
        mode: SamplerMode,
    ) -> Result<Tensor> {
        let negative = negative_prompt.unwrap_or("");
        let cond = self.encode_prompt(prompt)?;
        let uncond = self.encode_prompt(negative)?;
        let latents_shape = vec![cond.batch_size, height / 8, width / 8, 4];
        let time_ids = build_time_ids(
            cond.batch_size,
            height as f32,
            width as f32,
            0.0,
            0.0,
            height as f32,
            width as f32,
            &self.device,
        )?;
        self.run_sampler(
            latents_shape,
            &cond.context,
            &uncond.context,
            &cond.pooled,
            &uncond.pooled,
            &time_ids,
            steps,
            guidance_scale,
            seed,
            mode,
        )
    }

    fn run_sampler(
        &self,
        latent_shape: Vec<usize>,
        cond_ctx: &Tensor,
        uncond_ctx: &Tensor,
        cond_pooled: &Tensor,
        uncond_pooled: &Tensor,
        time_ids: &Tensor,
        steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
        mode: SamplerMode,
    ) -> Result<Tensor> {
        let mut latents =
            randn_on(Shape::from_dims(&latent_shape), &self.device, DType::F32, seed)?;
        let schedule = SchedulerCfg { steps, ..self.schedule };
        let sigmas = make_sigmas(&schedule);
        if sigmas.len() < 2 {
            return Err(anyhow!("sigma schedule must contain at least two entries"));
        }
        let sigma0 = sigmas.get(0);
        latents = latents.mul_scalar(sigma0)?;

        let cond_ctx = cond_ctx.to_dtype(DType::BF16)?;
        let uncond_ctx = uncond_ctx.to_dtype(DType::BF16)?;
        let cond_pooled = cond_pooled.to_dtype(DType::BF16)?;
        let uncond_pooled = uncond_pooled.to_dtype(DType::BF16)?;
        let time_ids = time_ids.to_dtype(DType::F32)?;

        let mut latents_f32 = latents;
        let batch_size = cond_ctx.shape().dims()[0] as usize;
        for i in 0..(sigmas.len() - 1) {
            let sigma = sigmas.get(i);
            let next_sigma = sigmas.get(i + 1);
            let t = sigma_to_timestep(batch_size, sigma, &self.device)?;
            let eps_cond = self.denoise(&latents_f32, &t, &cond_ctx, &cond_pooled, &time_ids)?;
            let eps_uncond = if guidance_scale > 1.0 {
                self.denoise(&latents_f32, &t, &uncond_ctx, &uncond_pooled, &time_ids)?
            } else {
                eps_cond.clone()
            };
            let eps = cfg_mix(&eps_uncond, &eps_cond, guidance_scale)?;
            let delta = next_sigma - sigma;
            match mode {
                SamplerMode::Euler => {
                    latents_f32 = latents_f32.add(&eps.mul_scalar(delta)?)?;
                }
                SamplerMode::Heun => {
                    let latents_euler = latents_f32.clone().add(&eps.mul_scalar(delta)?)?;
                    let t_next = sigma_to_timestep(batch_size, next_sigma, &self.device)?;
                    let eps_cond_1 =
                        self.denoise(&latents_euler, &t_next, &cond_ctx, &cond_pooled, &time_ids)?;
                    let eps_uncond_1 = if guidance_scale > 1.0 {
                        self.denoise(&latents_euler, &t_next, &uncond_ctx, &uncond_pooled, &time_ids)?
                    } else {
                        eps_cond_1.clone()
                    };
                    let eps1 = cfg_mix(&eps_uncond_1, &eps_cond_1, guidance_scale)?;
                    let avg = eps.mul_scalar(0.5)?.add(&eps1.mul_scalar(0.5)?)?;
                    latents_f32 = latents_f32.add(&avg.mul_scalar(delta)?)?;
                }
            }
        }

        let latents_bf16 = latents_f32.to_dtype(DType::BF16)?;
        vae_decode(&self.vae_spec, &latents_bf16, VaePolicy::GpuFirst)
    }

    fn encode_prompt(&self, prompt: &str) -> Result<PromptConditioning> {
        let texts = vec![prompt.to_string()];
        let (ids, _, _) = self.tokenizer.encode_batch_on(&texts, &self.flame_device)?;
        let clip_l_ctx = self.clip_l.forward(&ids)?;
        let clip_g_ctx = self.clip_g.forward(&ids)?;
        let pooled = self.clip_g.pooled(&clip_g_ctx)?;
        let context = Tensor::cat(&[&clip_l_ctx, &clip_g_ctx], 2)?;
        Ok(PromptConditioning { batch_size: texts.len(), context, pooled })
    }

    fn denoise(
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
        if std::env::var_os("SDXL_STAGE_DEBUG").is_some() {
            eprintln!("[denoise] lowres_skip {:?}", lowres_skip.shape().dims());
        }
        h = apply_resnet(&self.res_block4, &h, &time_emb)?;
        if std::env::var_os("SDXL_STAGE_DEBUG").is_some() {
            eprintln!("[denoise] after res_block4 {:?}", h.shape().dims());
        }

        let cond = self.registry.make_conditioning(pooled, timesteps, time_ids)?;
        if std::env::var_os("SDXL_STAGE_DEBUG").is_some() {
            let has_transition = self.registry.transition_for(4).is_some();
            eprintln!("[denoise] transition idx4 present={}", has_transition);
        }
        let prefilled_skips = [lowres_skip.clone_result()?];
        let (mut h, mut skip_bank) =
            self.infer_runtime.forward(h, ctx, &cond, &prefilled_skips)?;
        let stage_debug = matches!(std::env::var("SDXL_STAGE_DEBUG").as_deref(), Ok("1"));
        if stage_debug {
            eprintln!("[denoise] runtime out {:?} skip_bank={}", h.shape().dims(), skip_bank.len());
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
                let candidate = skip_bank
                    .pop()
                    .ok_or_else(|| anyhow!("decoder residual stack underflow before {label}"))?;
                let dims = candidate.shape().dims().to_vec();
                if dims.len() == 4 {
                    let mut working = candidate.clone_result()?;
                    let mut converted = false;

                    // Upsample until spatial dims match target if needed.
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
                            eprintln!("[denoise] {label} skip {:?} {note}", skip.shape().dims());
                        }
                        return Ok(skip);
                    }
                }
                if stage_debug {
                    eprintln!(
                        "[denoise] discarded residual {:?} while searching for {label} (expected C={})",
                        dims,
                        expected_channels
                    );
                }
                if skip_bank.is_empty() {
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
                                "[denoise] {label} using fallback skip {:?}",
                                skip.shape().dims()
                            );
                        }
                        return Ok(skip);
                    }
                }
            }
        };

        let target_hw = h.shape().dims()[1] as usize;
        let skip6 = take_skip("up_res_block6", 320, target_hw, None)?;
        let cat6 = Tensor::cat(&[&h, &skip6], 3)?;
        h = apply_resnet(&self.up_res_block6, &cat6, &time_emb)?;
        if stage_debug {
            eprintln!("[denoise] after up_res_block6 {:?}", h.shape().dims());
        }

        let target_hw = h.shape().dims()[1] as usize;
        let skip7 = take_skip("up_res_block7", 320, target_hw, None)?;
        let cat7 = Tensor::cat(&[&h, &skip7], 3)?;
        h = apply_resnet(&self.up_res_block7, &cat7, &time_emb)?;
        if stage_debug {
            eprintln!("[denoise] after up_res_block7 {:?}", h.shape().dims());
        }

        let target_hw = h.shape().dims()[1] as usize;
        let skip8 = take_skip("up_res_block8", 320, target_hw, Some(&lowres_skip))?;
        let cat8 = Tensor::cat(&[&h, &skip8], 3)?;
        h = apply_resnet(&self.up_res_block8, &cat8, &time_emb)?;
        if stage_debug {
            eprintln!(
                "[denoise] after up_res_block8 {:?} skip_bank={}",
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

        eps.to_dtype(DType::F32).map_err(Into::into)
    }

    fn latent_shape(&self, batch: &SdxlBatch) -> Result<Vec<usize>> {
        let dims = batch.latents.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(anyhow!("expected latents to be [B,4,H/8,W/8], got {:?}", dims));
        }
        Ok(vec![dims[0] as usize, dims[2] as usize, dims[3] as usize, dims[1] as usize])
    }
}

fn cfg_mix(uncond: &Tensor, cond: &Tensor, scale: f32) -> Result<Tensor> {
    let dims_uncond = uncond.shape().dims().to_vec();
    let dims_cond = cond.shape().dims().to_vec();
    anyhow::ensure!(dims_uncond.len() == 4 && dims_cond.len() == 4, "cfg_mix expects NHWC tensors");
    let target_hw = usize::max(dims_uncond[1] as usize, dims_cond[1] as usize);

    let align_tensor = |label: &str, src: &Tensor| -> Result<Tensor> {
        let mut tensor = if src.dtype() == DType::BF16 {
            src.clone_result()?
        } else {
            src.to_dtype(DType::BF16)?
        };
        while (tensor.shape().dims()[1] as usize) < target_hw {
            tensor = upsample_nearest2x_nhwc(&tensor)?;
        }
        if std::env::var_os("SDXL_STAGE_DEBUG").is_some() {
            let dims = tensor.shape().dims().to_vec();
            eprintln!("[cfg_mix] {label} aligned to {:?}", dims);
        }
        Ok(tensor.to_dtype(DType::F32)?)
    };

    let uncond_f32 = align_tensor("uncond", uncond)?;
    let cond_f32 = align_tensor("cond", cond)?;

    let diff = cond_f32.sub(&uncond_f32)?;
    let scaled = diff.mul_scalar(scale)?;
    uncond_f32.add(&scaled).map_err(Into::into)
}

fn upsample_nearest2x_nhwc(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(anyhow!("nearest upsample expects NHWC tensor, got {:?}", dims));
    }
    let (b, h, w, c) = (dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize);
    let reshaped = x.reshape(&[b, h, 1, w, 1, c])?;
    let broadcast = broadcast_to_as(&reshaped, &[b, h, 2, w, 2, c], x.dtype())?;
    let view = broadcast.reshape(&[b, h * 2, w * 2, c])?;
    Ok(view.clone_result()?)
}

struct PromptConditioning {
    batch_size: usize,
    context: Tensor,
    pooled: Tensor,
}
