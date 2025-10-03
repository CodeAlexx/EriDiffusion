//! Flux-specific training implementation (LoRA adapters only, no stubs).

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, ensure, Result};
use chrono::Utc;
use eridiffusion_core::{Device, FluxVariant};
use eridiffusion_models::{
    devtensor::{randn_on, tensor_from_slice_on, zeros_on},
    flux::{ae::FluxAE, text::FluxText},
};
use flame_core::{device::Device as FlameDevice, DType, Shape, Tensor};
use safetensors::Dtype as SafeDtype;
use serde_json;

use crate::{
    checkpoint_safetensors::{
        bf16_bytes_to_f32_vec, deserialize_tensors, f32_bytes_to_vec, serialize_tensors,
        tensor_to_bf16_bytes, tensor_to_f32_bytes, OwnedTensorView,
    },
    flame_ctx,
    flux::{
        data::FluxBatch,
        lora::{FluxBlockLora, FluxLoraHandles, FluxLoraLinear},
        registry::{build_layer_registry, FluxRegistry, FluxRegistryPlan},
        runtime::{BlockMetadata, LayerRegistry},
        weights::FluxWeightProvider,
    },
    gradient_accumulator::GradientAccumulator,
    lora_keys::LoraSpec as TrainerLoraSpec,
    optimizer::{
        clip_grads_global_norm_fp32_tensors, create_optimizer, Optimizer, OptimizerConfig,
    },
    policy,
    tensor_utils::mean_keepdim_fp32,
};

/// Flux training configuration (LoRA-only).
#[derive(Debug, Clone)]
pub struct FluxTrainingConfig {
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub gradient_accumulation_steps: usize,
    pub gradient_checkpointing: bool,
    pub guidance_scale: f32,
    pub text_drop_prob: f32,
    pub ema_decay: f32,
    pub mixed_precision: bool,
    pub max_grad_norm: f64,
    pub seed: u64,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub lora_zero_init: bool,
    pub sigma_min: f32,
    pub sigma_max: f32,
}

impl Default for FluxTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            warmup_steps: 1000,
            max_steps: 100_000,
            gradient_accumulation_steps: 4,
            gradient_checkpointing: false,
            guidance_scale: 3.5,
            text_drop_prob: 0.1,
            ema_decay: 0.9999,
            mixed_precision: false,
            max_grad_norm: 1.0,
            seed: 1337,
            lora_rank: 8,
            lora_alpha: 8.0,
            lora_zero_init: true,
            sigma_min: 0.01,
            sigma_max: 50.0,
        }
    }
}

pub struct FluxTrainer {
    registry: LayerRegistry,
    #[allow(dead_code)]
    flux_registry: FluxRegistry,
    #[allow(dead_code)]
    metadata: Vec<BlockMetadata>,
    lora: FluxLoraHandles,
    #[allow(dead_code)]
    lora_params: Vec<flame_core::Parameter>,
    param_views: Vec<Tensor>,
    param_names: Vec<String>,
    ae: FluxAE,
    text: FluxText,
    img_proj: Tensor,
    text_proj: Tensor,
    img_head: Tensor,
    cond_proj_img: Tensor,
    cond_proj_txt: Tensor,
    optimizer: Box<dyn Optimizer>,
    gradient_accumulator: GradientAccumulator,
    config: FluxTrainingConfig,
    device: Device,
    hidden: usize,
    text_hidden_in: usize,
    #[allow(dead_code)]
    image_cond_dim: usize,
    #[allow(dead_code)]
    text_cond_dim: usize,
    last_grad_norm: f32,
    step_counter: u64,
    last_lora_bytes: usize,
    ema_shadow: Vec<Tensor>,
    ema_enabled: bool,
}

impl FluxTrainer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        registry: LayerRegistry,
        flux_registry: FluxRegistry,
        metadata: Vec<BlockMetadata>,
        lora: FluxLoraHandles,
        lora_params: Vec<flame_core::Parameter>,
        param_views: Vec<Tensor>,
        param_names: Vec<String>,
        ae: FluxAE,
        text: FluxText,
        optimizer: Box<dyn Optimizer>,
        gradient_accumulator: GradientAccumulator,
        config: FluxTrainingConfig,
        device: Device,
        hidden: usize,
        text_hidden_in: usize,
        image_cond_dim: usize,
        text_cond_dim: usize,
    ) -> Result<Self> {
        let (img_proj, text_proj, img_head, cond_proj_img, cond_proj_txt) =
            Self::create_projections(
                &device,
                hidden,
                text_hidden_in,
                image_cond_dim,
                text_cond_dim,
            )?;

        let ema_enabled =
            config.ema_decay.is_finite() && config.ema_decay > 0.0 && config.ema_decay < 1.0;
        let mut ema_shadow = Vec::new();
        if ema_enabled {
            for param in &lora_params {
                let initial = param.tensor()?.to_dtype(DType::F32)?;
                ema_shadow.push(initial);
            }
        }

        Ok(Self {
            registry,
            flux_registry,
            metadata,
            lora,
            lora_params,
            param_views,
            param_names,
            ae,
            text,
            img_proj,
            text_proj,
            img_head,
            cond_proj_img,
            cond_proj_txt,
            optimizer,
            gradient_accumulator,
            config,
            device,
            hidden,
            text_hidden_in,
            image_cond_dim,
            text_cond_dim,
            last_grad_norm: 0.0,
            step_counter: 0,
            last_lora_bytes: 0,
            ema_shadow,
            ema_enabled,
        })
    }

    fn create_projections(
        device: &Device,
        hidden: usize,
        text_hidden: usize,
        image_cond: usize,
        text_cond: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let scale = 1e-3f32;
        let img_proj = randn_on(Shape::from_dims(&[4, hidden]), device, DType::F32, None)?
            .mul_scalar(scale)?;
        let text_proj =
            randn_on(Shape::from_dims(&[text_hidden, hidden]), device, DType::F32, None)?
                .mul_scalar(scale)?;
        let img_head = zeros_on(Shape::from_dims(&[hidden, 4]), device, DType::F32)?;
        let cond_in = text_hidden + 1;
        let cond_proj_img =
            randn_on(Shape::from_dims(&[cond_in, image_cond]), device, DType::F32, None)?
                .mul_scalar(scale)?;
        let cond_proj_txt =
            randn_on(Shape::from_dims(&[cond_in, text_cond]), device, DType::F32, None)?
                .mul_scalar(scale)?;
        Ok((img_proj, text_proj, img_head, cond_proj_img, cond_proj_txt))
    }

    pub fn grad_norm(&self) -> f32 {
        self.last_grad_norm
    }

    pub fn lora_param_bytes(&self) -> usize {
        self.last_lora_bytes
    }

    pub fn step(&self) -> u64 {
        self.step_counter
    }

    fn param_refs(&self) -> Vec<Tensor> {
        self.param_views.iter().cloned().collect()
    }

    fn compute_lora_bytes(&self) -> Result<usize> {
        let mut total = 0usize;
        for p in &self.param_views {
            let elem = p.shape().elem_count();
            let bytes = match p.dtype() {
                DType::BF16 | DType::F16 => 2,
                DType::F32 | DType::I32 => 4,
                DType::F64 | DType::I64 => 8,
                _ => 4,
            };
            total += elem * bytes;
        }
        Ok(total)
    }

    fn pooled_text_and_time(&self, text_tokens: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        let b = text_tokens.shape().dims()[0];
        let pooled = flame_ctx!(
            mean_keepdim_fp32(text_tokens, 1),
            "flux_trainer::pooled_text_and_time.mean"
        )?;
        let dtype = text_tokens.dtype();
        let pooled = pooled.to_dtype(dtype)?;
        let pooled = pooled.reshape(&[b, self.text_hidden_in])?;
        let time = timesteps.to_dtype(dtype)?;
        let cond_in = Tensor::cat(&[&pooled, &time], 1)?; // [B, text_hidden+1]
        Ok(cond_in)
    }

    pub async fn train_step(
        &mut self,
        images: &Tensor,
        captions: &[String],
        _negative_prompts: &[String],
    ) -> Result<f32> {
        let z_nhwc = self.ae.encode_latents(images)?;
        let z_nchw = z_nhwc.permute(&[0, 3, 1, 2])?;
        ensure!(
            z_nchw.shape().dims().len() == 4 && z_nchw.shape().dims()[1] == 4,
            "AE latents must be [B,4,H,W], got {:?}",
            z_nchw.shape().dims()
        );
        let captions_vec: Vec<String> = captions.iter().cloned().collect();
        let text_ctx = self.text.encode(&captions_vec)?;
        self.train_step_from_encodings(z_nchw, text_ctx)
    }

    /// Train directly from cached latents + text embeddings (no AE/Text encode).
    pub fn train_step_precomputed(&mut self, batch: &FluxBatch) -> Result<f32> {
        self.train_step_from_encodings(batch.latents.clone(), batch.text_ctx.clone())
    }

    fn train_step_from_encodings(&mut self, latents_nchw: Tensor, text_ctx: Tensor) -> Result<f32> {
        let lat_dims = latents_nchw.shape().dims().to_vec();
        ensure!(
            lat_dims.len() == 4 && lat_dims[1] == 4,
            "Latents must be [B,4,H,W], got {:?}",
            lat_dims
        );
        let b = lat_dims[0];
        let h_lat = lat_dims[2];
        let w_lat = lat_dims[3];
        let seq_img = h_lat * w_lat;

        let z0_f32 = latents_nchw.to_dtype(DType::F32)?;
        let eps_target =
            randn_on(Shape::from_dims(&[b, 4, h_lat, w_lat]), &self.device, DType::F32, None)?;
        let t_b = policy::sample_timesteps01(b, &self.device)?; // [B,1]
        let mut sigma_b1 =
            policy::sigma_for_bounded(&t_b, self.config.sigma_min, self.config.sigma_max)?;
        let clamp_min = Tensor::from_scalar(1e-3f32, sigma_b1.device().clone())?;
        let clamp_max = Tensor::from_scalar(1.0f32, sigma_b1.device().clone())?;
        sigma_b1 = sigma_b1.maximum(&clamp_min)?.minimum(&clamp_max)?;
        let sigma_exp = sigma_b1.to_dtype(DType::F32)?.reshape(&[b, 1, 1, 1])?;
        let xt_f32 = z0_f32.add(&eps_target.mul(&sigma_exp)?)?;
        let xt_nchw = xt_f32.to_dtype(DType::BF16)?;
        let eps_target = eps_target.to_dtype(DType::BF16)?;

        // Image tokens: [B,T_img,D]
        let latents_bhwc = xt_nchw.permute(&[0, 2, 3, 1])?;
        let latents_tokens = latents_bhwc.reshape(&[b, seq_img, 4])?;
        let tokens_flat = latents_tokens.reshape(&[b * seq_img, 4])?;
        let proj =
            tokens_flat.matmul(&self.img_proj.to_dtype(DType::F32)?.transpose_dims(0, 1)?)?;
        let img_tokens = proj.reshape(&[b, seq_img, self.hidden])?;

        // Text tokens (assume padded length is dims[1])
        let dims = text_ctx.shape().dims().to_vec();
        ensure!(dims.len() == 3, "Text ctx rank {:?}", dims);
        let (bt, text_len, hidden_in) = (dims[0], dims[1], dims[2]);
        ensure!(bt == b, "Batch mismatch: {} vs {}", bt, b);
        ensure!(
            hidden_in == self.text_hidden_in,
            "Unexpected text hidden dim: {} vs {}",
            hidden_in,
            self.text_hidden_in
        );
        let text_flat = text_ctx.reshape(&[b * text_len, self.text_hidden_in])?;
        let text_proj =
            text_flat.matmul(&self.text_proj.to_dtype(DType::F32)?.transpose_dims(0, 1)?)?;
        let text_tokens = text_proj.reshape(&[b, text_len, self.hidden])?;

        // Conditioning
        let cond_input = self.pooled_text_and_time(&text_tokens, &t_b)?;
        let img_cond = cond_input.matmul(&self.cond_proj_img.to_dtype(DType::F32)?)?;
        let txt_cond = cond_input.matmul(&self.cond_proj_txt.to_dtype(DType::F32)?)?;

        let (img_out, _txt_out) = self.registry.forward(
            &img_tokens,
            &img_cond,
            Some(self.lora.blocks()),
            &text_tokens,
            &txt_cond,
        )?;

        let pred_flat = img_out
            .reshape(&[b * seq_img, self.hidden])?
            .matmul(&self.img_head.to_dtype(DType::F32)?.transpose_dims(0, 1)?)?;
        let pred_eps_tokens = pred_flat.reshape(&[b, seq_img, 4])?;
        let pred_eps = pred_eps_tokens
            .reshape(&[b, h_lat, w_lat, 4])?
            .permute(&[0, 3, 1, 2])?
            .to_dtype(DType::BF16)?;

        let eps_target_bf = eps_target.to_dtype(DType::BF16)?;
        let loss = self.compute_loss(&pred_eps, &eps_target_bf)?;
        let loss_scalar = tensor_scalar_f32(&loss)?;

        let param_refs_storage = self.param_refs();
        let param_refs: Vec<&Tensor> = param_refs_storage.iter().collect();
        self.gradient_accumulator.accumulate(&loss, &param_refs)?;

        if self.gradient_accumulator.should_step() {
            let param_refs_storage = self.param_refs();
            let param_refs: Vec<&Tensor> = param_refs_storage.iter().collect();
            let mut grads = self.gradient_accumulator.get_gradients(&param_refs)?;
            let mut sum_sq = 0.0f32;
            for g in &grads {
                let squared = g.mul(&g)?;
                let sum_val = squared.sum()?;
                sum_sq += tensor_scalar_f32(&sum_val)?;
            }
            self.last_grad_norm = sum_sq.sqrt();
            if self.config.max_grad_norm > 0.0 {
                let _ = clip_grads_global_norm_fp32_tensors(
                    &mut grads,
                    self.config.max_grad_norm as f32,
                )?;
            }
            let param_refs_step_storage = self.param_refs();
            let param_refs_step: Vec<&Tensor> = param_refs_step_storage.iter().collect();
            self.optimizer.step(&param_refs_step, &grads, self.config.learning_rate)?;
            self.gradient_accumulator.reset()?;
            self.step_counter = self.step_counter.saturating_add(1);
            self.last_lora_bytes = self.compute_lora_bytes()?;
            self.update_ema()?;
        }

        Ok(loss_scalar)
    }

    fn compute_loss(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let pred32 = pred.to_dtype(DType::F32)?;
        let target32 = target.to_dtype(DType::F32)?;
        let diff = pred32.sub(&target32)?;
        let sq = diff.mul(&diff)?;
        flame_ctx!(policy::reduce_mean_fp32_keepdim(&sq), "flux_trainer::compute_loss")
    }

    pub fn save_checkpoint(&mut self, root: &str, step: u64, seed: u64) -> Result<String> {
        let dir = Path::new(root).join(format!("step_{:06}", step));
        fs::create_dir_all(&dir)?;
        let lora_file = dir.join("lora.safetensors");
        self.write_lora_safetensor(&lora_file, step, seed)?;
        if self.ema_enabled {
            let ema_file = dir.join("ema.safetensors");
            self.write_ema_safetensor(&ema_file, step, seed)?;
        }
        let meta = dir.join("meta.json");
        let meta_payload = serde_json::json!({
            "step": step,
            "seed": seed,
            "time": Utc::now().to_rfc3339(),
            "param_names": self.param_names.clone(),
            "lora_file": "lora.safetensors",
            "ema_file": if self.ema_enabled { Some("ema.safetensors") } else { None },
        });
        fs::write(meta, serde_json::to_string_pretty(&meta_payload)?)?;
        if let Ok(meta) = fs::metadata(&lora_file) {
            self.last_lora_bytes = meta.len() as usize;
        }
        Ok(dir.display().to_string())
    }

    pub fn load_latest_checkpoint(&mut self, root: &str) -> Result<Option<(u64, u64)>> {
        let root_path = Path::new(root);
        if !root_path.exists() {
            return Ok(None);
        }
        let mut latest: Option<(u64, PathBuf)> = None;
        for entry in fs::read_dir(root_path)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().into_string().unwrap_or_default();
            if !name.starts_with("step_") {
                continue;
            }
            if let Ok(step) = name.trim_start_matches("step_").parse::<u64>() {
                if latest.as_ref().map(|(s, _)| step > *s).unwrap_or(true) {
                    latest = Some((step, entry.path()));
                }
            }
        }
        let Some((step, dir)) = latest else {
            return Ok(None);
        };
        let lora_path = dir.join("lora.safetensors");
        if !lora_path.exists() {
            return Ok(None);
        }
        let lora_tensors = deserialize_tensors(&lora_path)?;
        let param_indices: Vec<(usize, String)> =
            self.param_names.iter().enumerate().map(|(i, n)| (i, n.clone())).collect();
        for (_idx, name) in param_indices.iter() {
            let key = format!("lora.{name}");
            let loaded = lora_tensors
                .get(&key)
                .ok_or_else(|| anyhow!("missing tensor {} in {}", key, lora_path.display()))?;
            let values = match loaded.dtype {
                SafeDtype::BF16 => bf16_bytes_to_f32_vec(&loaded.bytes)?,
                SafeDtype::F32 => f32_bytes_to_vec(&loaded.bytes)?,
                other => bail!("unsupported dtype {:?} for {}", other, key),
            };
            let tensor = tensor_from_slice_on(
                &values,
                Shape::from_dims(&loaded.shape),
                &self.device,
                DType::F32,
            )?
            .to_dtype(DType::BF16)?;
            self.apply_lora_tensor(name, &tensor)?;
        }

        if self.ema_enabled {
            let ema_path = dir.join("ema.safetensors");
            if ema_path.exists() {
                let ema_tensors = deserialize_tensors(&ema_path)?;
                if self.ema_shadow.len() != self.lora_params.len() {
                    self.ema_shadow.clear();
                    for param in &self.lora_params {
                        self.ema_shadow.push(param.tensor()?.to_dtype(DType::F32)?);
                    }
                }
                for (idx, name) in param_indices.iter() {
                    let key = format!("ema.{name}");
                    if let Some(loaded) = ema_tensors.get(&key) {
                        let values = match loaded.dtype {
                            SafeDtype::F32 => f32_bytes_to_vec(&loaded.bytes)?,
                            SafeDtype::BF16 => bf16_bytes_to_f32_vec(&loaded.bytes)?,
                            other => bail!("unsupported dtype {:?} for {}", other, key),
                        };
                        let tensor = tensor_from_slice_on(
                            &values,
                            Shape::from_dims(&loaded.shape),
                            &self.device,
                            DType::F32,
                        )?;
                        self.ema_shadow[*idx] = tensor;
                    }
                }
            } else {
                self.ema_shadow.clear();
                for param in &self.lora_params {
                    self.ema_shadow.push(param.tensor()?.to_dtype(DType::F32)?);
                }
            }
        }

        let meta = dir.join("meta.json");
        let mut seed = self.config.seed;
        if meta.exists() {
            if let Ok(value) =
                serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&meta)?)
            {
                if let Some(s) = value.get("seed").and_then(|v| v.as_u64()) {
                    seed = s;
                }
            }
        }
        if let Ok(meta) = fs::metadata(&lora_path) {
            self.last_lora_bytes = meta.len() as usize;
        }
        self.step_counter = step;
        Ok(Some((step, seed)))
    }

    fn apply_lora_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() != 5 {
            return Ok(());
        }
        let block_idx = parts[0].trim_start_matches("block").parse::<usize>().unwrap_or(0);
        let blocks = self.lora.blocks_mut();
        if let Some(block) = blocks.get_mut(block_idx) {
            match (parts[1], parts[2], parts[4]) {
                ("attn", "q", "A") => {
                    if let Some(adapter) = block.q.as_mut() {
                        adapter.a.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("attn", "q", "B") => {
                    if let Some(adapter) = block.q.as_mut() {
                        adapter.b.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("attn", "k", "A") => {
                    if let Some(adapter) = block.k.as_mut() {
                        adapter.a.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("attn", "k", "B") => {
                    if let Some(adapter) = block.k.as_mut() {
                        adapter.b.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("attn", "v", "A") => {
                    if let Some(adapter) = block.v.as_mut() {
                        adapter.a.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("attn", "v", "B") => {
                    if let Some(adapter) = block.v.as_mut() {
                        adapter.b.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("attn", "o", "A") => {
                    if let Some(adapter) = block.o.as_mut() {
                        adapter.a.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("attn", "o", "B") => {
                    if let Some(adapter) = block.o.as_mut() {
                        adapter.b.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("mlp", "fc1", "A") => {
                    if let Some(adapter) = block.fc1.as_mut() {
                        adapter.a.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("mlp", "fc1", "B") => {
                    if let Some(adapter) = block.fc1.as_mut() {
                        adapter.b.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("mlp", "fc2", "A") => {
                    if let Some(adapter) = block.fc2.as_mut() {
                        adapter.a.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                ("mlp", "fc2", "B") => {
                    if let Some(adapter) = block.fc2.as_mut() {
                        adapter.b.set_data(tensor.to_dtype(DType::BF16)?)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn update_ema(&mut self) -> Result<()> {
        if !self.ema_enabled {
            return Ok(());
        }
        if self.ema_shadow.len() != self.lora_params.len() {
            self.ema_shadow.clear();
            for param in &self.lora_params {
                self.ema_shadow.push(param.tensor()?.to_dtype(DType::F32)?);
            }
        }
        let decay = self.config.ema_decay as f32;
        let one_minus = 1.0f32 - decay;
        for (idx, param) in self.lora_params.iter().enumerate() {
            let current = param.tensor()?.to_dtype(DType::F32)?;
            let ema_prev = &self.ema_shadow[idx];
            let decayed = ema_prev.mul_scalar(decay)?;
            let added = current.mul_scalar(one_minus)?;
            self.ema_shadow[idx] = decayed.add(&added)?;
        }
        Ok(())
    }

    fn write_lora_safetensor(&self, path: &Path, step: u64, seed: u64) -> Result<()> {
        let mut tensors: Vec<(String, OwnedTensorView)> = Vec::new();
        for (idx, name) in self.param_names.iter().enumerate() {
            let param = self.lora_params[idx].tensor()?;
            let (shape, data) = tensor_to_bf16_bytes(&param)?;
            tensors
                .push((format!("lora.{name}"), OwnedTensorView::new(SafeDtype::BF16, shape, data)));
        }
        let mut metadata = HashMap::new();
        metadata.insert("step".to_string(), step.to_string());
        metadata.insert("seed".to_string(), seed.to_string());
        metadata.insert("dtype_policy".to_string(), "storage=bf16,compute=fp32".to_string());
        serialize_tensors(path, metadata, tensors)?;
        Ok(())
    }

    fn write_ema_safetensor(&self, path: &Path, step: u64, seed: u64) -> Result<()> {
        if !self.ema_enabled || self.ema_shadow.is_empty() {
            return Ok(());
        }
        let mut tensors: Vec<(String, OwnedTensorView)> = Vec::new();
        for (idx, name) in self.param_names.iter().enumerate() {
            let ema = &self.ema_shadow[idx];
            let (shape, data) = tensor_to_f32_bytes(ema)?;
            tensors
                .push((format!("ema.{name}"), OwnedTensorView::new(SafeDtype::F32, shape, data)));
        }
        let mut metadata = HashMap::new();
        metadata.insert("step".to_string(), step.to_string());
        metadata.insert("seed".to_string(), seed.to_string());
        metadata.insert("ema_decay".to_string(), self.config.ema_decay.to_string());
        serialize_tensors(path, metadata, tensors)?;
        Ok(())
    }
}

fn tensor_scalar_f32(tensor: &Tensor) -> Result<f32> {
    let vec = tensor.to_dtype(DType::F32)?.to_vec()?;
    vec.first().copied().ok_or_else(|| anyhow!("expected scalar tensor"))
}

/// Create a Flux trainer from strict weights and adapters.
pub async fn create_flux_trainer(
    model_path: &Path,
    vae_path: &Path,
    t5_path: &Path,
    clip_path: &Path,
    t5_tokenizer_path: &Path,
    clip_tokenizer_path: &Path,
    _variant: FluxVariant,
    config: FluxTrainingConfig,
    device: Device,
) -> Result<FluxTrainer> {
    ensure!(clip_path.exists(), "CLIP weights not found at {}", clip_path.display());
    ensure!(
        clip_tokenizer_path.exists(),
        "CLIP tokenizer missing at {}",
        clip_tokenizer_path.display()
    );

    let flame_dev = match device {
        Device::Cuda(ix) => FlameDevice::cuda(ix)?,
        _ => bail!("Flux training requires a CUDA device"),
    };
    let provider = FluxWeightProvider::from_path(
        model_path,
        Device::from_flame_cuda(flame_dev.cuda_device().as_ref()),
    )?;
    let registry = build_layer_registry(&provider)?;
    let metadata = registry.metadata();
    ensure!(!metadata.is_empty(), "no Flux blocks discovered");

    let hidden = metadata[0].hidden;
    let _mlp_inner = metadata[0].mlp_inner;
    let image_cond_dim = metadata[0].image_cond;
    let text_cond_dim = metadata[0].text_cond;

    let rank = config.lora_rank.max(1);
    let alpha = if config.lora_alpha.is_finite() && config.lora_alpha > 0.0 {
        config.lora_alpha
    } else {
        rank as f32
    };

    let mut lora_handles = FluxLoraHandles::new();
    for block_meta in &metadata {
        let mut block = FluxBlockLora::empty();
        block.q = Some(FluxLoraLinear::new(hidden, hidden, rank, alpha, flame_dev.clone())?);
        block.k = Some(FluxLoraLinear::new(hidden, hidden, rank, alpha, flame_dev.clone())?);
        block.v = Some(FluxLoraLinear::new(hidden, hidden, rank, alpha, flame_dev.clone())?);
        block.o = Some(FluxLoraLinear::new(hidden, hidden, rank, alpha, flame_dev.clone())?);
        block.fc1 = Some(FluxLoraLinear::new(
            hidden,
            block_meta.mlp_inner,
            rank,
            alpha,
            flame_dev.clone(),
        )?);
        block.fc2 = Some(FluxLoraLinear::new(
            block_meta.mlp_inner,
            hidden,
            rank,
            alpha,
            flame_dev.clone(),
        )?);
        lora_handles.push_block(block);
    }

    let lora_params = lora_handles.all_params();
    let param_views: Vec<Tensor> = lora_params.iter().filter_map(|p| p.as_tensor().ok()).collect();
    let param_refs: Vec<&Tensor> = param_views.iter().collect();

    let mut optimizer = create_optimizer(
        OptimizerConfig { lr: config.learning_rate, ..OptimizerConfig::default() },
        &param_refs,
    )?;
    let param_names = lora_handles.names_in_order();
    let _ = optimizer.set_param_names(param_names.clone());

    let gradient_accumulator =
        GradientAccumulator::new(config.gradient_accumulation_steps, device.clone())?;

    let registry_plan = FluxRegistryPlan {
        num_blocks: metadata.len(),
        with_mid: false,
        lora: Some(TrainerLoraSpec { rank, alpha, zero_init: config.lora_zero_init }),
    };
    let flux_registry = FluxRegistry::build(&registry_plan);

    let ae = FluxAE::load(vae_path.to_string_lossy().as_ref(), device.clone(), DType::BF16)
        .map_err(|e| anyhow!("FluxAE load failed: {e}"))?;
    let text = FluxText::load(
        t5_tokenizer_path.to_string_lossy().as_ref(),
        t5_path.to_string_lossy().as_ref(),
        device.clone(),
        DType::BF16,
    )
    .map_err(|e| anyhow!("FluxText load failed: {e}"))?;

    let text_hidden_in = text.hidden_dim();

    FluxTrainer::new(
        registry,
        flux_registry,
        metadata,
        lora_handles,
        lora_params,
        param_views,
        param_names,
        ae,
        text,
        optimizer,
        gradient_accumulator,
        config,
        device,
        hidden,
        text_hidden_in,
        image_cond_dim,
        text_cond_dim,
    )
}
