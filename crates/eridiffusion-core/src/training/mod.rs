pub mod block_offload;
pub mod checkpoint;
pub mod ema;
pub mod logging;
pub mod offload;
pub mod schedule;
pub mod training_features;
pub mod training_offload;

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Tensor};
use flame_core::autograd::AutogradContext;
use flame_core::gradient_clip::GradientClipper;
use flame_core::adam::AdamW;

use crate::config::TrainConfig;
use crate::data::CachedDataset;
use crate::lora::LoRALinear;
use crate::models::TrainableModel;
use crate::pipeline::TrainingPipeline;
use crate::Result;

use self::ema::ParameterEma;

/// EriDiffusion's training loop — mirrors GenericTrainer from Python EriDiffusion.
///
/// Flow per step:
///   sample = dataset.load(idx, &device)
///   (inputs, targets) = pipeline.prepare_inputs(sample.latent, sample.embeddings)
///   pred = model.forward(inputs.noisy, inputs.timestep, inputs.context, pooled)
///   loss = pipeline.compute_loss(pred, targets.target, timestep_idx)
///   grads = loss.backward()
///   accumulate_parameter_grads(params, grads)
///   clip_grads(params, clipper)
///   {no_grad: optimizer.step(params); optimizer.zero_grad(params)}
///   model.post_optimizer_step()
///   ema.update(params)
///   AutogradContext::clear()
pub struct GenericTrainer<T: TrainableModel, P: TrainingPipeline> {
    pub config: TrainConfig,
    pub device: Arc<CudaDevice>,
    pub model: T,
    pub pipeline: P,
    pub dataset: CachedDataset,
    pub optimizer: AdamW,
    pub ema: Option<ParameterEma>,
    pub output_dir: PathBuf,
    pub step: usize,
}

impl<T: TrainableModel, P: TrainingPipeline> GenericTrainer<T, P> {
    pub fn new(
        config: TrainConfig,
        model: T,
        pipeline: P,
        dataset: CachedDataset,
        output_dir: PathBuf,
    ) -> Result<Self> {
        flame_core::config::set_default_dtype(DType::BF16);
        let device = flame_core::global_cuda_device();

        let params = model.parameters();
        let opt_cfg = &config.optimizer;
        let lr = opt_cfg.learning_rate.unwrap_or(config.learning_rate) as f32;

        let optimizer = AdamW::new(
            lr,
            opt_cfg.beta1 as f32,
            opt_cfg.beta2 as f32,
            opt_cfg.eps as f32,
            opt_cfg.weight_decay as f32,
        );

        let ema = if config.ema != crate::config::EmAMode::Off {
            Some(ParameterEma::new(&params, config.ema_decay as f32)?)
        } else {
            None
        };

        log::info!("Trainer init: {} params, lr={}, wd={}, ema={:?}",
            params.len(), lr, opt_cfg.weight_decay, config.ema);

        Ok(Self { config, device, model, pipeline, dataset, optimizer, ema, output_dir, step: 0 })
    }

    /// Single training step
    pub fn train_step(&mut self) -> Result<f32> {
        let idx = self.step % self.dataset.len();
        let sample = self.dataset.get(idx)
            .ok_or_else(|| crate::EriDiffusionError::Data("sample out of bounds".into()))?;

        // Load tensors from cached sample
        let latent = sample.latent_tensor(&self.device)?;
        let embeddings: Vec<Tensor> = sample.embedding_keys.iter()
            .filter_map(|k| sample.embedding(k, &self.device).ok())
            .collect();

        // Pipeline: add noise, compute targets
        let (inputs, targets) = self.pipeline.prepare_inputs(&latent, &embeddings, None)?;

        // Model forward
        let pred = self.model.forward(&inputs.noisy, &inputs.timestep, &inputs.context, inputs.pooled.as_ref())?;

        // Loss
        let loss = self.pipeline.compute_loss(&pred, &targets.target, None)?;

        // Check loss is finite
        let loss_f32 = loss.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let loss_val = loss_f32.first().copied().unwrap_or(f32::NAN);
        if !loss_val.is_finite() {
            return Err(crate::EriDiffusionError::Training(
                format!("step {}: non-finite loss {}", self.step, loss_val)
            ));
        }

        // Backward + optimize
        let grads = loss.backward()?;
        let params = self.model.parameters();
        accumulate_parameter_grads(&params, &grads)?;

        let clipper = GradientClipper::clip_by_norm(self.config.clip_grad_norm as f32);
        let grad_norm = clip_parameter_grads(&params, &clipper)?;

        {
            let _guard = AutogradContext::no_grad();
            self.optimizer.step(&params)?;
            self.optimizer.zero_grad(&params);
            self.model.post_optimizer_step();

            if let Some(ref mut ema) = self.ema {
                ema.update(&params)?;
            }
        }
        AutogradContext::clear();

        self.step += 1;

        if self.step % 100 == 0 {
            log::info!("step {} | loss {:.4} | grad_norm {:.4}", self.step, loss_val, grad_norm);
        }

        Ok(loss_val)
    }

    /// Run full training loop
    pub fn train(&mut self, max_steps: usize) -> Result<()> {
        log::info!("Starting training: {} steps", max_steps);
        let t0 = std::time::Instant::now();

        for _ in 0..max_steps {
            self.train_step()?;

            // Periodic checkpoint
            if self.step > 0 && self.step % 500 == 0 {
                let ckpt_dir = self.output_dir.join("checkpoints");
                std::fs::create_dir_all(&ckpt_dir)?;
                let ckpt = ckpt_dir.join(format!("lora_step_{:06}.safetensors", self.step));
                self.model.save_weights(&ckpt.to_string_lossy())?;
                log::info!("checkpoint: {}", ckpt.display());
            }
        }

        let elapsed = t0.elapsed().as_secs_f32();
        log::info!("Training complete. {} steps in {:.1}s", max_steps, elapsed);
        Ok(())
    }
}

/// Gradient accumulation: copy grads from GradientMap to Parameters
pub fn accumulate_parameter_grads(
    params: &[Parameter],
    grads: &flame_core::gradient::GradientMap,
) -> Result<()> {
    for param in params {
        if let Some(g) = grads.get(param.id()) {
            let g = if g.dtype() == DType::F32 { g.clone() } else { g.to_dtype(DType::F32)? };
            param.set_grad(g)?;
        }
    }
    Ok(())
}

/// Clip parameter gradients by norm, return norm
pub fn clip_parameter_grads(params: &[Parameter], clipper: &GradientClipper) -> Result<f32> {
    let mut grads: Vec<Tensor> = Vec::new();
    let mut owners: Vec<usize> = Vec::new();
    for (idx, param) in params.iter().enumerate() {
        if let Some(g) = param.grad() {
            grads.push(g);
            owners.push(idx);
        }
    }
    if grads.is_empty() {
        return Ok(0.0);
    }
    let mut grad_refs: Vec<&mut Tensor> = grads.iter_mut().collect();
    let norm = clipper.clip_grads(&mut grad_refs)?;
    for (owner, grad) in owners.into_iter().zip(grads.into_iter()) {
        params[owner].set_grad(grad)?;
    }
    Ok(norm)
}
