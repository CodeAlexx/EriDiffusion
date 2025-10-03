pub mod block_swap;
pub mod weights;
pub mod schedule;
pub mod forward;
pub mod tokenizer;
pub mod high_noise;
pub mod low_noise;
pub mod lora_core;
pub mod lora_layer;
pub mod lora_attach;

/// WAN 2.2 T2I (dual-expert: high-noise, low-noise) — model facade
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wan22T2IConfig {
    pub path_high: String,
    pub path_low: String,
    pub dtype: flame_core::DType,   // default: BF16
    pub low_vram: bool,             // stream weights
    pub quantize: bool,             // if loader supports quant-on-CPU
}

#[derive(Debug)]
pub struct Wan22T2I {
    pub experts: weights::Wan22Experts,
    pub handoff: schedule::HandoffPolicy,
}

impl Wan22T2I {
    pub fn load(cfg: &Wan22T2IConfig, device: flame_core::Device) -> anyhow::Result<Self> {
        use weights::load_wan22_t2i;
        let experts = load_wan22_t2i(&cfg.path_high, &cfg.path_low, device, cfg.dtype)?;
        // Publish experts for stateless forward path
        super::weights::set_global_experts(experts.clone());
        Ok(Self { experts, handoff: schedule::HandoffPolicy::FixedFrac(0.5) })
    }

    pub fn with_handoff(mut self, policy: schedule::HandoffPolicy) -> Self { self.handoff = policy; self }

    /// Best-effort expected context (text) dim from weights: use cross-attn K input dim.
    pub fn expected_ctx_dim(&self) -> Option<usize> {
        self.experts.high.blocks.get(0).map(|b| b.cross_k.w_in_out.shape().dims()[1])
    }
}

// Registry wiring — register both a legacy stub (wan22) and the new dual-expert T2I (wan22_t2i).
mod register {
    use anyhow::Result;
    use eridiffusion_model_registry as regy;
    use eridiffusion_common_weights::ParamRegistry;
    use tracing::info;

    use super::{Wan22T2I, Wan22T2IConfig};

    struct Stub;
    impl regy::DiffusionModule for Stub {
        fn forward(&self, latents: &flame_core::Tensor, _t: &flame_core::Tensor, _ctx: Option<&flame_core::Tensor>) -> Result<flame_core::Tensor> {
            Ok(flame_core::Tensor::zeros(latents.shape().clone(), latents.device().clone())?)
        }
    }

    fn build_stub(_cfg: &serde_yaml::Value, _load: &regy::LoadSpec, _reg: &mut ParamRegistry, _device: &flame_core::Device) -> Result<Box<dyn regy::DiffusionModule>> {
        Ok(Box::new(Stub))
    }

    struct Wan22Module(Wan22T2I);
    impl regy::DiffusionModule for Wan22Module {
        fn forward(&self, latents: &flame_core::Tensor, t: &flame_core::Tensor, ctx: Option<&flame_core::Tensor>) -> Result<flame_core::Tensor> {
            // Thin dispatch to current expert selection policy; for now, training-first stub returns zeros
            let _ = (&self.0, t, ctx);
            Ok(flame_core::Tensor::zeros(latents.shape().clone(), latents.device().clone())?)
        }
    }

    fn build_wan22_t2i(cfg: &serde_yaml::Value, _load: &regy::LoadSpec, _reg: &mut ParamRegistry, device: &flame_core::Device) -> Result<Box<dyn regy::DiffusionModule>> {
        // Read minimal fields from YAML; defaults align with training-first path
        let model = cfg.get("model").cloned().unwrap_or(serde_yaml::Value::Null);
        let path_high = model.get("path_high").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let path_low  = model.get("path_low") .and_then(|v| v.as_str()).unwrap_or("").to_string();
        let dtype_str = model.get("dtype").and_then(|v| v.as_str()).unwrap_or("bf16");
        let dtype = match dtype_str.to_ascii_lowercase().as_str() {
            "bf16" => flame_core::DType::BF16,
            "f32"  => flame_core::DType::F32,
            _ => flame_core::DType::BF16,
        };
        let low_vram = model.get("low_vram").and_then(|v| v.as_bool()).unwrap_or(true);
        let quantize = model.get("quantize").and_then(|v| v.as_bool()).unwrap_or(true);

        if path_high.is_empty() || path_low.is_empty() {
            info!("wan22_t2i: weights not specified in YAML; constructing compile stub");
            return Ok(Box::new(Stub));
        }

        let cfg = Wan22T2IConfig { path_high, path_low, dtype, low_vram, quantize };
        let m = Wan22T2I::load(&cfg, device.clone())?;
        // Metadata note: dual_expert=true, supports_lora_only=true
        info!("registered wan22_t2i (dual_expert=true, supports_lora_only=true)");
        Ok(Box::new(Wan22Module(m)))
    }

    inventory::submit! { regy::ModelEntry { id: "wan22", build: build_stub } }
    inventory::submit! { regy::ModelEntry { id: "wan22_t2i", build: build_wan22_t2i } }
}
