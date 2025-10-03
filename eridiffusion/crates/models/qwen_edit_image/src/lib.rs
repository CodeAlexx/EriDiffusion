use anyhow::Result;
use flame_core::Tensor;

pub struct EditCond {
    pub ref_image: Option<Tensor>,  // NHWC [B,H,W,3] bf16
    pub mask: Option<Tensor>,       // NHW1 [B,H,W,1]
    pub text_delta: Option<Tensor>, // [B,seq,ctx_dim]
}

pub trait DiffusionModule {
    fn forward(&self, latents: &Tensor, t: &Tensor, ctx: Option<&Tensor>) -> Result<Tensor>;
}

pub trait EditModule {
    fn forward_edit(&self, latents: &Tensor, t: &Tensor, base_ctx: Option<&Tensor>, edit: &EditCond) -> Result<Tensor>;
}

pub struct QwenEditModel { pub ctx_dim: usize }
impl QwenEditModel { pub fn new() -> Self { Self { ctx_dim: 2048 } } }

impl EditModule for QwenEditModel {
    fn forward_edit(&self, latents: &Tensor, _t: &Tensor, _base_ctx: Option<&Tensor>, _edit: &EditCond) -> Result<Tensor> {
        Ok(flame_core::Tensor::zeros(latents.shape().clone(), latents.device().clone())?)
    }
}

impl DiffusionModule for QwenEditModel {
    fn forward(&self, latents: &Tensor, t: &Tensor, ctx: Option<&Tensor>) -> Result<Tensor> {
        let dummy = EditCond { ref_image: None, mask: None, text_delta: None };
        self.forward_edit(latents, t, ctx, &dummy)
    }
}

mod register {
    use super::*;
    use anyhow::Result;
    use eridiffusion_common_weights::ParamRegistry;
    use eridiffusion_model_registry as regy;
    struct Wrap(super::QwenEditModel);
    impl regy::DiffusionModule for Wrap {
        fn forward(&self, latents: &Tensor, t: &Tensor, ctx: Option<&Tensor>) -> Result<Tensor> { super::QwenEditModel::forward(&self.0, latents, t, ctx) }
    }
    fn build(_cfg: &serde_yaml::Value, _load: &regy::LoadSpec, _reg: &mut ParamRegistry, _device: &flame_core::Device) -> Result<Box<dyn regy::DiffusionModule>> {
        Ok(Box::new(Wrap(QwenEditModel::new())))
    }
    inventory::submit! { regy::ModelEntry { id: "qwen", build } }
}
