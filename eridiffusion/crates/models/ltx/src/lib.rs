use anyhow::Result;
use flame_core::Tensor;

pub trait DiffusionModule {
    fn forward(&self, latents: &Tensor, t: &Tensor, ctx: Option<&Tensor>) -> Result<Tensor>;
}

pub struct LtxModel { pub dim: usize, pub n_layers: usize, pub n_heads: usize }
impl LtxModel { pub fn new() -> Self { Self { dim: 3072, n_layers: 24, n_heads: 24 } } }

impl DiffusionModule for LtxModel {
    fn forward(&self, latents: &Tensor, _t: &Tensor, _ctx: Option<&Tensor>) -> Result<Tensor> {
        Ok(flame_core::Tensor::zeros(latents.shape().clone(), latents.device().clone())?)
    }
}

mod register {
    use super::*;
    use anyhow::Result;
    use eridiffusion_common_weights::ParamRegistry;
    use eridiffusion_model_registry as regy;
    struct Wrap(super::LtxModel);
    impl regy::DiffusionModule for Wrap {
        fn forward(&self, latents: &Tensor, t: &Tensor, _ctx: Option<&Tensor>) -> Result<Tensor> { super::LtxModel::forward(&self.0, latents, t, None) }
    }
    fn build(_cfg: &serde_yaml::Value, _load: &regy::LoadSpec, _reg: &mut ParamRegistry, _device: &flame_core::Device) -> Result<Box<dyn regy::DiffusionModule>> {
        Ok(Box::new(Wrap(LtxModel::new())))
    }
    inventory::submit! { regy::ModelEntry { id: "ltx", build } }
}
