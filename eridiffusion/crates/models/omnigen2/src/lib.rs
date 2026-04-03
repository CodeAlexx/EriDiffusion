use anyhow::Result;
use flame_core::Tensor;

pub trait DiffusionModule {
    fn forward(&self, latents: &Tensor, t: &Tensor, ctx: Option<&Tensor>) -> Result<Tensor>;
}

pub struct Omnigen2Model { pub dim: usize, pub n_layers: usize, pub n_heads: usize, pub ctx_dim: usize }
impl Omnigen2Model { pub fn new() -> Self { Self { dim: 3072, n_layers: 24, n_heads: 24, ctx_dim: 2048 } } }

impl DiffusionModule for Omnigen2Model {
    fn forward(&self, latents: &Tensor, _t: &Tensor, _ctx: Option<&Tensor>) -> Result<Tensor> {
        Ok(flame_core::Tensor::zeros(latents.shape().clone(), latents.device().clone())?)
    }
}

mod register {
    use super::*;
    use anyhow::Result;
    use eridiffusion_common_weights::ParamRegistry;
    use eridiffusion_model_registry as regy;
    struct Wrap(super::Omnigen2Model);
    impl regy::DiffusionModule for Wrap {
        fn forward(&self, latents: &Tensor, t: &Tensor, ctx: Option<&Tensor>) -> Result<Tensor> { super::Omnigen2Model::forward(&self.0, latents, t, ctx) }
    }
    fn build(_cfg: &serde_yaml::Value, _load: &regy::LoadSpec, _reg: &mut ParamRegistry, _device: &flame_core::Device) -> Result<Box<dyn regy::DiffusionModule>> {
        Ok(Box::new(Wrap(Omnigen2Model::new())))
    }
    inventory::submit! { regy::ModelEntry { id: "omnigen2", build } }
}
