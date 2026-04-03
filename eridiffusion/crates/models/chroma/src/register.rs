use anyhow::Result;
use flame_core::Device;
use eridiffusion_common_weights::ParamRegistry;
use eridiffusion_model_registry as regy;
use crate::ChromaModel;
use crate::forward::ChromaModule;

struct Wrapper(crate::ChromaModel);

impl regy::DiffusionModule for Wrapper {
    fn forward(&self, latents: &flame_core::Tensor, t: &flame_core::Tensor, ctx: Option<&flame_core::Tensor>) -> Result<flame_core::Tensor> {
        // ChromaModule requires ctx Some(&Tensor)
        let ctx_ref = ctx.expect("Chroma requires context tensor");
        self.0.forward(latents, t, ctx_ref)
    }
}

fn build_chroma(_cfg: &serde_yaml::Value, _load: &regy::LoadSpec, _reg: &mut ParamRegistry, _device: &Device) -> Result<Box<dyn regy::DiffusionModule>> {
    let model = ChromaModel::new(1536, 4096);
    Ok(Box::new(Wrapper(model)))
}

inventory::submit! {
    regy::ModelEntry { id: "chroma", build: build_chroma }
}
