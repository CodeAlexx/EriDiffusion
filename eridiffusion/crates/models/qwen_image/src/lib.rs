pub mod block_swap;

mod register {
    use anyhow::Result;
    use eridiffusion_model_registry as regy;
    use eridiffusion_common_weights::ParamRegistry;
    struct Stub;
    impl regy::DiffusionModule for Stub {
        fn forward(&self, latents: &flame_core::Tensor, _t: &flame_core::Tensor, _ctx: Option<&flame_core::Tensor>) -> Result<flame_core::Tensor> {
            Ok(flame_core::Tensor::zeros(latents.shape().clone(), latents.device().clone())?)
        }
    }
    fn build(_cfg: &serde_yaml::Value, _load: &regy::LoadSpec, _reg: &mut ParamRegistry, _device: &flame_core::Device) -> Result<Box<dyn regy::DiffusionModule>> {
        Ok(Box::new(Stub))
    }
    inventory::submit! { regy::ModelEntry { id: "qwen_image", build } }
}
