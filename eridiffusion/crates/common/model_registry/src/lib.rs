use anyhow::{Result, bail};
use eridiffusion_common_weights as cw;
use flame_core::{Device, Tensor};

pub trait DiffusionModule: Send {
    fn forward(&self, latents: &Tensor, t: &Tensor, ctx: Option<&Tensor>) -> Result<Tensor>;
}

pub struct LoadSpec {
    pub weights: String,
    pub vae: Option<String>,
    pub te1: Option<String>,
    pub te2: Option<String>,
}

pub type BuildFn = fn(
    cfg: &serde_yaml::Value,
    load: &LoadSpec,
    reg: &mut cw::ParamRegistry,
    device: &Device,
) -> Result<Box<dyn DiffusionModule>>;

pub struct ModelEntry {
    pub id: &'static str,
    pub build: BuildFn,
}

inventory::collect!(ModelEntry);

pub fn known_models() -> Vec<&'static str> {
    inventory::iter::<ModelEntry>.into_iter().map(|e| e.id).collect()
}

pub fn build_model(
    id: &str,
    cfg: &serde_yaml::Value,
    load: &LoadSpec,
    reg: &mut cw::ParamRegistry,
    device: &Device,
) -> Result<Box<dyn DiffusionModule>> {
    for e in inventory::iter::<ModelEntry> {
        if e.id == id {
            return (e.build)(cfg, load, reg, device);
        }
    }
    bail!("Unknown model id '{}'. Known: {:?}", id, known_models());
}

pub fn open_guarded(weights: &str) -> Result<cw::SafeLoader> {
    let ld = cw::SafeLoader::open(weights)?;
    let keys = ld.list_keys()?;
    cw::assert_not_text_encoder(&keys)?;
    Ok(ld)
}

