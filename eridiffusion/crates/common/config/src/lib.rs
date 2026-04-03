use anyhow::{Result, Context, bail};
use serde::{Deserialize, Serialize};
use hashbrown::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VaeKind { Sdxl, Sd35, Flux }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaeBlock {
    pub kind: VaeKind,
    pub path: String,
    pub latent_div: usize,
    pub latent_channels: usize,
    pub latent_scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VaeField {
    Path(String),
    Block(VaeBlock),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelSpec {
    pub weights: Option<String>,
    pub weights_img: Option<String>,
    pub weights_txt: Option<String>,
    pub vae: Option<VaeField>,
    pub seq: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TESpec {
    pub path: String,
    pub seq: Option<usize>,
    pub ctx_dim: Option<serde_yaml::Value>, // allow "auto" or number
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VaePolicy { pub device: Option<String>, pub fallback_on_oom: Option<bool> }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MoePolicy { pub quant: Option<String>, pub top_k: Option<usize> }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Policies { pub vae: Option<VaePolicy>, pub moe: Option<MoePolicy> }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelPaths {
    pub models: HashMap<String, ModelSpec>,
    pub text_encoders: HashMap<String, TESpec>,
    pub policies: Option<Policies>,
}

pub fn load(path: &str) -> Result<ModelPaths> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path))?;
    let mp: ModelPaths = serde_yaml::from_reader(f).with_context(|| format!("parse YAML {}", path))?;
    Ok(mp)
}

pub fn path_for_model<'a>(mp: &'a ModelPaths, id: &str) -> Result<&'a ModelSpec> {
    mp.models.get(id).ok_or_else(|| anyhow::anyhow!("model '{}' not found in ModelPaths", id))
}

/// Resolve a VAE block from a model spec. Enforces explicit latent_scale.
pub fn resolve_vae_block(spec: &ModelSpec) -> Result<VaeBlock> {
    match &spec.vae {
        Some(VaeField::Block(b)) => Ok(b.clone()),
        Some(VaeField::Path(_path)) => {
            bail!("modelPath.yaml: 'vae' must be a block with kind/path/latent_div/latent_channels/latent_scale; string paths are no longer accepted")
        }
        None => bail!("modelPath.yaml: missing 'vae' block for model"),
    }
}

/// Read run_meta.json from a precompute directory and assert VAE compatibility
pub fn assert_precomp_vae_compat<P: AsRef<std::path::Path>>(precomp_dir: P, expected: &VaeBlock) -> Result<()> {
    let p = precomp_dir.as_ref().join("run_meta.json");
    let s = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&s).with_context(|| "parse run_meta.json")?;
    let vae = v.get("vae").ok_or_else(|| anyhow::anyhow!("run_meta.json missing 'vae'"))?;
    let kind = vae.get("kind").and_then(|x| x.as_str()).ok_or_else(|| anyhow::anyhow!("run_meta.json.vae.kind missing"))?;
    let latent_div = vae.get("latent_div").and_then(|x| x.as_u64()).ok_or_else(|| anyhow::anyhow!("run_meta.json.vae.latent_div missing"))? as usize;
    let latent_channels = vae.get("latent_channels").and_then(|x| x.as_u64()).ok_or_else(|| anyhow::anyhow!("run_meta.json.vae.latent_channels missing"))? as usize;
    let latent_scale = vae.get("latent_scale").and_then(|x| x.as_f64()).ok_or_else(|| anyhow::anyhow!("run_meta.json.vae.latent_scale missing"))? as f32;
    let expected_kind = match expected.kind { VaeKind::Sdxl => "sdxl", VaeKind::Sd35 => "sd35", VaeKind::Flux => "flux" };
    if kind != expected_kind || latent_div != expected.latent_div || latent_channels != expected.latent_channels || (latent_scale - expected.latent_scale).abs() > 1e-6 {
        bail!("precomp VAE mismatch: have kind={}, div={}, channels={}, scale={}, expected kind={}, div={}, channels={}, scale={}",
            kind, latent_div, latent_channels, latent_scale, expected_kind, expected.latent_div, expected.latent_channels, expected.latent_scale);
    }
    Ok(())
}

pub fn text_encoder_spec<'a>(mp: &'a ModelPaths, id: &str) -> Result<&'a TESpec> {
    mp.text_encoders.get(id).ok_or_else(|| anyhow::anyhow!("text encoder '{}' not found", id))
}

pub fn vae_policy(mp: &ModelPaths) -> VaePolicy {
    mp.policies.as_ref().and_then(|p| p.vae.clone()).unwrap_or_default()
}

pub fn wan_moe_policy(mp: &ModelPaths) -> MoePolicy {
    // Defaults: quant=q8, top_k=2
    let mut m = mp.policies.as_ref().and_then(|p| p.moe.clone()).unwrap_or_default();
    if m.quant.is_none() { m.quant = Some("q8".into()); }
    if m.top_k.is_none() { m.top_k = Some(2); }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip_template() -> Result<()> {
        let mp = load("configs/modelPath.yaml")?;
        assert!(mp.models.contains_key("flux"));
        let moe = wan_moe_policy(&mp);
        assert_eq!(moe.top_k.unwrap_or(0), 2);
        Ok(())
    }
}
