use anyhow::{Result, Context, bail};
use eridiffusion_common_weights as cw;
use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use flame_core::DType;

pub fn guard_unet_loader(ld: &cw::SafeLoader) -> Result<()> {
    // Prefix histogram
    let keys = ld.list_keys()?;
    let total = keys.len();
    let enc = keys.iter().filter(|k| k.starts_with("encoder.")).count();
    tracing::info!("prefix histogram: encoder.*={} / total={}", enc, total);
    if enc * 2 > total { bail!("Refusing text-encoder checkpoint for image backbone"); }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RenameRule { from: String, to: String }
#[derive(Debug, Deserialize)]
struct RenameMap { rules: Vec<RenameRule> }

pub fn load_rename_map(path: &str) -> Result<Vec<(Regex, String)>> {
    let txt = std::fs::read_to_string(path).with_context(|| format!("read rename map {}", path))?;
    let rm: RenameMap = serde_yaml::from_str(&txt).with_context(|| "parse rename map yaml")?;
    let mut out = Vec::new();
    for r in rm.rules { out.push((Regex::new(&r.from)?, r.to)); }
    Ok(out)
}

pub fn apply_rules(rules: &[(Regex, String)], key: &str) -> Option<String> {
    for (re, to) in rules {
        if let Some(caps) = re.captures(key) {
            let mut out = to.clone();
            for (i, c) in caps.iter().enumerate() {
                if i==0 { continue; }
                if let Some(m) = c { out = out.replace(&format!("${{{}}}", i), m.as_str()); }
            }
            return Some(out);
        }
    }
    None
}

pub fn map_and_register(ld: &mut cw::SafeLoader, reg: &mut cw::ParamRegistry, device: &flame_core::Device) -> Result<()> {
    let rules = load_rename_map("configs/sdxl_rename_map.yaml")?;
    let keys = ld.list_keys()?;
    let mut used: HashSet<String> = HashSet::new();
    let mut mapped = 0usize;
    for k in &keys {
        if !k.starts_with("model.diffusion_model.") { continue; }
        if let Some(name) = apply_rules(&rules, k) {
            // Load tensor as BF16 param on device
            let mut t = ld.get_bf16(k)?;
            // Normalize layouts by rank
            let dims = t.shape().dims().to_vec();
            t = match dims.len() {
                2 => {
                    // Linear: assume [OUT,IN] -> transpose to [IN,OUT]
                    t.transpose()?.to_dtype(DType::BF16)?
                }
                4 => {
                    // Conv: assume [OC,IC,KH,KW] -> convert
                    flame_core::cuda_ops::GpuOps::weight_ocickhkw_to_khwkicoc(&t)?.to_dtype(DType::BF16)?
                }
                _ => t.to_dtype(DType::BF16)?,
            };
            reg.insert(&name, t);
            used.insert(k.clone());
            mapped += 1;
        }
    }
    // Coverage: any diffusion_model keys not mapped are errors
    let mut unmapped: Vec<&str> = Vec::new();
    for k in &keys {
        if k.starts_with("model.diffusion_model.") && !used.contains(k) { unmapped.push(k.as_str()); }
    }
    if !unmapped.is_empty() {
        bail!("unmapped UNet keys: e.g. {:?} ({} total)", &unmapped[..unmapped.len().min(10)], unmapped.len());
    }
    tracing::info!("UNet mapping complete: {} tensors", mapped);
    Ok(())
}
