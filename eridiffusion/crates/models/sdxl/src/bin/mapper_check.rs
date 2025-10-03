use anyhow::{Result, bail};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Path to UNet weights (.safetensors)
    #[arg(long)]
    weights: String,
    /// Optional rename map YAML (defaults to configs/sdxl_rename_map.yaml)
    #[arg(long)]
    rename_map: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut ld = eridiffusion_common_weights::SafeLoader::open(&args.weights)?;
    eridiffusion_models_sdxl::weight_load::guard_unet_loader(&ld)?;
    let rules_path = args.rename_map.as_deref().unwrap_or("configs/sdxl_rename_map.yaml");
    let rules = eridiffusion_models_sdxl::weight_load::load_rename_map(rules_path)?;
    let keys = ld.list_keys()?;
    let mut unmapped = Vec::new();
    let mut mapped = 0usize;
    for k in &keys {
        if !k.starts_with("model.diffusion_model.") { continue; }
        if let Some(_name) = eridiffusion_models_sdxl::weight_load::apply_rules(&rules, k) {
            // Invariants: if attn2 to_k/to_v, second dim must be 2048; to_q or to_out rows==cols
            if k.contains("attn2.to_k.weight") || k.contains("attn2.to_v.weight") {
                let shp = ld.shape_of(k)?; if shp.len()!=2 || shp[1]!=2048 { bail!("{} must be [C,2048], got {:?}", k, shp); }
            }
            if k.contains("attn2.to_q.weight") || k.contains("attn2.to_out.0.weight") {
                let shp = ld.shape_of(k)?; if shp.len()!=2 || shp[0]!=shp[1] { bail!("{} must be square [C,C], got {:?}", k, shp); }
            }
            mapped += 1;
        } else {
            unmapped.push(k.clone());
        }
    }
    if !unmapped.is_empty() {
        eprintln!("Unmapped keys (showing up to 20):");
        for k in unmapped.iter().take(20) { eprintln!("  {}", k); }
        bail!("unmapped {} keys under model.diffusion_model", unmapped.len());
    }
    println!("OK: mapped {} UNet keys", mapped);
    Ok(())
}
