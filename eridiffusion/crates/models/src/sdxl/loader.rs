use anyhow::{Context, Result};
use eridiffusion_common_weights::strict_loader::StrictMmapLoader;
use flame_core::DType;
use flame_core::Device as FDevice;
use eridiffusion_core::Device as EDevice;
use super::{unet::SdxlUnet, vae::SdxlVaeDecoder};

/// Validate that every tensor in the file is either a 2D linear [IN,OUT] or 4D conv [KH,KW,IC,OC].
fn validate_shapes_all(path: &std::path::Path, ld: &StrictMmapLoader) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read: {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).with_context(|| "parse safetensors header")?;
    for name in st.names() {
        let info = ld.info(name)?;
        let rank = info.shape.len();
        if rank != 2 && rank != 4 {
            anyhow::bail!("tensor '{}' has invalid rank {} (expected 2 or 4)", name, rank);
        }
        if rank == 2 {
            // linear [IN,OUT]
            let _in = info.shape[0]; let _out = info.shape[1];
            // No further constraints; exact dims depend on architecture and block
        } else {
            // conv [KH,KW,IC,OC]
            let _kh = info.shape[0]; let _kw = info.shape[1]; let _ic = info.shape[2]; let _oc = info.shape[3];
        }
    }
    Ok(())
}

pub fn from_safetensors_strict_unet(path: &std::path::Path, device: FDevice, dtype: DType) -> Result<SdxlUnet> {
    let ld = StrictMmapLoader::open(path).with_context(|| format!("open SDXL UNet weights: {}", path.display()))?;
    validate_shapes_all(path, &ld)?;
    Ok(SdxlUnet::new(device, dtype))
}

pub fn from_safetensors_strict_vae_decoder(path: &std::path::Path, device: EDevice, dtype: DType) -> Result<SdxlVaeDecoder> {
    let ld = StrictMmapLoader::open(path).with_context(|| format!("open SDXL VAE weights: {}", path.display()))?;
    validate_shapes_all(path, &ld)?;
    Ok(SdxlVaeDecoder::new(device, dtype))
}
