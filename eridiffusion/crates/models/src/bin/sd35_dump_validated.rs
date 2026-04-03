use std::env;

#[cfg(feature = "sd35_strict_loader")]
fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let mut model_path = None;
    while let Some(arg) = args.next() {
        if arg == "--model" {
            model_path = args.next();
        }
    }
    let path = model_path.ok_or_else(|| anyhow::anyhow!("--model path required"))?;
    use eridiffusion_models::sd35::strict_iface::Sd35StrictBundleExt;

    let bundle = eridiffusion_models::sd35::strict_loader::load_sd35_strict(&path)?;
    println!("validated tensors: {}", bundle.tensors().len());
    Ok(())
}

#[cfg(not(feature = "sd35_strict_loader"))]
fn main() {
    eprintln!("sd35_strict_loader feature not enabled");
}
