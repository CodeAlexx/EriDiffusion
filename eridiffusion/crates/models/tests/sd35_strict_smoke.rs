#![cfg(feature = "sd35_strict_loader")]

use anyhow::Result;

use eridiffusion_models::sd35::keymap::{EXPANDED_DIM, HIDDEN_SIZE};
use eridiffusion_models::sd35::strict_iface::Sd35StrictBundleExt;
use eridiffusion_models::sd35::strict_loader::{last_strict_loader_message, load_sd35_strict};
use flame_core::DType;

const MODEL_PATH: &str = "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors";

#[test]
fn sd35_strict_loader_smoke() -> Result<()> {
    let bundle = match load_sd35_strict(MODEL_PATH) {
        Ok(bundle) => bundle,
        Err(err) => {
            let mut skip = err.to_string().contains("create CUDA device");
            if !skip {
                skip = err.chain().any(|cause| cause.to_string().contains("CUDA error"));
            }
            if skip {
                eprintln!("skipping sd35 strict loader smoke test: {err:?}");
                return Ok(());
            }
            return Err(err);
        }
    };

    assert_eq!(last_strict_loader_message().as_deref(), Some("Strict load complete."),);

    let q_key = "model.diffusion_model.joint_blocks.0.x_block.attn.q.weight";
    let fc_key = "model.diffusion_model.joint_blocks.0.x_block.mlp.fc1.weight";

    let q = bundle.tensor(q_key).expect("missing q weight");
    assert_eq!(q.shape().dims(), &[HIDDEN_SIZE, HIDDEN_SIZE]);
    assert_eq!(q.dtype(), DType::BF16);

    let fc = bundle.tensor(fc_key).expect("missing fc1 weight");
    assert_eq!(fc.shape().dims(), &[HIDDEN_SIZE, EXPANDED_DIM]);
    assert_eq!(fc.dtype(), DType::BF16);

    assert!(bundle.has("model.diffusion_model.final_layer.linear.weight"));

    Ok(())
}
