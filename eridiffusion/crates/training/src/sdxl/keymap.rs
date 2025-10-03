use crate::streaming::{KeyMap, KeyMapOwned};

pub struct SdxlKeyMap;

/// Flattened transformer blocks (attention or FF) in SDXL UNet.
pub const BASES: &[&str] = &[
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.1",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.1",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.1",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.2",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.3",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.4",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.5",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.6",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.7",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.8",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.9",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.1",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.2",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.3",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.4",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.5",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.6",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.7",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.8",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.9",
    "model.diffusion_model.middle_block.1.transformer_blocks.0",
    "model.diffusion_model.middle_block.1.transformer_blocks.1",
    "model.diffusion_model.middle_block.1.transformer_blocks.2",
    "model.diffusion_model.middle_block.1.transformer_blocks.3",
    "model.diffusion_model.middle_block.1.transformer_blocks.4",
    "model.diffusion_model.middle_block.1.transformer_blocks.5",
    "model.diffusion_model.middle_block.1.transformer_blocks.6",
    "model.diffusion_model.middle_block.1.transformer_blocks.7",
    "model.diffusion_model.middle_block.1.transformer_blocks.8",
    "model.diffusion_model.middle_block.1.transformer_blocks.9",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.0",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.1",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.2",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.3",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.4",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.5",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.6",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.7",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.8",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.9",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.0",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.1",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.2",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.3",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.4",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.5",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.6",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.7",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.8",
    "model.diffusion_model.output_blocks.1.1.transformer_blocks.9",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.0",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.1",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.2",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.3",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.4",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.5",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.6",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.7",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.8",
    "model.diffusion_model.output_blocks.2.1.transformer_blocks.9",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.1",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.1",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.1",
];

impl KeyMap for SdxlKeyMap {
    fn block_count() -> usize {
        BASES.len()
    }
    fn keys_for_block(_i: usize) -> &'static [&'static str] {
        &[]
    }
    fn keys_for_head() -> &'static [&'static str] {
        &[]
    }
}

/// Canonical tensor order per transformer block.
#[derive(Clone, Copy)]
enum KeyLoc {
    /// Key lives under the transformer block prefix (`{BASES[i]}`).
    Block(&'static str),
    /// Key is attached to the parent module (strip trailing `.transformer_blocks.*`).
    Parent(&'static str),
}

const BLOCK_SPECS: &[KeyLoc] = &[
    KeyLoc::Parent(".norm.weight"),
    KeyLoc::Parent(".norm.bias"),
    KeyLoc::Parent(".proj_in.weight"),
    KeyLoc::Parent(".proj_in.bias"),
    KeyLoc::Block(".norm1.weight"),
    KeyLoc::Block(".norm1.bias"),
    KeyLoc::Block(".attn1.to_q.weight"),
    KeyLoc::Block(".attn1.to_k.weight"),
    KeyLoc::Block(".attn1.to_v.weight"),
    KeyLoc::Block(".attn1.to_out.0.weight"),
    KeyLoc::Block(".attn1.to_out.0.bias"),
    KeyLoc::Block(".norm2.weight"),
    KeyLoc::Block(".norm2.bias"),
    KeyLoc::Block(".attn2.to_q.weight"),
    KeyLoc::Block(".attn2.to_k.weight"),
    KeyLoc::Block(".attn2.to_v.weight"),
    KeyLoc::Block(".attn2.to_out.0.weight"),
    KeyLoc::Block(".attn2.to_out.0.bias"),
    KeyLoc::Block(".norm3.weight"),
    KeyLoc::Block(".norm3.bias"),
    KeyLoc::Block(".ff.net.0.proj.weight"),
    KeyLoc::Block(".ff.net.0.proj.bias"),
    KeyLoc::Block(".ff.net.2.weight"),
    KeyLoc::Block(".ff.net.2.bias"),
    KeyLoc::Parent(".proj_out.weight"),
    KeyLoc::Parent(".proj_out.bias"),
];

impl KeyMapOwned for SdxlKeyMap {
    fn gen_keys_for_block(i: usize) -> Vec<String> {
        let base = BASES[i];
        let parent =
            base.rsplit_once(".transformer_blocks.").map(|(prefix, _)| prefix).unwrap_or(base);

        BLOCK_SPECS
            .iter()
            .map(|spec| match spec {
                KeyLoc::Block(suffix) => format!("{base}{suffix}"),
                KeyLoc::Parent(suffix) => format!("{parent}{suffix}"),
            })
            .collect()
    }
}
