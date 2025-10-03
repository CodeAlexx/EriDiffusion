use crate::streaming::{KeyMap, KeyMapOwned};

pub struct Sd35KeyMap;

impl KeyMap for Sd35KeyMap {
    // From sd3.5_large.safetensors dump: joint_blocks.0..=37
    fn block_count() -> usize { 38 }
    fn keys_for_block(_i: usize) -> &'static [&'static str] { &[] }
    fn keys_for_head() -> &'static [&'static str] { &[] }
}

impl KeyMapOwned for Sd35KeyMap {
    fn gen_keys_for_block(i: usize) -> Vec<String> {
        // Use x_block weights; attention is fused qkv in file.
        let base = format!("model.diffusion_model.joint_blocks.{i}.x_block");
        vec![
            format!("{base}.attn.q.weight"),     // logical → fused qkv
            format!("{base}.attn.k.weight"),     // logical → fused qkv
            format!("{base}.attn.v.weight"),     // logical → fused qkv
            format!("{base}.attn.proj.weight"),  // out proj
            format!("{base}.mlp.fc1.weight"),    // fc1
            format!("{base}.mlp.fc2.weight"),    // fc2
        ]
    }
}

