use crate::streaming::{KeyMap, KeyMapOwned};

pub struct FluxKeyMap;

impl KeyMap for FluxKeyMap {
    fn block_count() -> usize {
        19
    } // double_blocks.{0..=18}
    fn keys_for_block(_i: usize) -> &'static [&'static str] {
        &[]
    }
    fn keys_for_head() -> &'static [&'static str] {
        &[]
    }
}

impl KeyMapOwned for FluxKeyMap {
    fn gen_keys_for_block(i: usize) -> Vec<String> {
        // Logical keys match Flux's fused qkv layout; loader will slice q/k/v.
        let base = format!("double_blocks.{i}");
        vec![
            format!("{base}.img_attn.q.weight"),    // Q (logical → fused qkv)
            format!("{base}.img_attn.k.weight"),    // K
            format!("{base}.img_attn.v.weight"),    // V
            format!("{base}.img_attn.proj.weight"), // O
            format!("{base}.img_mlp.0.weight"),     // fc1
            format!("{base}.img_mlp.2.weight"),     // fc2
        ]
    }
}
