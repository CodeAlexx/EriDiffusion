use crate::streaming::{KeyMap, KeyMapOwned};

pub struct ChromaKeyMap;

impl KeyMap for ChromaKeyMap {
    fn block_count() -> usize {
        19
    } // double_blocks.{0..=18}
    /// Keep legacy static API empty so the owned path is used.
    fn keys_for_block(_i: usize) -> &'static [&'static str] {
        &[]
    }
    fn keys_for_head() -> &'static [&'static str] {
        &[]
    }
}

impl KeyMapOwned for ChromaKeyMap {
    fn gen_keys_for_block(i: usize) -> Vec<String> {
        // Logical keys: the loader maps q/k/v to fused qkv in the file and slices.
        let base = format!("double_blocks.{i}");
        vec![
            format!("{base}.img_attn.q.weight"),    // Q (logical)
            format!("{base}.img_attn.k.weight"),    // K (logical)
            format!("{base}.img_attn.v.weight"),    // V (logical)
            format!("{base}.img_attn.proj.weight"), // O
            format!("{base}.img_mlp.0.weight"),     // fc1
            format!("{base}.img_mlp.2.weight"),     // fc2
        ]
    }
}
