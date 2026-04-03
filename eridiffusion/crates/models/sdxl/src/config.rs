use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
pub struct SdxlConfig {
    pub in_ch: usize,        // 4
    pub base_ch: usize,      // 320
    pub ch_mult: [usize; 4], // [1,2,4,4]
    pub num_heads: usize,    // e.g. 20
    pub head_dim: usize,     // e.g. 64
    pub ctx1_dim: usize,     // from CLIP-L
    pub ctx2_dim: usize,     // from OpenCLIP-G
    pub seq: usize,          // tokenizer seq
    pub prediction_type: String, // "epsilon"
}

impl Default for SdxlConfig {
    fn default() -> Self {
        Self {
            in_ch: 4,
            base_ch: 320,
            ch_mult: [1,2,4,4],
            num_heads: 20,
            head_dim: 64,
            ctx1_dim: 768,
            ctx2_dim: 1280,
            seq: 64,
            prediction_type: "epsilon".into(),
        }
    }
}

