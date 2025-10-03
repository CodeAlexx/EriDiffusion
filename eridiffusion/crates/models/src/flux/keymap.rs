/// Strict mapping for Flux weight keys and expected layout.
/// Convs use [KH,KW,IC,OC]; Linears use [IN,OUT].

pub const ATTN_Q: &str = "attn.q.weight";
pub const ATTN_K: &str = "attn.k.weight";
pub const ATTN_V: &str = "attn.v.weight";
pub const ATTN_O: &str = "attn.o.weight";
pub const MLP_FC1: &str = "mlp.fc1.weight";
pub const MLP_FC2: &str = "mlp.fc2.weight";

#[derive(Clone, Debug)]
pub enum WeightKind {
    Linear(usize, usize),
    Conv2d(usize, usize, usize, usize),
}

/// Minimal key→meta map stub; in a real loader this would read safetensors headers and validate shapes.
pub fn expected_shape_for(key: &str) -> Option<WeightKind> {
    match key {
        ATTN_Q | ATTN_K | ATTN_V | ATTN_O => Some(WeightKind::Linear(0, 0)), // unknown until file read
        MLP_FC1 | MLP_FC2 => Some(WeightKind::Linear(0, 0)),
        _ => None,
    }
}
