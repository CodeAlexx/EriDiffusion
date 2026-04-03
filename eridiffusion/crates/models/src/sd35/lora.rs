pub const LORA_TARGETS: &[&str] = &[
    "attn.q","attn.k","attn.v","attn.o","mlp.fc1","mlp.fc2",
];

pub trait HasLoRA { fn lora_targets(&self) -> &'static [&'static str] { LORA_TARGETS } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn has_targets() {
        assert!(LORA_TARGETS.len() >= 6);
    }
}
