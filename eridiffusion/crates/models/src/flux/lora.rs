/// LoRA target map for Flux DiT: q,k,v,o,fc1,fc2.

pub const LORA_TARGETS: &[&str] = &["attn.q", "attn.k", "attn.v", "attn.o", "mlp.fc1", "mlp.fc2"];

/// Minimal trait to expose LoRA target collection and base freeze behavior.
pub trait HasLoRA {
    /// Stable key list for adapter placement
    fn lora_targets(&self) -> &'static [&'static str] {
        LORA_TARGETS
    }
    /// Freeze base weights (adapters remain trainable)
    fn freeze_base(&mut self);
    fn is_base_frozen(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flux::{FluxConfig, FluxModel};
    use flame_core::{DType, Device};

    #[test]
    fn lora_targets_present_and_freeze_base() -> anyhow::Result<()> {
        let dev = Device::cuda(0)?;
        let cfg = FluxConfig {
            hidden: 32,
            heads: 4,
            layers: 1,
            param_dtype: DType::BF16,
            matmul_policy: crate::flux::dtype_policy::MatmulDTypePolicy::MatchParams,
        };
        let mut model = FluxModel::new(cfg, dev, DType::BF16)?;
        // Collect targets
        let keys = model.lora_targets();
        assert!(keys.len() > 0);
        // Freeze base
        model.freeze_base();
        assert!(model.is_base_frozen());
        Ok(())
    }
}

// Implement the trait for our FluxModel
impl HasLoRA for crate::flux::FluxModel {
    fn freeze_base(&mut self) {
        crate::flux::FluxModel::freeze_base(self)
    }
    fn is_base_frozen(&self) -> bool {
        crate::flux::FluxModel::is_base_frozen(self)
    }
}
