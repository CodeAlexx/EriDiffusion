use anyhow::Result;
use flame_core::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamId { Q, K, V, O, Fc1, Fc2, CrossQ, CrossK, CrossV, CrossO }

pub trait HasLoRA {
    fn lora_param_ids(&self) -> Vec<ParamId> { vec![ParamId::Q, ParamId::K, ParamId::V, ParamId::O, ParamId::Fc1, ParamId::Fc2] }
    fn enable_lora_grads(&mut self) {}
}

