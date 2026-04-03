//! Data type definitions aligned with FLAME

// Re-export FLAME's DType to avoid duplicate enums and mismatches in tests/FFI.
pub use flame_core::DType;

// Serde helpers for external DType (serialize as canonical string like "f32")
pub mod dtype_serde {
    use super::DType;
    use serde::{Serializer, Deserializer, Deserialize};

    pub fn serialize<S>(dt: &DType, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(dt.as_str())
    }

    pub fn deserialize<'de, DSer>(deserializer: DSer) -> Result<DType, DSer::Error>
    where
        DSer: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "f32" | "F32" => Ok(DType::F32),
            "f16" | "F16" => Ok(DType::F16),
            "bf16" | "BF16" => Ok(DType::BF16),
            "f64" | "F64" => Ok(DType::F64),
            "i32" | "I32" => Ok(DType::I32),
            "i64" | "I64" => Ok(DType::I64),
            "u8" | "U8" => Ok(DType::U8),
            "u32" | "U32" => Ok(DType::U32),
            "i8" | "I8" => Ok(DType::I8),
            "bool" | "Bool" => Ok(DType::Bool),
            other => Err(serde::de::Error::custom(format!("invalid dtype: {}", other))),
        }
    }
}

use crate::Result;
use flame_core::Tensor;

/// Default casting policy for parameters: store as BF16.
pub fn cast_params_default(t: &Tensor) -> Result<Tensor> {
    Ok(t.to_dtype(DType::BF16)?)
}

/// Reduce or cast values to FP32 for numerically stable reductions.
pub fn reduce_to_fp32(t: &Tensor) -> Result<Tensor> {
    Ok(t.to_dtype(DType::F32)?)
}
