//! Data type definitions and conversions

use candle_core::DType as CandleDType;
use serde::{Deserialize, Serialize};

/// Data type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// Brain floating point
    BF16,
    /// 64-bit floating point
    F64,
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 32-bit unsigned integer
    U32,
}

impl DType {
    /// Convert to Candle dtype
    pub fn to_candle(&self) -> CandleDType {
        match self {
            DType::F32 => CandleDType::F32,
            DType::F16 => CandleDType::F16,
            DType::BF16 => CandleDType::BF16,
            DType::F64 => CandleDType::F64,
            DType::I32 => CandleDType::I64, // I32 not available in candle 0.9, use I64
            DType::I64 => CandleDType::I64,
            DType::U8 => CandleDType::U8,
            DType::U32 => CandleDType::U32,
        }
    }
    
    /// Get size in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 | DType::I64 => 8,
            DType::U8 => 1,
        }
    }
    
    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16 | DType::F64)
    }
    
    /// Check if this is an integer type
    pub fn is_int(&self) -> bool {
        matches!(self, DType::I32 | DType::I64 | DType::U8 | DType::U32)
    }
    
    /// Get the default dtype for computations
    pub fn default_float() -> Self {
        DType::F32
    }
    
    /// Get the preferred dtype for mixed precision training
    pub fn mixed_precision() -> (Self, Self) {
        (DType::F16, DType::F32) // Compute in F16, accumulate in F32
    }
}

impl From<CandleDType> for DType {
    fn from(dtype: CandleDType) -> Self {
        match dtype {
            CandleDType::F32 => DType::F32,
            CandleDType::F16 => DType::F16,
            CandleDType::BF16 => DType::BF16,
            CandleDType::F64 => DType::F64,
            CandleDType::I64 => DType::I64,
            CandleDType::U8 => DType::U8,
            CandleDType::U32 => DType::U32,
        }
    }
}

impl Default for DType {
    fn default() -> Self {
        DType::F32
    }
}