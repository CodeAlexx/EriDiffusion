use flame_core::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulDTypePolicy {
    /// Use parameter storage dtype (default BF16).
    MatchParams,
    /// Upcast activations/weights to FP32 before matmul.
    ForceFP32,
}

impl Default for MatmulDTypePolicy {
    fn default() -> Self {
        Self::MatchParams
    }
}

#[inline]
pub fn resolve_compute_dtype(param_dtype: DType, policy: MatmulDTypePolicy) -> DType {
    match policy {
        MatmulDTypePolicy::MatchParams => param_dtype,
        MatmulDTypePolicy::ForceFP32 => DType::F32,
    }
}
