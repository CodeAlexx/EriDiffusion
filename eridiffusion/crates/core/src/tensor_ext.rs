//! Dimension helpers to adapt to common dimsN() usage across code.
//!
//! Provides a small trait `DimExt` that exposes dims(), dims1..dims4() in a
//! safe, panicking-on-programmer-error way. This avoids scattered shape indexing
//! and normalizes call sites that previously relied on non-existent dimsN() helpers.

use flame_core::Tensor as FlameTensor;

/// Extension trait offering convenient dimension accessors.
pub trait DimExt {
    /// Borrow the concrete dimension slice.
    fn dims(&self) -> &[usize];

    #[inline]
    fn dims1(&self) -> [usize; 1] { self.dims().try_into().expect("expected 1D tensor") }
    #[inline]
    fn dims2(&self) -> [usize; 2] { self.dims().try_into().expect("expected 2D tensor") }
    #[inline]
    fn dims3(&self) -> [usize; 3] { self.dims().try_into().expect("expected 3D tensor") }
    #[inline]
    fn dims4(&self) -> [usize; 4] { self.dims().try_into().expect("expected 4D tensor") }
}

impl DimExt for FlameTensor {
    #[inline]
    fn dims(&self) -> &[usize] { self.shape().dims() }
}

impl DimExt for crate::tensor::TensorView {
    #[inline]
    fn dims(&self) -> &[usize] { self.shape().dims() }
}

