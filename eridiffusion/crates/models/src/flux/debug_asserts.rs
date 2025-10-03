#[macro_export]
macro_rules! assert_same_dtype {
    ($lhs:expr, $rhs:expr, $ctx:expr) => {
        debug_assert_eq!(
            $lhs.dtype(),
            $rhs.dtype(),
            "dtype mismatch in {}: lhs={:?} rhs={:?}",
            $ctx,
            $lhs.dtype(),
            $rhs.dtype()
        );
    };
}
