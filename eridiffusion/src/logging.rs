use flame_core::device::Device;
use flame_core::GradientMap;
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
// backward is a method on Tensor, not a standalone function
use flame_core::optimizers::{Adam, SGD};

// Logging utilities for eridiffusion

/// Initialize logging with the given verbosity level
pub fn init_logger() -> flame_core::Result<()> {
    // Use env_logger with a custom format
    env_logger::Builder::from_default_env()
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .format_module_path(false)
        .format_target(false)
        .init();
    Ok(())
}

/// Convenience macros for common logging patterns

#[macro_export]
macro_rules! log_info {
($($arg:tt)*) => {
log::info!($($arg)*);
};
}

#[macro_export]
macro_rules! log_debug {
($($arg:tt)*) => {
log::debug!($($arg)*);
};
}

#[macro_export]
macro_rules! log_warn {
($($arg:tt)*) => {
log::warn!($($arg)*);
};
}

#[macro_export]
macro_rules! log_error {
($($arg:tt)*) => {
log::error!($($arg)*);
};
}

#[macro_export]
macro_rules! log_trace {
($($arg:tt)*) => {
log::trace!($($arg)*);
};
}

// Log levels guide:
// - ERROR: Critical failures that prevent operation
// - WARN: Issues that may affect results but don't stop execution
// - INFO: Important status updates (model loading, training progress)
// - DEBUG: Detailed information for debugging (tensor shapes, memory stats)
// - TRACE: Very detailed information (individual operations)
