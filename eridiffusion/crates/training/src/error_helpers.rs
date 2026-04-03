//! Helpers to bridge flame_core/eridiffusion errors with anyhow while we migrate APIs.

use anyhow::{anyhow, Result as AnyResult};
use eridiffusion_core::{Error as EriError, Result as EriResult};

/// Convert an `eridiffusion_core::Result<T>` into `anyhow::Result<T>` with context.
pub fn flame_to_anyhow<T>(r: EriResult<T>, ctx: &'static str) -> AnyResult<T> {
    r.map_err(|e| anyhow!("[{ctx}] {e}"))
}

/// Execute a closure returning `eridiffusion_core::Result<T>` and wrap failures with context.
pub fn with_ctx<T, F: FnOnce() -> EriResult<T>>(f: F, ctx: &'static str) -> AnyResult<T> {
    f().map_err(|e| anyhow!("[{ctx}] {e}"))
}

/// Convert `anyhow::Result<T>` back into the core error type (use sparingly).
pub fn anyhow_to_flame<T>(r: AnyResult<T>, code: EriError) -> EriResult<T> {
    match r {
        Ok(v) => Ok(v),
        Err(_) => Err(code),
    }
}

/// Helper macro to cut down on repetitive `.map_err` when calling Core APIs from anyhow code.
#[macro_export]
macro_rules! flame_ctx {
    ($expr:expr, $ctx:expr) => {
        $expr.map_err(|e| anyhow::anyhow!(concat!("[", $ctx, "] ", "{}"), e))
    };
}
