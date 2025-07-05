//! SD 3.5 trainer module

// Re-export the implementation
pub use self::sd35_trainer_impl::*;

// Include the implementation
#[path = "sd35_trainer_impl.rs"]
mod sd35_trainer_impl;

// Include tests when in test mode
#[cfg(test)]
#[path = "sd35_trainer_test.rs"]
mod tests;