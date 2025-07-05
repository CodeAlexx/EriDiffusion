//! Error types for eridiffusion

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Network adapter error: {0}")]
    Network(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Plugin error: {0}")]
    Plugin(String),
    
    #[error("Device error: {0}")]
    Device(String),
    
    #[error("Tensor operation error: {0}")]
    TensorOp(#[from] candle_core::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("IO error: {0}")]
    IoError(String),
    
    #[error("Tensor error: {0}")]
    TensorError(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Invalid architecture: {0}")]
    InvalidArchitecture(String),
    
    #[error("Unsupported feature: {0}")]
    Unsupported(String),
    
    #[error("Runtime error: {0}")]
    Runtime(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Tensor error: {0}")]
    Tensor(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Data error: {0}")]
    DataError(String),
    
    #[error("Invalid shape: {0}")]
    InvalidShape(String),
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("Conversion error: {0}")]
    Conversion(String),
    
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(err: safetensors::SafeTensorError) -> Self {
        Error::SafeTensors(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Extension trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn context(self, msg: &str) -> Result<T>;
    
    /// Add context with a closure
    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T>;
}

impl<T, E: Into<Error>> ErrorContext<T> for std::result::Result<T, E> {
    fn context(self, msg: &str) -> Result<T> {
        self.map_err(|e| {
            let base_err = e.into();
            Error::Runtime(format!("{}: {}", msg, base_err))
        })
    }
    
    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T> {
        self.map_err(|e| {
            let base_err = e.into();
            Error::Runtime(format!("{}: {}", f(), base_err))
        })
    }
}

/// Macro for adding context to errors
#[macro_export]
macro_rules! context {
    ($result:expr, $msg:literal) => {
        $result.context($msg)
    };
    ($result:expr, $fmt:literal, $($arg:tt)*) => {
        $result.with_context(|| format!($fmt, $($arg)*))
    };
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::Runtime(err.to_string())
    }
}