use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, InferenceError>;

pub fn msg(msg: impl Into<String>) -> InferenceError {
    InferenceError::Message(msg.into())
}
