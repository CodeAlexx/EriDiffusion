// Re-export CLIP components for compatibility
pub use super::text_encoder::CLIPTextEncoder as ClipTextTransformer;
pub use super::text_encoder::{CLIPConfig, CLIPTextEncoder, CLIPTextEncoderOutput};

// Additional exports that might be needed
pub use super::text_encoder_complete::CLIPConfig as Config;
