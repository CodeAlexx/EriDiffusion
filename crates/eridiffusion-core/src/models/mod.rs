use flame_core::{parameter::Parameter, Tensor};
use crate::Result;

pub mod acestep; pub mod chroma; pub mod flux; pub mod ernie; pub mod klein; pub mod qwenimage; pub mod sdxl; pub mod sd35; pub mod zimage; pub mod ltx2; pub mod anima; pub mod wan22; pub mod wan22_fwd;
pub mod sensenova_u1;
pub use acestep::AceStepLoRAModel;
pub use chroma::ChromaTrainingModel;
pub use qwenimage::QwenImageTrainingModel;
pub use flux::FluxModel; pub use ernie::ErnieModel; pub use klein::KleinModel;
pub use sdxl::SDXLModel; pub use sd35::SD35Model; pub use zimage::ZImageModel;
pub use ltx2::Ltx2Model; pub use anima::AnimaModel;
pub use wan22::{Wan22Config, Wan22Model, Wan22Variant, Wan22LoraBundle, LoraTarget as Wan22LoraTarget};
pub use sensenova_u1::{SenseNovaU1, SenseNovaU1Config};

pub trait TrainableModel: Send + Sync {
    /// `&mut self` so impls can do per-layer weight streaming (BlockOffloader)
    /// inside the forward pass without resorting to interior mutability.
    fn forward(&mut self, noisy: &Tensor, timestep: &Tensor, context: &[Tensor], pooled: Option<&Tensor>) -> Result<Tensor>;
    fn parameters(&self) -> Vec<Parameter>;
    fn post_optimizer_step(&mut self);
    fn save_weights(&self, path: &str) -> Result<()>;
    fn load_weights(&mut self, path: &str) -> Result<()>;
}
