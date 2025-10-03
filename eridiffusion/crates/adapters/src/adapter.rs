use flame_core::Tensor;
use hashbrown::HashMap;
use std::sync::Arc;

pub trait Adapter {
    fn name(&self) -> &str;
    fn scale(&self) -> f32;
    fn params(&self) -> Vec<Tensor>;
    fn apply_linear(&self, base_w: &Tensor, x: &Tensor) -> eridiffusion_core::Result<Tensor>;
    fn apply_conv2d(&self, base_w: &Tensor, x: &Tensor, stride: (usize,usize), pad: (usize,usize)) -> eridiffusion_core::Result<Tensor>;
    fn delta_weight_linear(&self, dtype: eridiffusion_core::DType) -> eridiffusion_core::Result<Tensor>;
    fn delta_out_linear(&self, x: &Tensor) -> eridiffusion_core::Result<Tensor>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterKind { LoRA, LoCon2d, LoHa, LoKr, DoRA, IA3 }

#[derive(thiserror::Error, Debug)]
pub enum AdapterError {
    #[error("shape mismatch for {target}: expected {expected:?}, got {got:?}")]
    Shape { target: String, expected: Vec<i64>, got: Vec<i64> },
    #[error("missing key {key} for {target}")]
    Missing { key: String, target: String },
    #[error("unused keys present: {0:?}")]
    Unused(Vec<String>),
    #[error("unsupported adapter kind for module {target}: {kind:?}")]
    Unsupported { target: String, kind: AdapterKind },
}

pub struct AdapterSet {
    pub by_target: HashMap<String, Arc<dyn Adapter + Send + Sync>>,
}

impl AdapterSet {
    pub fn new() -> Self { Self { by_target: HashMap::new() } }
    pub fn get(&self, target: &str) -> Option<&(dyn Adapter + Send + Sync)> {
        self.by_target.get(target).map(|a| a.as_ref())
    }
    pub fn params(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for a in self.by_target.values() { out.extend(a.params()); }
        out
    }
    pub fn insert(&mut self, target: String, a: Arc<dyn Adapter + Send + Sync>) {
        self.by_target.insert(target, a);
    }
}
