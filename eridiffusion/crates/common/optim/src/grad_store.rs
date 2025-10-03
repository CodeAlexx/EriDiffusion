use anyhow::Result;
use hashbrown::HashMap;
use flame_core::Tensor;
use eridiffusion_common_weights::ParamId;

/// Minimal gradient store keyed by ParamId.
pub struct GradStore {
    map: HashMap<ParamId, Tensor>,
}

impl GradStore {
    pub fn new() -> Self { Self { map: HashMap::new() } }
    pub fn set(&mut self, id: ParamId, g: Tensor) { self.map.insert(id, g); }
    pub fn get(&self, id: &ParamId) -> Option<&Tensor> { self.map.get(id) }
    pub fn zero(&mut self, ids: &[ParamId]) -> Result<()> {
        for id in ids {
            if let Some(t) = self.map.get_mut(id) {
                let z = t.zeros_like()?;
                *t = z;
            }
        }
        Ok(())
    }
}

