use anyhow::Result;
use tracing::info;
use eridiffusion_common_weights::{SafeLoader, ParamRegistry};

#[derive(Clone, Copy, Debug, Default)]
pub struct BlockSwapCfg {
    pub prefetch_ahead: usize,
    pub release_after: bool,
    pub async_prefetch: bool,
}

pub struct BlockSwapManager {
    cfg: BlockSwapCfg,
    plan_keys: Vec<String>,
    idx: usize,
    last_loaded: Option<Vec<String>>,
}

impl BlockSwapManager {
    pub fn new(cfg: BlockSwapCfg) -> Self {
        Self { cfg, plan_keys: Vec::new(), idx: 0, last_loaded: None }
    }

    pub fn plan(&mut self, layer_keys: Vec<String>) { self.plan_keys = layer_keys; self.idx = 0; }

    pub fn prefetch_next(&mut self, ld: &mut SafeLoader, reg: &mut ParamRegistry) -> Result<Vec<String>> {
        if self.idx >= self.plan_keys.len() { return Ok(vec![]); }
        let prefix = &self.plan_keys[self.idx];
        self.idx += 1;

        // Enumerate all keys for this layer prefix
        let keys = ld.list_keys()?;
        let mut loaded = Vec::new();
        for k in keys.into_iter().filter(|k| k.starts_with(prefix)) {
            // Load BF16 tensor to device and register, track key for release
            let t = ld.get_bf16(&k)?;
            reg.insert(&k, t);
            loaded.push(k);
        }
        self.last_loaded = Some(loaded.clone());
        Ok(loaded)
    }

    pub fn release_previous(&mut self, _reg: &mut ParamRegistry) -> Result<()> {
        // Free previously loaded tensors from registry
        if let Some(keys) = self.last_loaded.take() {
            for k in keys { let _ = _reg.remove(&k); }
        }
        Ok(())
    }
}
