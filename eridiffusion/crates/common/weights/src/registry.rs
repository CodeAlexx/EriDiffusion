use anyhow::{Result, bail};
use hashbrown::HashMap;
use serde::Serialize;
use flame_core::{Tensor, DType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub u64);

pub struct ParamRegistry {
    map: HashMap<String, (ParamId, Tensor)>,
    next_id: u64,
}

impl ParamRegistry {
    pub fn new() -> Self { Self { map: HashMap::new(), next_id: 1 } }

    pub fn insert(&mut self, name: &str, t: Tensor) -> ParamId {
        let id = ParamId(self.next_id);
        self.next_id += 1;
        self.map.insert(name.to_string(), (id, t));
        id
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.map.get(name).map(|(_, t)| t)
    }

    pub fn id_of(&self, name: &str) -> Option<ParamId> {
        self.map.get(name).map(|(id, _)| *id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, (&ParamId, &Tensor))> {
        self.map.iter().map(|(k, v)| (k.as_str(), (&v.0, &v.1)))
    }

    /// Default grad policy: off. Enable only given IDs.
    pub fn set_requires_grad(&mut self, ids: &[ParamId], on: bool) {
        use std::collections::HashSet;
        let idset: HashSet<ParamId> = ids.iter().copied().collect();
        for (_k, (pid, t)) in self.map.iter_mut() {
            let want = idset.contains(pid) && on;
            let cloned = t.clone();
            *t = cloned.requires_grad_(want);
        }
    }

    /// Get parameter by id (immutable)
    pub fn get_by_id(&self, id: ParamId) -> Option<&Tensor> {
        self.map.values().find_map(|(pid, t)| if *pid == id { Some(t) } else { None })
    }

    /// Get parameter by id (mutable)
    pub fn get_mut_by_id(&mut self, id: ParamId) -> Option<&mut Tensor> {
        // HashMap API doesn't provide a direct way; do a simple scan
        for (_k, (pid, t)) in self.map.iter_mut() {
            if *pid == id { return Some(t); }
        }
        None
    }

    /// Replace the tensor for a given ParamId
    pub fn assign(&mut self, id: ParamId, new_t: Tensor) -> anyhow::Result<()> {
        for (_k, (pid, t)) in self.map.iter_mut() {
            if *pid == id { *t = new_t; return Ok(()); }
        }
        anyhow::bail!("ParamId {:?} not found for assign()", id)
    }

    /// Remove a parameter by name (used to release block-swapped layers)
    pub fn remove(&mut self, name: &str) -> bool { self.map.remove(name).is_some() }

    pub fn emit_json_summary(&self) -> Result<String> {
        #[derive(Serialize)]
        struct Summary {
            count: usize,
            bytes: usize,
            dtype_hist: std::collections::BTreeMap<String, usize>,
            top_prefixes: std::collections::BTreeMap<String, usize>,
        }
        let mut bytes = 0usize;
        let mut dh: std::collections::BTreeMap<String, usize> = Default::default();
        let mut pref: std::collections::BTreeMap<String, usize> = Default::default();
        for (name, (_id, t)) in &self.map {
            let dt = match t.dtype() { DType::F16 => 2, DType::BF16 => 2, _ => 4 };
            bytes += t.shape().elem_count() * dt;
            *dh.entry(format!("{:?}", t.dtype())).or_default() += 1;
            let p = name.split('.').next().unwrap_or("").to_string();
            *pref.entry(p).or_default() += 1;
        }
        let s = Summary { count: self.map.len(), bytes, dtype_hist: dh, top_prefixes: pref };
        Ok(serde_json::to_string_pretty(&s)?)
    }
}

#[cfg(feature = "nvml")]
pub fn vram_used_gb() -> Result<f32> {
    let nvml = nvml_wrapper::NVML::init()?;
    let dev = nvml.device_by_index(0)?;
    let mem = dev.memory_info()?;
    Ok(mem.used as f32 / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(not(feature = "nvml"))]
pub fn vram_used_gb() -> Result<f32> {
    anyhow::bail!("NVML feature not enabled for vram_used_gb")
}

pub fn assert_vram_below(limit_gb: f32) -> Result<()> {
    let used = vram_used_gb()?;
    if used > limit_gb {
        bail!("VRAM used {:.2} GB exceeds limit {:.2} GB", used, limit_gb);
    }
    Ok(())
}
