use std::sync::Arc;
use flame_core::{Device as FlameDevice, Tensor};
use eridiffusion_common_weights::strict_loader::{StrictMmapLoader, tensor_from_bytes};
use crate::streaming::{DeviceWeights, WeightProvider, KeyMap, KeyMapOwned};
use std::collections::HashMap;
use anyhow::anyhow;

#[derive(Copy, Clone, Debug)]
enum QkvSlice { Q, K, V }

/// Map logical q/k/v keys to the physical fused qkv key.
fn resolve_qkv_key(key: &str) -> Option<(String, QkvSlice)> {
    if key.ends_with(".img_attn.q.weight") {
        return Some((key.replace(".img_attn.q.weight", ".img_attn.qkv.weight"), QkvSlice::Q));
    }
    if key.ends_with(".img_attn.k.weight") {
        return Some((key.replace(".img_attn.k.weight", ".img_attn.qkv.weight"), QkvSlice::K));
    }
    if key.ends_with(".img_attn.v.weight") {
        return Some((key.replace(".img_attn.v.weight", ".img_attn.qkv.weight"), QkvSlice::V));
    }
    // Flux text branch support (not used by Chroma, harmless for others)
    if key.ends_with(".txt_attn.q.weight") {
        return Some((key.replace(".txt_attn.q.weight", ".txt_attn.qkv.weight"), QkvSlice::Q));
    }
    if key.ends_with(".txt_attn.k.weight") {
        return Some((key.replace(".txt_attn.k.weight", ".txt_attn.qkv.weight"), QkvSlice::K));
    }
    if key.ends_with(".txt_attn.v.weight") {
        return Some((key.replace(".txt_attn.v.weight", ".txt_attn.qkv.weight"), QkvSlice::V));
    }
    // SD3.5 x_block naming uses `.attn.qkv.weight`
    if key.ends_with(".attn.q.weight") {
        return Some((key.replace(".attn.q.weight", ".attn.qkv.weight"), QkvSlice::Q));
    }
    if key.ends_with(".attn.k.weight") {
        return Some((key.replace(".attn.k.weight", ".attn.qkv.weight"), QkvSlice::K));
    }
    if key.ends_with(".attn.v.weight") {
        return Some((key.replace(".attn.v.weight", ".attn.qkv.weight"), QkvSlice::V));
    }
    None
}

/// Split a fused QKV weight into (Q,K,V). Supports [(3*D), D] or [D, (3*D)].
fn split_qkv(fused: &Tensor) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let dims = fused.shape().dims();
    let (d0, d1) = (dims[0], dims[1]);
    if d0 % 3 == 0 {
        let d = d0 / 3;
        let q = fused.narrow(0, 0 * d, d)?;
        let k = fused.narrow(0, 1 * d, d)?;
        let v = fused.narrow(0, 2 * d, d)?;
        return Ok((q, k, v));
    }
    if d1 % 3 == 0 {
        let d = d1 / 3;
        let q = fused.narrow(1, 0 * d, d)?;
        let k = fused.narrow(1, 1 * d, d)?;
        let v = fused.narrow(1, 2 * d, d)?;
        return Ok((q, k, v));
    }
    Err(anyhow!("QKV fused weight has incompatible shape: {:?}", dims.to_vec()))
}

pub struct MmapWeightProvider<K: KeyMap + Send + Sync + 'static> {
    pub ld: Arc<StrictMmapLoader>,
    pub dev: FlameDevice,
    _km: std::marker::PhantomData<K>,
}

impl<K: KeyMap + Send + Sync + 'static> MmapWeightProvider<K> {
    pub fn new(ld: Arc<StrictMmapLoader>, dev: FlameDevice) -> Self {
        Self { ld, dev, _km: Default::default() }
    }
}

impl<K: KeyMap + KeyMapOwned + Send + Sync + 'static> WeightProvider for MmapWeightProvider<K> {
    fn load_block_to_gpu(&self, i: usize) -> anyhow::Result<DeviceWeights> {
        let weight_logs: bool = std::env::var("WEIGHT_LOGS").ok().map(|v| v != "0").unwrap_or(false);
        // Cache fused qkv tensor per physical key so we only load once.
        let mut qkv_cache: HashMap<String, (Tensor, Tensor, Tensor)> = HashMap::new();
        let mut out: Vec<Tensor> = Vec::new();
        for key in K::owned_keys_for_block(i) {
            if let Some((phys, which)) = resolve_qkv_key(&key) {
                let (q, k, v) = if let Some(trip) = qkv_cache.get(&phys) {
                    trip.clone()
                } else {
                    let ti = self.ld.info(&phys).map_err(anyhow::Error::from)?;
                    let b  = self.ld.bytes(&phys).map_err(anyhow::Error::from)?;
                    let fused0 = tensor_from_bytes(self.dev.clone(), &ti, b).map_err(anyhow::Error::from)?;
                    // Ensure base weights are not tracked by autograd
                    let fused = fused0.requires_grad_(false).detach().map_err(anyhow::Error::from)?;
                    let split = split_qkv(&fused)?;
                    if weight_logs || i == 0 {
                        println!(
                            "[weights] fused={} {:?} -> q={:?} k={:?} v={:?}",
                            phys,
                            fused.shape().dims().to_vec(),
                            split.0.shape().dims().to_vec(),
                            split.1.shape().dims().to_vec(),
                            split.2.shape().dims().to_vec()
                        );
                    }
                    qkv_cache.insert(phys.clone(), split.clone());
                    split
                };
                let t_raw = match which { QkvSlice::Q => q, QkvSlice::K => k, QkvSlice::V => v };
                // Detach slices too (views inherit detach state, but make explicit)
                let t = t_raw.requires_grad_(false).detach().map_err(anyhow::Error::from)?;
                if (weight_logs || i == 0) && matches!(which, QkvSlice::Q | QkvSlice::K | QkvSlice::V) {
                    println!("[weights] key={} (slice {:?}) shape={:?}", key, which, t.shape().dims().to_vec());
                }
                out.push(t);
            } else {
                let ti = self.ld.info(&key).map_err(anyhow::Error::from)?;
                let b  = self.ld.bytes(&key).map_err(anyhow::Error::from)?;
                let t0 = tensor_from_bytes(self.dev.clone(), &ti, b).map_err(anyhow::Error::from)?;
                let t = t0.requires_grad_(false).detach().map_err(anyhow::Error::from)?;
                if weight_logs || i == 0 {
                    println!("[weights] key={} shape={:?}", key, t.shape().dims().to_vec());
                }
                out.push(t);
            }
        }
        Ok(DeviceWeights { tensors: out })
    }
    fn load_head_to_gpu(&self) -> anyhow::Result<DeviceWeights> { Ok(DeviceWeights { tensors: vec![] }) }
    fn prefetch_block(&self, _block_id: usize) -> anyhow::Result<()> { Ok(()) }
    fn release_block(&self, _block_id: isize) -> anyhow::Result<()> { Ok(()) }
}
