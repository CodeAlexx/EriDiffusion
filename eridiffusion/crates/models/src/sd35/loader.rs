use anyhow::Result;
use flame_core::{Device, DType};
use eridiffusion_common_weights::strict_loader::{StrictMmapLoader, tensor_from_bytes};
use super::backbone::{Sd35Backbone, Sd35WeightPack};

/// Strictly load SD3.5 model and return a backbone model.
pub fn from_safetensors_strict_sd35(path: &std::path::Path, device: Device, dtype: DType, hidden: usize, heads: usize, layers: usize) -> Result<Sd35Backbone> {
    let mut ld = StrictMmapLoader::open(path)?;
    let mut packs: Vec<Sd35WeightPack> = Vec::with_capacity(layers);
    for i in 0..layers {
        let prefix = format!("block{}.", i);
        let keys = [
            format!("{}attn.q.weight", prefix),
            format!("{}attn.k.weight", prefix),
            format!("{}attn.v.weight", prefix),
            format!("{}attn.o.weight", prefix),
            format!("{}mlp.fc1.weight", prefix),
            format!("{}mlp.fc2.weight", prefix),
        ];
        let info_q = ld.info(&keys[0])?; let wq = tensor_from_bytes(device.clone(), &info_q, ld.bytes(&keys[0])?)?;
        let info_k = ld.info(&keys[1])?; let wk = tensor_from_bytes(device.clone(), &info_k, ld.bytes(&keys[1])?)?;
        let info_v = ld.info(&keys[2])?; let wv = tensor_from_bytes(device.clone(), &info_v, ld.bytes(&keys[2])?)?;
        let info_o = ld.info(&keys[3])?; let wo = tensor_from_bytes(device.clone(), &info_o, ld.bytes(&keys[3])?)?;
        let info_f1 = ld.info(&keys[4])?; let fc1 = tensor_from_bytes(device.clone(), &info_f1, ld.bytes(&keys[4])?)?;
        let info_f2 = ld.info(&keys[5])?; let fc2 = tensor_from_bytes(device.clone(), &info_f2, ld.bytes(&keys[5])?)?;

        // Validate shapes
        anyhow::ensure!(wq.shape().dims()==&[hidden, hidden], "q shape mismatch");
        anyhow::ensure!(wk.shape().dims()==&[hidden, hidden], "k shape mismatch");
        anyhow::ensure!(wv.shape().dims()==&[hidden, hidden], "v shape mismatch");
        anyhow::ensure!(wo.shape().dims()==&[hidden, hidden], "o shape mismatch");
        anyhow::ensure!(fc1.shape().dims()==&[hidden, hidden*4], "fc1 shape mismatch");
        anyhow::ensure!(fc2.shape().dims()==&[hidden*4, hidden], "fc2 shape mismatch");

        packs.push(Sd35WeightPack { wq, wk, wv, wo, fc1, fc2 });
        for k in &keys { ld.mark_used(k); }
    }
    ld.validate_used_exactly()?;
    Sd35Backbone::from_packs(device, dtype, hidden, heads, packs)
}

