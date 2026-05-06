use flame_core::{self, DType};
use std::path::Path;

#[test]
fn inspect_ernie_keys() {
    let device = flame_core::global_cuda_device();
    let dir = Path::new("/home/alex/models/ERNIE-Image/transformer");
    for shard in &["diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"] {
        let path = dir.join(shard);
        let weights = flame_core::serialization::load_file(&path, &device).unwrap();
        println!("=== {} ({} tensors) ===", shard, weights.len());
        // Show top-level keys only (not layers.*)
        let tops: Vec<_> = weights.keys().filter(|k| !k.starts_with("layers.")).collect();
        for k in &tops {
            if let Some(t) = weights.get(*k) {
                println!("  {} -> {:?}", k, t.shape().dims());
            }
        }
        // Also show key count per prefix
        use std::collections::BTreeMap;
        let mut prefix_counts: BTreeMap<String, usize> = BTreeMap::new();
        for k in weights.keys() {
            let parts: Vec<&str> = k.splitn(2, '.').collect();
            let prefix = if parts.len() > 1 && parts[0].starts_with("layers") {
                format!("{}.*", parts[0])
            } else {
                parts[0].to_string()
            };
            *prefix_counts.entry(prefix).or_default() += 1;
        }
        println!("  Key prefixes:");
        for (p, c) in &prefix_counts {
            println!("    {}: {}", p, c);
        }
    }
}
