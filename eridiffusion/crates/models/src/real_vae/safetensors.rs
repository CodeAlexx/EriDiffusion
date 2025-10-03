
//! Minimal safetensors reader (same as earlier StrictWeights but local to real_vae).

#![allow(dead_code)]
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct Entry { pub dtype: String, pub shape: Vec<usize>, pub off: usize, pub end: usize }
pub struct StrictWeights { pub entries: BTreeMap<String, Entry>, pub blob: Vec<u8> }
impl StrictWeights {
    pub fn open(p: &Path) -> std::io::Result<Self> {
        let mut f = File::open(p)?;
        let mut len = [0u8;8]; f.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len) as usize;
        let mut hdr = vec![0u8; n]; f.read_exact(&mut hdr)?;
        let v: serde_json::Value = serde_json::from_slice(&hdr)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header"))?;
        let mut entries = BTreeMap::new();
        if let Some(obj) = v.as_object() {
            for (k,val) in obj {
                if k=="__metadata__" { continue; }
                let dtype = val.get("dtype").and_then(|x| x.as_str()).unwrap_or("F32").to_string();
                let shape = val.get("shape").and_then(|x| x.as_array()).unwrap_or(&vec![])
                    .iter().filter_map(|n| n.as_u64()).map(|n| n as usize).collect::<Vec<_>>();
                let offs = val.get("data_offsets").and_then(|x| x.as_array()).unwrap_or(&vec![]);
                if offs.len()!=2 { continue; }
                let off = offs[0].as_u64().unwrap_or(0) as usize;
                let end = offs[1].as_u64().unwrap_or(0) as usize;
                entries.insert(k.clone(), Entry { dtype, shape, off, end });
            }
        }
        let mut blob = Vec::new(); f.read_to_end(&mut blob)?;
        Ok(Self { entries, blob })
    }
    pub fn get_raw(&mut self, key: &str) -> std::io::Result<&[u8]> {
        let e = self.entries.get(key).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, key.to_string()))?;
        Ok(&self.blob[e.off..e.end])
    }
}
