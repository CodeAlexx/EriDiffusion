
//! Strict checkpoint I/O for LoRA and optimizer states.
//! Keeps metadata in safetensors header and enforces exact-key usage.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CkptMetadata {
    pub format_version: u32,
    pub trainer: String,
    pub model: String,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub step: u64,
}

#[derive(Clone, Debug)]
pub struct CkptWriter {
    file: PathBuf,
    header: serde_json::Map<String, serde_json::Value>,
    buffers: Vec<(String, Vec<u8>)>,
    offsets: BTreeMap<String, (usize, usize)>, // (start, end)
}

impl CkptWriter {
    pub fn new(file: &Path, meta: &CkptMetadata) -> Self {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::to_value(meta).unwrap_or(serde_json::json!({})));
        Self { file: file.to_path_buf(), header, buffers: Vec::new(), offsets: BTreeMap::new() }
    }

    /// Add a named tensor buffer. Caller guarantees dtype/shape; we just store bytes with offsets.
    pub fn add_tensor(&mut self, name: &str, shape: &[usize], dtype: &str, buf: Vec<u8>) {
        let start = self.buffers.iter().map(|(_, b)| b.len()).sum::<usize>();
        let end = start + buf.len();
        self.buffers.push((name.to_string(), buf));
        self.offsets.insert(name.to_string(), (start, end));

        let mut m = serde_json::Map::new();
        m.insert("dtype".into(), serde_json::Value::String(dtype.to_string()));
        m.insert("shape".into(), serde_json::json!(shape));
        m.insert("data_offsets".into(), serde_json::json!([start, end]));
        self.header.insert(name.to_string(), serde_json::Value::Object(m));
    }

    pub fn finish(self) -> std::io::Result<()> {
        let mut f = File::create(&self.file)?;
        let header_str = serde_json::to_string(&serde_json::Value::Object(self.header)).unwrap_or("{}".to_string());
        let header_bytes = header_str.as_bytes();
        let mut len = (header_bytes.len() as u64).to_le_bytes().to_vec();
        f.write_all(&len)?;
        f.write_all(header_bytes)?;
        for (_, buf) in self.buffers {
            f.write_all(&buf)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum CkptError {
    Io(std::io::Error),
    Missing(String),
    Unexpected(String),
}

impl From<std::io::Error> for CkptError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

pub struct CkptReader {
    pub header: serde_json::Map<String, serde_json::Value>,
    pub blobs: Vec<u8>,
    used: BTreeSet<String>,
}

impl CkptReader {
    pub fn open(file: &Path) -> Result<Self, CkptError> {
        use std::io::Read;
        let mut f = File::open(file)?;
        let mut hdr_len_bytes = [0u8; 8];
        f.read_exact(&mut hdr_len_bytes)?;
        let hdr_len = u64::from_le_bytes(hdr_len_bytes) as usize;
        let mut hdr = vec![0u8; hdr_len];
        f.read_exact(&mut hdr)?;
        let header_json: serde_json::Value = serde_json::from_slice(&hdr)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header"))?;
        let header = header_json.as_object().cloned().unwrap_or_default();
        let mut blobs = Vec::new();
        f.read_to_end(&mut blobs)?;
        Ok(Self { header, blobs, used: BTreeSet::new() })
    }

    pub fn get_tensor(&mut self, name: &str) -> Result<&[u8], CkptError> {
        use std::io::ErrorKind;
        let entry = self.header.get(name).ok_or_else(|| CkptError::Missing(name.to_string()))?;
        let start = entry.get("data_offsets").and_then(|v| v.as_array()).and_then(|a| a.get(0)).and_then(|x| x.as_u64()).ok_or_else(|| CkptError::Missing(format!("offset:{name}")))? as usize;
        let end = entry.get("data_offsets").and_then(|v| v.as_array()).and_then(|a| a.get(1)).and_then(|x| x.as_u64()).ok_or_else(|| CkptError::Missing(format!("end:{name}")))? as usize;
        if end > self.blobs.len() { return Err(CkptError::Io(std::io::Error::new(ErrorKind::UnexpectedEof, "blob OOB"))); }
        self.used.insert(name.to_string());
        Ok(&self.blobs[start..end])
    }

    pub fn validate_used_exactly(&self, expected: &BTreeSet<String>) -> Result<(), CkptError> {
        for e in expected {
            if !self.used.contains(e) { return Err(CkptError::Missing(e.clone())); }
        }
        for u in &self.used {
            if !expected.contains(u) { return Err(CkptError::Unexpected(u.clone())); }
        }
        Ok(())
    }
}
