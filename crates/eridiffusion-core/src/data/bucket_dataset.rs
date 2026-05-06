//! Latent dataset loader + aspect-ratio bucketing.
//!
//! Ported verbatim from flame-diffusion/src/dataset.rs (2026-05-05). Caller
//! owns per-model packing — concrete latent/text shapes are NOT enforced
//! beyond rank checks for bucket-key derivation.
//!
//! Public types:
//! - [`LatentDataset`] — list of cached `.safetensors` samples
//! - [`TrainSample`] — loaded sample (latent, text_embedding, text_mask) on GPU in BF16
//! - [`BucketKey`] / [`Bucket`] / [`BucketPlan`] — aspect-ratio grouping
//! - [`BucketBatchSampler`] — fixed-size, same-shape batches per epoch

use cudarc::driver::CudaDevice;
use flame_core::{serialization, DType, Error, Result, Tensor};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

// -----------------------------------------------------------------------------
// LatentDataset + TrainSample
// -----------------------------------------------------------------------------

pub struct LatentDataset {
    files: Vec<PathBuf>,
}

pub struct TrainSample {
    pub source: PathBuf,
    pub latent: Tensor,
    pub text_embedding: Tensor,
    pub text_mask: Tensor,
}

impl LatentDataset {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let entries = fs::read_dir(data_dir)
            .map_err(|e| Error::Io(format!("Failed to read {}: {e}", data_dir.display())))?;

        let mut files = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| Error::Io(format!("read_dir entry failed: {e}")))?;
            let path = entry.path();
            let is_safetensors = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("safetensors"))
                .unwrap_or(false);
            if is_safetensors {
                files.push(path);
            }
        }

        files.sort();
        if files.is_empty() {
            return Err(Error::InvalidInput(format!(
                "No .safetensors files found in {}",
                data_dir.display()
            )));
        }

        Ok(Self { files })
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    pub fn load(&self, index: usize, device: &Arc<CudaDevice>) -> Result<TrainSample> {
        let path = self.files[index % self.files.len()].clone();
        let tensors = serialization::load_file(&path, device)?;
        TrainSample::from_tensors(path, tensors)
    }
}

impl TrainSample {
    pub fn from_tensors(source: PathBuf, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        let latent = tensors
            .remove("latent")
            .ok_or_else(|| Error::InvalidInput(format!("{} missing latent", source.display())))?
            .to_dtype(DType::BF16)?;
        let text_embedding = tensors
            .remove("text_embedding")
            .ok_or_else(|| {
                Error::InvalidInput(format!("{} missing text_embedding", source.display()))
            })?
            .to_dtype(DType::BF16)?;
        let text_mask = tensors.remove("text_mask").ok_or_else(|| {
            Error::InvalidInput(format!("{} missing text_mask", source.display()))
        })?;

        Ok(Self {
            source,
            latent,
            text_embedding,
            text_mask,
        })
    }
}

// -----------------------------------------------------------------------------
// Bucketing
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct BucketKey {
    pub latent_c: usize,
    pub latent_h: usize,
    pub latent_w: usize,
    pub text_seq: usize,
}

#[derive(Debug)]
pub struct Bucket {
    pub key: BucketKey,
    pub indices: Vec<usize>,
}

#[derive(Debug)]
pub struct BucketPlan {
    pub buckets: Vec<Bucket>,
}

impl BucketPlan {
    pub fn build(files: &[PathBuf]) -> Result<Self> {
        let mut groups: HashMap<BucketKey, Vec<usize>> = HashMap::new();
        for (idx, path) in files.iter().enumerate() {
            let key = read_bucket_key(path)?;
            groups.entry(key).or_default().push(idx);
        }
        let mut buckets: Vec<Bucket> = groups
            .into_iter()
            .map(|(key, indices)| Bucket { key, indices })
            .collect();
        buckets.sort_by_key(|b| (b.key.latent_h, b.key.latent_w, b.key.latent_c, b.key.text_seq));
        Ok(Self { buckets })
    }

    pub fn total_samples(&self) -> usize {
        self.buckets.iter().map(|b| b.indices.len()).sum()
    }
}

pub struct BucketBatchSampler {
    plan: BucketPlan,
    batch_size: usize,
    drop_last: bool,
    seed: u64,
    epoch: u64,
    queue: Vec<Vec<usize>>,
    cursor: usize,
}

impl BucketBatchSampler {
    pub fn new(plan: BucketPlan, batch_size: usize, drop_last: bool, seed: u64) -> Self {
        let mut me = Self {
            plan,
            batch_size,
            drop_last,
            seed,
            epoch: 0,
            queue: Vec::new(),
            cursor: 0,
        };
        me.rebuild_queue();
        me
    }

    fn rebuild_queue(&mut self) {
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(self.epoch));
        let mut all_batches: Vec<Vec<usize>> = Vec::new();

        for bucket in &self.plan.buckets {
            let mut indices = bucket.indices.clone();
            indices.shuffle(&mut rng);
            for chunk in indices.chunks(self.batch_size) {
                if chunk.len() < self.batch_size && self.drop_last {
                    continue;
                }
                all_batches.push(chunk.to_vec());
            }
        }
        all_batches.shuffle(&mut rng);
        self.queue = all_batches;
        self.cursor = 0;
    }

    pub fn next_batch(&mut self) -> Vec<usize> {
        if self.cursor >= self.queue.len() {
            self.epoch = self.epoch.wrapping_add(1);
            self.rebuild_queue();
            if self.queue.is_empty() {
                return Vec::new();
            }
        }
        let batch = self.queue[self.cursor].clone();
        self.cursor += 1;
        batch
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn batches_per_epoch(&self) -> usize {
        self.queue.len().max(1)
    }
}

// -----------------------------------------------------------------------------
// safetensors header probing
// -----------------------------------------------------------------------------

fn read_bucket_key(path: &Path) -> Result<BucketKey> {
    let header = read_safetensors_header(path)?;
    let json: serde_json::Value = serde_json::from_str(&header).map_err(|e| {
        Error::InvalidInput(format!("{}: header JSON parse error: {e}", path.display()))
    })?;
    let obj = json.as_object().ok_or_else(|| {
        Error::InvalidInput(format!("{}: header is not a JSON object", path.display()))
    })?;
    let latent_shape = read_shape(obj, "latent").ok_or_else(|| {
        Error::InvalidInput(format!("{}: missing latent in header", path.display()))
    })?;
    let text_shape = read_shape(obj, "text_embedding").ok_or_else(|| {
        Error::InvalidInput(format!("{}: missing text_embedding in header", path.display()))
    })?;
    if latent_shape.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "{}: latent must be rank 4, got shape {:?}",
            path.display(),
            latent_shape
        )));
    }
    if text_shape.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "{}: text_embedding must be rank 3, got shape {:?}",
            path.display(),
            text_shape
        )));
    }
    Ok(BucketKey {
        latent_c: latent_shape[1],
        latent_h: latent_shape[2],
        latent_w: latent_shape[3],
        text_seq: text_shape[1],
    })
}

fn read_shape(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<Vec<usize>> {
    let entry = obj.get(key)?.as_object()?;
    let arr = entry.get("shape")?.as_array()?;
    arr.iter().map(|v| v.as_u64().map(|x| x as usize)).collect()
}

fn read_safetensors_header(path: &Path) -> Result<String> {
    let f = File::open(path).map_err(|e| Error::Io(format!("open {}: {e}", path.display())))?;
    let mut reader = BufReader::new(f);
    let mut len_bytes = [0u8; 8];
    reader
        .read_exact(&mut len_bytes)
        .map_err(|e| Error::Io(format!("read header len {}: {e}", path.display())))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len > 16 * 1024 * 1024 {
        return Err(Error::InvalidInput(format!(
            "{}: header len {} unreasonable",
            path.display(),
            header_len
        )));
    }
    let mut buf = vec![0u8; header_len];
    reader
        .seek(SeekFrom::Start(8))
        .map_err(|e| Error::Io(format!("seek {}: {e}", path.display())))?;
    reader
        .read_exact(&mut buf)
        .map_err(|e| Error::Io(format!("read header {}: {e}", path.display())))?;
    String::from_utf8(buf)
        .map_err(|e| Error::InvalidInput(format!("{}: header not UTF-8: {e}", path.display())))
}
