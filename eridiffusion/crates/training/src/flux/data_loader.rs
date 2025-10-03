//! Flux data loaders (synthetic + precomputed cached latents).
//! All tensors are created on the target CUDA device in BF16 storage.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Result};
use csv::ReaderBuilder;
use eridiffusion_core::Device;
use eridiffusion_models::{
    common_text::attn_mask_from_lengths,
    devtensor::{to_dtype, BF16},
};
use flame_core::Tensor;
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use sha2::{Digest, Sha256};

use crate::{
    flux::data::{
        make_synthetic_batch, FluxBatch, FluxBatchStats, FluxDataConfig, FluxSampleRecord,
    },
    flux_preprocessor::{
        DatasetStats, PreprocessedFluxBatch, PreprocessedFluxDataset, PreprocessedFluxItem,
    },
};

/// Modes supported by the Flux data loader.
#[derive(Clone)]
pub enum FluxDataMode {
    /// Deterministic synthetic zeros (useful for plumbing tests).
    Synthetic(FluxSyntheticCfg),
    /// Cached latents + text embeddings stored on disk.
    Precomputed(FluxPrecomputedCfg),
}

#[derive(Clone)]
pub struct FluxSyntheticCfg {
    pub data_cfg: FluxDataConfig,
    pub device: Device,
}

#[derive(Clone)]
pub struct FluxPrecomputedCfg {
    pub manifest_path: PathBuf,
    pub root_dir: Option<PathBuf>,
    pub batch_size: usize,
    pub shuffle: bool,
    pub max_epochs: Option<usize>,
    pub enforce_bf16: bool,
    pub validate_on_load: bool,
    pub cache_index: Option<PathBuf>,
    pub device: Device,
}

/// Build or refresh a cached index without instantiating the GPU dataset.
///
/// Returns dataset statistics once the index has been written.
pub fn build_manifest_index(
    manifest_path: PathBuf,
    root_dir: Option<PathBuf>,
    out_path: PathBuf,
    validate: bool,
    overwrite: bool,
) -> Result<DatasetStats> {
    if !overwrite && out_path.exists() {
        bail!("cached index {} already exists (pass --overwrite to rebuild)", out_path.display());
    }

    let cfg = FluxPrecomputedCfg {
        manifest_path,
        root_dir,
        batch_size: 1,
        shuffle: false,
        max_epochs: None,
        enforce_bf16: true,
        validate_on_load: validate,
        cache_index: Some(out_path),
        device: Device::Cpu,
    };

    let items = read_manifest(&cfg)?;
    if cfg.validate_on_load {
        for item in &items {
            validate_path(&item.latents_path)?;
            validate_path(&item.t5_embeds_path)?;
            validate_path(&item.clip_pooled_path)?;
        }
    }

    let mut stats = compute_dataset_stats(&items)?;
    let manifest_hash = manifest_sha256(&cfg.manifest_path)?;
    stats.manifest_hash = Some(manifest_hash.clone());
    write_cached_index(&cfg, &items, &stats, &manifest_hash)?;
    Ok(stats)
}

pub struct FluxDataLoader {
    inner: LoaderInner,
}

enum LoaderInner {
    Synthetic(SyntheticLoader),
    Precomputed(PrecomputedLoader),
}

impl FluxDataLoader {
    pub fn from_mode(mode: FluxDataMode) -> Result<Self> {
        let inner = match mode {
            FluxDataMode::Synthetic(cfg) => LoaderInner::Synthetic(SyntheticLoader { cfg }),
            FluxDataMode::Precomputed(cfg) => {
                let (dataset, cache_hit) = build_dataset_from_manifest(&cfg)?;
                LoaderInner::Precomputed(PrecomputedLoader::new(cfg, dataset, cache_hit))
            }
        };
        Ok(Self { inner })
    }

    pub fn from_precomputed(cfg: FluxPrecomputedCfg) -> Result<Self> {
        Self::from_mode(FluxDataMode::Precomputed(cfg))
    }

    /// Legacy constructor used by older call sites (synthetic zeros).
    pub fn new(data_cfg: FluxDataConfig, device: Device) -> Self {
        Self::from_mode(FluxDataMode::Synthetic(FluxSyntheticCfg { data_cfg, device }))
            .expect("synthetic loader should not fail")
    }

    pub fn next(&mut self) -> Result<Option<FluxBatch>> {
        match &mut self.inner {
            LoaderInner::Synthetic(loader) => loader.next_batch(),
            LoaderInner::Precomputed(loader) => loader.next_batch(),
        }
    }

    pub fn precomputed_stats(&self) -> Option<DatasetStats> {
        match &self.inner {
            LoaderInner::Precomputed(loader) => loader.dataset.stats().cloned(),
            _ => None,
        }
    }

    pub fn precomputed_len(&self) -> Option<usize> {
        match &self.inner {
            LoaderInner::Precomputed(loader) => Some(loader.dataset.len()),
            _ => None,
        }
    }

    pub fn cache_index_hit(&self) -> Option<bool> {
        match &self.inner {
            LoaderInner::Precomputed(loader) => Some(loader.cache_hit),
            _ => None,
        }
    }
}

struct SyntheticLoader {
    cfg: FluxSyntheticCfg,
}

impl SyntheticLoader {
    fn next_batch(&mut self) -> Result<Option<FluxBatch>> {
        let batch = make_synthetic_batch(&self.cfg.data_cfg, &self.cfg.device)?;
        Ok(Some(batch))
    }
}

struct PrecomputedLoader {
    cfg: FluxPrecomputedCfg,
    dataset: Arc<PreprocessedFluxDataset>,
    indices: Vec<usize>,
    cursor: usize,
    epoch: usize,
    cache_hit: bool,
}

impl PrecomputedLoader {
    fn new(
        cfg: FluxPrecomputedCfg,
        dataset: Arc<PreprocessedFluxDataset>,
        cache_hit: bool,
    ) -> Self {
        let len = dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();
        if cfg.shuffle {
            shuffle_deterministic(&mut indices, 0);
        }
        Self { cfg, dataset, indices, cursor: 0, epoch: 0, cache_hit }
    }

    fn reset_epoch(&mut self) {
        self.cursor = 0;
        self.epoch += 1;
        if self.cfg.shuffle {
            shuffle_deterministic(&mut self.indices, self.epoch as u64);
        }
    }

    fn next_batch(&mut self) -> Result<Option<FluxBatch>> {
        let len = self.dataset.len();
        if len == 0 {
            return Ok(None);
        }
        if let Some(max_epochs) = self.cfg.max_epochs {
            if self.epoch >= max_epochs {
                return Ok(None);
            }
        }
        if self.cursor >= len {
            self.reset_epoch();
            if let Some(max_epochs) = self.cfg.max_epochs {
                if self.epoch >= max_epochs {
                    return Ok(None);
                }
            }
        }

        let end = (self.cursor + self.cfg.batch_size).min(len);
        let span = &self.indices[self.cursor..end];
        self.cursor = end;
        if span.is_empty() {
            return Ok(None);
        }

        let dataset = &self.dataset;
        let mut latents_list = Vec::with_capacity(span.len());
        let mut text_list = Vec::with_capacity(span.len());
        let mut pooled_list: Vec<Tensor> = Vec::new();
        let mut seq_lens = Vec::with_capacity(span.len());
        let mut records = Vec::with_capacity(span.len());
        let mut pooled_present = true;

        for &idx in span {
            let item = dataset.get_item(idx)?;
            let prepared = self.prepare_item(item)?;
            seq_lens.push(prepared.seq_len);
            records.push(prepared.record);
            latents_list.push(prepared.latents);
            text_list.push(prepared.text_ctx);
            if let Some(pooled) = prepared.pooled {
                pooled_list.push(pooled);
            } else {
                pooled_present = false;
            }
        }

        // Pad text embeddings to same length before stacking
        let max_len = seq_lens.iter().copied().max().unwrap_or(0);
        let device = dataset.device();
        let padded_text_list: Vec<Tensor> = text_list
            .into_iter()
            .map(|t| pad_to_length(t, max_len as usize, device))
            .collect::<Result<Vec<_>>>()?;

        let latents = Tensor::stack(&latents_list, 0)?;
        let text_ctx = Tensor::stack(&padded_text_list, 0)?;
        let attn_mask = attn_mask_from_lengths(&seq_lens, max_len, device)?;
        let pooled = if pooled_present && !pooled_list.is_empty() {
            Some(Tensor::stack(&pooled_list, 0)?)
        } else {
            None
        };

        let telemetry = compute_batch_stats(&latents, &text_ctx, pooled.as_ref(), span.len());

        Ok(Some(FluxBatch {
            latents,
            text_ctx,
            pooled,
            time_ids: None,
            attn_mask,
            seq_lens,
            records,
            telemetry: Some(telemetry),
        }))
    }

    fn prepare_item(&self, item: PreprocessedFluxBatch) -> Result<PreparedSample> {
        let enforce_bf16 = self.cfg.enforce_bf16;
        let PreprocessedFluxBatch { mut latents, mut t5_embeds, clip_pooled, caption, metadata } =
            item;
        if enforce_bf16 {
            latents = to_dtype(&latents, BF16)?;
            t5_embeds = to_dtype(&t5_embeds, BF16)?;
        }
        let pooled =
            if enforce_bf16 { Some(to_dtype(&clip_pooled, BF16)?) } else { Some(clip_pooled) };

        let dims = t5_embeds.shape().dims();
        if dims.is_empty() {
            bail!("precomputed sample missing text dims");
        }
        let seq_len = dims[0] as i32;
        if seq_len <= 0 {
            bail!("precomputed sample has empty text length");
        }

        Ok(PreparedSample {
            latents,
            text_ctx: t5_embeds,
            pooled,
            seq_len,
            record: FluxSampleRecord { caption, metadata },
        })
    }
}

struct PreparedSample {
    latents: Tensor,
    text_ctx: Tensor,
    pooled: Option<Tensor>,
    seq_len: i32,
    record: FluxSampleRecord,
}

fn build_dataset_from_manifest(
    cfg: &FluxPrecomputedCfg,
) -> Result<(Arc<PreprocessedFluxDataset>, bool)> {
    let manifest_hash = manifest_sha256(&cfg.manifest_path)?;
    let (items, stats, cache_hit) = if let Some(payload) = load_cached_index(cfg)? {
        let mut stats = payload.stats;
        stats.manifest_hash = Some(payload.hash.clone());
        (payload.items, stats, true)
    } else {
        let items = read_manifest(cfg)?;
        if cfg.validate_on_load {
            for item in &items {
                validate_path(&item.latents_path)?;
                validate_path(&item.t5_embeds_path)?;
                validate_path(&item.clip_pooled_path)?;
            }
        }
        let stats = compute_dataset_stats(&items)?;
        if cfg.cache_index.is_some() {
            write_cached_index(cfg, &items, &stats, &manifest_hash)?;
        }
        let mut stats = stats;
        stats.manifest_hash = Some(manifest_hash.clone());
        (items, stats, false)
    };
    Ok((Arc::new(PreprocessedFluxDataset::new(items, cfg.device.clone(), Some(stats))), cache_hit))
}

fn validate_path(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("cache file missing: {}", path.display());
    }
    Ok(())
}

fn read_manifest(cfg: &FluxPrecomputedCfg) -> Result<Vec<PreprocessedFluxItem>> {
    let ext = cfg
        .manifest_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ext == "tsv" {
        read_manifest_tsv(cfg)
    } else if ext == "jsonl" || ext == "json" {
        read_manifest_jsonl(cfg)
    } else {
        bail!("unsupported manifest extension '{}'; expected .tsv or .jsonl", ext)
    }
}

fn read_manifest_tsv(cfg: &FluxPrecomputedCfg) -> Result<Vec<PreprocessedFluxItem>> {
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .flexible(true)
        .from_path(&cfg.manifest_path)?;
    let headers = rdr.headers()?.clone();

    let find_col = |name: &str| -> Result<usize> {
        headers
            .iter()
            .position(|h| h.eq_ignore_ascii_case(name))
            .ok_or_else(|| anyhow!("manifest missing column '{name}'"))
    };

    let idx_latent = find_col("latents_path")?;
    let idx_t5 = find_col("t5_path")?;
    let idx_clip = find_col("clip_path")?;
    let idx_caption = headers.iter().position(|h| h.eq_ignore_ascii_case("caption"));

    let mut items = Vec::new();
    for record in rdr.records() {
        let record = record?;
        let resolve = |raw: &str| resolve_path(cfg.root_dir.as_deref(), raw);
        let latents = resolve(record.get(idx_latent).unwrap_or_default());
        let t5 = resolve(record.get(idx_t5).unwrap_or_default());
        let clip = resolve(record.get(idx_clip).unwrap_or_default());
        let caption = record.get(idx_caption.unwrap_or(idx_latent)).unwrap_or("").to_string();
        let mut metadata = HashMap::<String, Value>::new();
        for (header, value) in headers.iter().zip(record.iter()) {
            if header.eq_ignore_ascii_case("latents_path")
                || header.eq_ignore_ascii_case("t5_path")
                || header.eq_ignore_ascii_case("clip_path")
                || header.eq_ignore_ascii_case("caption")
            {
                continue;
            }
            if !value.trim().is_empty() {
                metadata.insert(header.to_string(), parse_metadata_value(value.trim()));
            }
        }

        let item = PreprocessedFluxItem {
            latents_path: latents,
            t5_embeds_path: t5,
            clip_pooled_path: clip,
            caption,
            metadata,
        };
        items.push(item);
    }
    Ok(items)
}

#[derive(Deserialize)]
struct JsonManifestRow {
    latents_path: Option<String>,
    t5_path: Option<String>,
    clip_path: Option<String>,
    caption: Option<String>,
    metadata: Option<HashMap<String, Value>>,
}

fn read_manifest_jsonl(cfg: &FluxPrecomputedCfg) -> Result<Vec<PreprocessedFluxItem>> {
    let file = File::open(&cfg.manifest_path)?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: JsonManifestRow = serde_json::from_str(&line)
            .map_err(|e| anyhow!("json manifest parse error on line {}: {e}", line_idx + 1))?;
        let resolve = |raw: &str| resolve_path(cfg.root_dir.as_deref(), raw);
        let latents_raw =
            row.latents_path.ok_or_else(|| anyhow!("json manifest missing 'latents_path'"))?;
        let t5_raw = row.t5_path.ok_or_else(|| anyhow!("json manifest missing 't5_path'"))?;
        let clip_raw = row.clip_path.ok_or_else(|| anyhow!("json manifest missing 'clip_path'"))?;
        let item = PreprocessedFluxItem {
            latents_path: resolve(&latents_raw),
            t5_embeds_path: resolve(&t5_raw),
            clip_pooled_path: resolve(&clip_raw),
            caption: row.caption.unwrap_or_default(),
            metadata: row.metadata.unwrap_or_default(),
        };
        items.push(item);
    }
    Ok(items)
}

fn resolve_path(root: Option<&Path>, raw: &str) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        path.to_path_buf()
    } else if let Some(root) = root {
        root.join(path)
    } else {
        path.to_path_buf()
    }
}

fn parse_metadata_value(raw: &str) -> Value {
    if let Ok(v) = raw.parse::<i64>() {
        Value::from(v)
    } else if let Ok(v) = raw.parse::<f64>() {
        Value::from(v)
    } else if raw.eq_ignore_ascii_case("true") || raw.eq_ignore_ascii_case("false") {
        Value::from(raw.eq_ignore_ascii_case("true"))
    } else {
        Value::from(raw.to_string())
    }
}

#[derive(Serialize, Deserialize)]
struct CachedIndex {
    manifest_path: String,
    manifest_hash: String,
    len: usize,
    total_latent_bytes: u64,
    total_t5_bytes: u64,
    total_clip_bytes: u64,
    #[serde(default)]
    total_bytes: u64,
    #[serde(default)]
    created_at_unix: u64,
    #[serde(default)]
    files: Vec<CachedFileMeta>,
    items: Vec<CachedIndexItem>,
}

#[derive(Serialize, Deserialize)]
struct CachedIndexItem {
    latents_path: String,
    t5_embeds_path: String,
    clip_pooled_path: String,
    caption: String,
    metadata: HashMap<String, Value>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct CachedFileMeta {
    path: String,
    bytes: u64,
    #[serde(default)]
    modified_unix: Option<u64>,
}

struct CachedPayload {
    items: Vec<PreprocessedFluxItem>,
    stats: DatasetStats,
    hash: String,
}

fn load_cached_index(cfg: &FluxPrecomputedCfg) -> Result<Option<CachedPayload>> {
    let Some(index_path) = cfg.cache_index.as_ref() else { return Ok(None) };
    if !index_path.exists() {
        return Ok(None);
    }
    let file = File::open(index_path)?;
    let cached: CachedIndex = match serde_json::from_reader(file) {
        Ok(c) => c,
        Err(e) => {
            println!("WARN: ignoring cached index {} (parse error: {e})", index_path.display());
            return Ok(None);
        }
    };
    let manifest_canon =
        cfg.manifest_path.canonicalize().unwrap_or_else(|_| cfg.manifest_path.clone());
    let manifest_str = manifest_canon.display().to_string();
    if cached.manifest_path != manifest_str {
        return Ok(None);
    }
    let current_hash = manifest_sha256(&cfg.manifest_path)?;
    if cached.manifest_hash != current_hash {
        return Ok(None);
    }
    let items = cached
        .items
        .into_iter()
        .map(|entry| PreprocessedFluxItem {
            latents_path: PathBuf::from(entry.latents_path),
            t5_embeds_path: PathBuf::from(entry.t5_embeds_path),
            clip_pooled_path: PathBuf::from(entry.clip_pooled_path),
            caption: entry.caption,
            metadata: entry.metadata,
        })
        .collect();
    let stats = DatasetStats {
        len: cached.len,
        total_latent_bytes: cached.total_latent_bytes,
        total_t5_bytes: cached.total_t5_bytes,
        total_clip_bytes: cached.total_clip_bytes,
        manifest_hash: Some(cached.manifest_hash.clone()),
    };
    Ok(Some(CachedPayload { items, stats, hash: cached.manifest_hash }))
}

fn write_cached_index(
    cfg: &FluxPrecomputedCfg,
    items: &[PreprocessedFluxItem],
    stats: &DatasetStats,
    manifest_hash: &str,
) -> Result<()> {
    let Some(index_path) = cfg.cache_index.as_ref() else {
        return Ok(());
    };
    if let Some(parent) = index_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let manifest_str = cfg
        .manifest_path
        .canonicalize()
        .unwrap_or_else(|_| cfg.manifest_path.clone())
        .display()
        .to_string();
    let cached = CachedIndex {
        manifest_path: manifest_str,
        manifest_hash: manifest_hash.to_string(),
        len: stats.len,
        total_latent_bytes: stats.total_latent_bytes,
        total_t5_bytes: stats.total_t5_bytes,
        total_clip_bytes: stats.total_clip_bytes,
        total_bytes: stats
            .total_latent_bytes
            .saturating_add(stats.total_t5_bytes)
            .saturating_add(stats.total_clip_bytes),
        created_at_unix: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        files: gather_file_meta(items)?,
        items: items
            .iter()
            .map(|item| CachedIndexItem {
                latents_path: item.latents_path.display().to_string(),
                t5_embeds_path: item.t5_embeds_path.display().to_string(),
                clip_pooled_path: item.clip_pooled_path.display().to_string(),
                caption: item.caption.clone(),
                metadata: item.metadata.clone(),
            })
            .collect(),
    };
    let file = File::create(index_path)?;
    serde_json::to_writer_pretty(file, &cached)?;
    Ok(())
}

fn gather_file_meta(items: &[PreprocessedFluxItem]) -> Result<Vec<CachedFileMeta>> {
    let mut map = HashMap::<String, CachedFileMeta>::new();
    for item in items {
        for path in [&item.latents_path, &item.t5_embeds_path, &item.clip_pooled_path] {
            let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            let key = canon.display().to_string();
            if map.contains_key(&key) {
                continue;
            }
            let meta = canon.metadata()?;
            let bytes = meta.len();
            let modified_unix = meta
                .modified()
                .ok()
                .and_then(|ts| ts.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs());
            map.insert(key.clone(), CachedFileMeta { path: key, bytes, modified_unix });
        }
    }
    Ok(map.into_values().collect())
}

fn compute_batch_stats(
    latents: &Tensor,
    text_ctx: &Tensor,
    pooled: Option<&Tensor>,
    samples: usize,
) -> FluxBatchStats {
    let mut stats = FluxBatchStats { samples, ..Default::default() };
    stats.bytes_latents = (latents.shape().elem_count() as u64)
        .saturating_mul(latents.dtype().size_in_bytes() as u64);
    stats.bytes_text = (text_ctx.shape().elem_count() as u64)
        .saturating_mul(text_ctx.dtype().size_in_bytes() as u64);
    stats.bytes_pooled = pooled
        .map(|t| (t.shape().elem_count() as u64).saturating_mul(t.dtype().size_in_bytes() as u64))
        .unwrap_or(0);
    stats
}

fn manifest_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn compute_dataset_stats(items: &[PreprocessedFluxItem]) -> Result<DatasetStats> {
    let mut latent_bytes = 0u64;
    let mut t5_bytes = 0u64;
    let mut clip_bytes = 0u64;
    let mut seen_latents = HashSet::new();
    let mut seen_t5 = HashSet::new();
    let mut seen_clip = HashSet::new();

    for item in items {
        if seen_latents.insert(item.latents_path.clone()) {
            latent_bytes = latent_bytes.saturating_add(file_size(&item.latents_path)?);
        }
        if seen_t5.insert(item.t5_embeds_path.clone()) {
            t5_bytes = t5_bytes.saturating_add(file_size(&item.t5_embeds_path)?);
        }
        if seen_clip.insert(item.clip_pooled_path.clone()) {
            clip_bytes = clip_bytes.saturating_add(file_size(&item.clip_pooled_path)?);
        }
    }

    Ok(DatasetStats {
        len: items.len(),
        total_latent_bytes: latent_bytes,
        total_t5_bytes: t5_bytes,
        total_clip_bytes: clip_bytes,
        manifest_hash: None,
    })
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(path.metadata()?.len())
}

fn pad_to_length(tensor: Tensor, target_len: usize, device: &Device) -> Result<Tensor> {
    use eridiffusion_models::devtensor::{zeros_on, BF16};
    use flame_core::Shape;

    let dims = tensor.shape().dims();
    let current_len = dims[0] as usize;

    if current_len >= target_len {
        return Ok(tensor);
    }

    // Pad with zeros: [current_len, hidden] -> [target_len, hidden]
    let hidden = dims[1] as usize;
    let pad_len = target_len - current_len;
    let pad_shape = Shape::from_dims(&[pad_len, hidden]);
    let padding = zeros_on(pad_shape, device, BF16)?;

    Ok(Tensor::cat(&[&tensor, &padding], 0)?)
}

fn shuffle_deterministic(v: &mut [usize], seed: u64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for i in (1..v.len()).rev() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let j = (state as usize) % (i + 1);
        v.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_shuffle() {
        let mut a = vec![0, 1, 2, 3, 4, 5];
        let mut b = a.clone();
        shuffle_deterministic(&mut a, 1234);
        shuffle_deterministic(&mut b, 1234);
        assert_eq!(a, b);
    }
}
