//! SDXL manifest loader (cached latents + CLIP-L/CLIP-G embeddings).
//! Mirrors the Flux loader but tailored for SDXL's dual-CLIP embeddings.

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
use eridiffusion_models::devtensor::{tensor_from_slice_on, to_dtype, BF16, F32_};
use flame_core::{Shape, Tensor};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    conditioning::time_ids::build_time_ids,
    flux::data::{FluxBatchStats, FluxSampleRecord},
};

/// SDXL-specific precomputed config
#[derive(Clone)]
pub struct SdxlPrecomputedCfg {
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

pub struct SdxlDataLoader {
    inner: LoaderInner,
}

enum LoaderInner {
    Precomputed(PrecomputedLoader),
}

impl SdxlDataLoader {
    pub fn from_precomputed(cfg: SdxlPrecomputedCfg) -> Result<Self> {
        let (dataset, cache_hit) = build_dataset_from_manifest(&cfg)?;
        let loader = PrecomputedLoader::new(cfg, dataset, cache_hit);
        Ok(Self { inner: LoaderInner::Precomputed(loader) })
    }

    pub fn next(&mut self) -> Result<Option<SdxlBatch>> {
        match &mut self.inner {
            LoaderInner::Precomputed(loader) => loader.next_batch(),
        }
    }

    pub fn precomputed_stats(&self) -> Option<&SdxlDatasetStats> {
        match &self.inner {
            LoaderInner::Precomputed(loader) => loader.dataset.stats(),
        }
    }

    pub fn cache_index_hit(&self) -> Option<bool> {
        match &self.inner {
            LoaderInner::Precomputed(loader) => Some(loader.cache_hit),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SdxlDatasetStats {
    pub len: usize,
    pub total_latent_bytes: u64,
    pub total_clip_l_bytes: u64,
    pub total_clip_g_bytes: u64,
    pub total_pooled_bytes: u64,
    pub manifest_hash: Option<String>,
}

pub struct SdxlBatch {
    pub latents: Tensor, // [B,4,H/8,W/8]
    pub clip_l: Tensor,  // [B,77,768]
    pub clip_g: Tensor,  // [B,77,1280]
    pub pooled: Tensor,  // [B,1280]
    pub captions: Vec<String>,
    pub timesteps: Tensor, // [B] (F32)
    pub time_ids: Tensor,  // [B,6] (F32)
    pub records: Vec<FluxSampleRecord>,
    pub telemetry: Option<FluxBatchStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreprocessedSdxlItem {
    latents_path: PathBuf,
    clip_l_path: PathBuf,
    clip_g_path: PathBuf,
    pooled_path: PathBuf,
    caption: String,
    metadata: HashMap<String, Value>,
}

#[derive(Clone)]
struct PreprocessedSdxlDataset {
    items: Vec<PreprocessedSdxlItem>,
    device: Device,
    stats: Option<SdxlDatasetStats>,
}

impl PreprocessedSdxlDataset {
    fn new(
        items: Vec<PreprocessedSdxlItem>,
        device: Device,
        stats: Option<SdxlDatasetStats>,
    ) -> Self {
        Self { items, device, stats }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn stats(&self) -> Option<&SdxlDatasetStats> {
        self.stats.as_ref()
    }

    fn get_item(&self, idx: usize) -> Result<PreprocessedSdxlBatch> {
        let item = &self.items[idx];
        let latents = load_tensor(&item.latents_path, &self.device)?;
        let clip_l = load_tensor(&item.clip_l_path, &self.device)?;
        let clip_g = load_tensor(&item.clip_g_path, &self.device)?;
        let pooled = load_tensor(&item.pooled_path, &self.device)?;
        let time_ids = compute_time_ids_tensor(&latents, &item.metadata, &self.device)?;
        let timestep = extract_timestep(&item.metadata);
        Ok(PreprocessedSdxlBatch {
            latents,
            clip_l,
            clip_g,
            pooled,
            time_ids,
            timestep,
            caption: item.caption.clone(),
            metadata: item.metadata.clone(),
        })
    }
}

struct PreprocessedSdxlBatch {
    latents: Tensor,
    clip_l: Tensor,
    clip_g: Tensor,
    pooled: Tensor,
    time_ids: Tensor,
    timestep: f32,
    caption: String,
    metadata: HashMap<String, Value>,
}

struct PrecomputedLoader {
    cfg: SdxlPrecomputedCfg,
    dataset: Arc<PreprocessedSdxlDataset>,
    indices: Vec<usize>,
    cursor: usize,
    epoch: usize,
    cache_hit: bool,
}

impl PrecomputedLoader {
    fn new(
        cfg: SdxlPrecomputedCfg,
        dataset: Arc<PreprocessedSdxlDataset>,
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

    fn next_batch(&mut self) -> Result<Option<SdxlBatch>> {
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

        let mut latents_list = Vec::with_capacity(span.len());
        let mut clip_l_list = Vec::with_capacity(span.len());
        let mut clip_g_list = Vec::with_capacity(span.len());
        let mut pooled_list = Vec::with_capacity(span.len());
        let mut captions = Vec::with_capacity(span.len());
        let mut records = Vec::with_capacity(span.len());
        let mut time_ids_list = Vec::with_capacity(span.len());
        let mut timestep_list = Vec::with_capacity(span.len());

        for &idx in span {
            let item = self.dataset.get_item(idx)?;
            let (latents, clip_l, clip_g, pooled, time_ids, timestep, record) =
                self.prepare_item(item)?;
            latents_list.push(latents);
            clip_l_list.push(clip_l);
            clip_g_list.push(clip_g);
            pooled_list.push(pooled);
            time_ids_list.push(time_ids);
            timestep_list.push(timestep);
            captions.push(record.caption.clone());
            records.push(record);
        }

        let latents = Tensor::stack(&latents_list, 0)?;
        let clip_l = Tensor::stack(&clip_l_list, 0)?;
        let clip_g = Tensor::stack(&clip_g_list, 0)?;
        let pooled = Tensor::stack(&pooled_list, 0)?;
        let time_ids = Tensor::stack(&time_ids_list, 0)?;
        let timesteps = tensor_from_slice_on(
            &timestep_list,
            Shape::from_dims(&[span.len()]),
            &self.cfg.device,
            F32_,
        )?;

        let telemetry = compute_batch_stats(&latents, &clip_l, &clip_g, &pooled, span.len());

        Ok(Some(SdxlBatch {
            latents,
            clip_l,
            clip_g,
            pooled,
            captions,
            timesteps,
            time_ids,
            records,
            telemetry: Some(telemetry),
        }))
    }

    fn prepare_item(
        &self,
        item: PreprocessedSdxlBatch,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, f32, FluxSampleRecord)> {
        let enforce_bf16 = self.cfg.enforce_bf16;
        let PreprocessedSdxlBatch {
            mut latents,
            mut clip_l,
            mut clip_g,
            mut pooled,
            time_ids,
            timestep,
            caption,
            metadata,
        } = item;
        if enforce_bf16 {
            latents = to_dtype(&latents, BF16)?;
            clip_l = to_dtype(&clip_l, BF16)?;
            clip_g = to_dtype(&clip_g, BF16)?;
            pooled = to_dtype(&pooled, BF16)?;
        }
        Ok((
            latents,
            clip_l,
            clip_g,
            pooled,
            time_ids,
            timestep,
            FluxSampleRecord { caption, metadata },
        ))
    }
}

fn build_dataset_from_manifest(
    cfg: &SdxlPrecomputedCfg,
) -> Result<(Arc<PreprocessedSdxlDataset>, bool)> {
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
                validate_path(&item.clip_l_path)?;
                validate_path(&item.clip_g_path)?;
                validate_path(&item.pooled_path)?;
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
    Ok((Arc::new(PreprocessedSdxlDataset::new(items, cfg.device.clone(), Some(stats))), cache_hit))
}

fn validate_path(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("cache file missing: {}", path.display());
    }
    Ok(())
}

fn read_manifest(cfg: &SdxlPrecomputedCfg) -> Result<Vec<PreprocessedSdxlItem>> {
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

fn read_manifest_tsv(cfg: &SdxlPrecomputedCfg) -> Result<Vec<PreprocessedSdxlItem>> {
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
    let idx_clip_l = find_col("clip_l_path")?;
    let idx_clip_g = find_col("clip_g_path")?;
    let idx_pooled = find_col("pooled_path")?;
    let idx_caption = headers.iter().position(|h| h.eq_ignore_ascii_case("caption"));

    let mut items = Vec::new();
    for record in rdr.records() {
        let record = record?;
        let resolve = |raw: &str| resolve_path(cfg.root_dir.as_deref(), raw);
        let latents = resolve(record.get(idx_latent).unwrap_or_default());
        let clip_l = resolve(record.get(idx_clip_l).unwrap_or_default());
        let clip_g = resolve(record.get(idx_clip_g).unwrap_or_default());
        let pooled = resolve(record.get(idx_pooled).unwrap_or_default());
        let caption = record.get(idx_caption.unwrap_or(idx_latent)).unwrap_or("").to_string();

        let mut metadata = HashMap::<String, Value>::new();
        for (header, value) in headers.iter().zip(record.iter()) {
            if header.eq_ignore_ascii_case("latents_path")
                || header.eq_ignore_ascii_case("clip_l_path")
                || header.eq_ignore_ascii_case("clip_g_path")
                || header.eq_ignore_ascii_case("pooled_path")
                || header.eq_ignore_ascii_case("caption")
            {
                continue;
            }
            if !value.trim().is_empty() {
                metadata.insert(header.to_string(), parse_metadata_value(value.trim()));
            }
        }

        items.push(PreprocessedSdxlItem {
            latents_path: latents,
            clip_l_path: clip_l,
            clip_g_path: clip_g,
            pooled_path: pooled,
            caption,
            metadata,
        });
    }
    Ok(items)
}

#[derive(Deserialize)]
struct JsonManifestRow {
    latents_path: Option<String>,
    clip_l_path: Option<String>,
    clip_g_path: Option<String>,
    pooled_path: Option<String>,
    caption: Option<String>,
    metadata: Option<HashMap<String, Value>>,
}

fn read_manifest_jsonl(cfg: &SdxlPrecomputedCfg) -> Result<Vec<PreprocessedSdxlItem>> {
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
        let clip_l_raw =
            row.clip_l_path.ok_or_else(|| anyhow!("json manifest missing 'clip_l_path'"))?;
        let clip_g_raw =
            row.clip_g_path.ok_or_else(|| anyhow!("json manifest missing 'clip_g_path'"))?;
        let pooled_raw =
            row.pooled_path.ok_or_else(|| anyhow!("json manifest missing 'pooled_path'"))?;

        let caption = row.caption.unwrap_or_default();
        let metadata = row.metadata.unwrap_or_default();

        items.push(PreprocessedSdxlItem {
            latents_path: resolve(&latents_raw),
            clip_l_path: resolve(&clip_l_raw),
            clip_g_path: resolve(&clip_g_raw),
            pooled_path: resolve(&pooled_raw),
            caption,
            metadata,
        });
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
    total_clip_l_bytes: u64,
    total_clip_g_bytes: u64,
    total_pooled_bytes: u64,
    total_bytes: u64,
    created_at_unix: u64,
    files: Vec<CachedFileMeta>,
    items: Vec<CachedIndexItem>,
}

#[derive(Serialize, Deserialize)]
struct CachedIndexItem {
    latents_path: String,
    clip_l_path: String,
    clip_g_path: String,
    pooled_path: String,
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
    items: Vec<PreprocessedSdxlItem>,
    stats: SdxlDatasetStats,
    hash: String,
}

fn load_cached_index(cfg: &SdxlPrecomputedCfg) -> Result<Option<CachedPayload>> {
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
        .map(|entry| PreprocessedSdxlItem {
            latents_path: PathBuf::from(entry.latents_path),
            clip_l_path: PathBuf::from(entry.clip_l_path),
            clip_g_path: PathBuf::from(entry.clip_g_path),
            pooled_path: PathBuf::from(entry.pooled_path),
            caption: entry.caption,
            metadata: entry.metadata,
        })
        .collect();
    let stats = SdxlDatasetStats {
        len: cached.len,
        total_latent_bytes: cached.total_latent_bytes,
        total_clip_l_bytes: cached.total_clip_l_bytes,
        total_clip_g_bytes: cached.total_clip_g_bytes,
        total_pooled_bytes: cached.total_pooled_bytes,
        manifest_hash: Some(cached.manifest_hash.clone()),
    };
    Ok(Some(CachedPayload { items, stats, hash: cached.manifest_hash }))
}

fn write_cached_index(
    cfg: &SdxlPrecomputedCfg,
    items: &[PreprocessedSdxlItem],
    stats: &SdxlDatasetStats,
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
        total_clip_l_bytes: stats.total_clip_l_bytes,
        total_clip_g_bytes: stats.total_clip_g_bytes,
        total_pooled_bytes: stats.total_pooled_bytes,
        total_bytes: stats
            .total_latent_bytes
            .saturating_add(stats.total_clip_l_bytes)
            .saturating_add(stats.total_clip_g_bytes)
            .saturating_add(stats.total_pooled_bytes),
        created_at_unix: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        files: gather_file_meta(items)?,
        items: items
            .iter()
            .map(|item| CachedIndexItem {
                latents_path: item.latents_path.display().to_string(),
                clip_l_path: item.clip_l_path.display().to_string(),
                clip_g_path: item.clip_g_path.display().to_string(),
                pooled_path: item.pooled_path.display().to_string(),
                caption: item.caption.clone(),
                metadata: item.metadata.clone(),
            })
            .collect(),
    };
    let file = File::create(index_path)?;
    serde_json::to_writer_pretty(file, &cached)?;
    Ok(())
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

fn compute_dataset_stats(items: &[PreprocessedSdxlItem]) -> Result<SdxlDatasetStats> {
    let mut latent_bytes = 0u64;
    let mut clip_l_bytes = 0u64;
    let mut clip_g_bytes = 0u64;
    let mut pooled_bytes = 0u64;
    let mut seen_latents = HashSet::new();
    let mut seen_clip_l = HashSet::new();
    let mut seen_clip_g = HashSet::new();
    let mut seen_pooled = HashSet::new();

    for item in items {
        if seen_latents.insert(item.latents_path.clone()) {
            latent_bytes = latent_bytes.saturating_add(file_size(&item.latents_path)?);
        }
        if seen_clip_l.insert(item.clip_l_path.clone()) {
            clip_l_bytes = clip_l_bytes.saturating_add(file_size(&item.clip_l_path)?);
        }
        if seen_clip_g.insert(item.clip_g_path.clone()) {
            clip_g_bytes = clip_g_bytes.saturating_add(file_size(&item.clip_g_path)?);
        }
        if seen_pooled.insert(item.pooled_path.clone()) {
            pooled_bytes = pooled_bytes.saturating_add(file_size(&item.pooled_path)?);
        }
    }

    Ok(SdxlDatasetStats {
        len: items.len(),
        total_latent_bytes: latent_bytes,
        total_clip_l_bytes: clip_l_bytes,
        total_clip_g_bytes: clip_g_bytes,
        total_pooled_bytes: pooled_bytes,
        manifest_hash: None,
    })
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(path.metadata()?.len())
}

fn gather_file_meta(items: &[PreprocessedSdxlItem]) -> Result<Vec<CachedFileMeta>> {
    let mut map = HashMap::<String, CachedFileMeta>::new();
    for item in items {
        for path in [&item.latents_path, &item.clip_l_path, &item.clip_g_path, &item.pooled_path] {
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

fn compute_time_ids_tensor(
    latents: &Tensor,
    metadata: &HashMap<String, Value>,
    device: &Device,
) -> Result<Tensor> {
    let (latent_h, latent_w) = infer_latent_hw(latents)?;
    let latent_scale = 8.0f32;
    let target_h_default = latent_h as f32 * latent_scale;
    let target_w_default = latent_w as f32 * latent_scale;

    let target_h = metadata_f32(metadata, &["target_height", "target_h", "height"])
        .unwrap_or(target_h_default);
    let target_w =
        metadata_f32(metadata, &["target_width", "target_w", "width"]).unwrap_or(target_w_default);
    let orig_h = metadata_f32(metadata, &["orig_height", "original_height"]).unwrap_or(target_h);
    let orig_w = metadata_f32(metadata, &["orig_width", "original_width"]).unwrap_or(target_w);
    let crop_y = metadata_f32(metadata, &["crop_top", "crop_y"]).unwrap_or(0.0);
    let crop_x = metadata_f32(metadata, &["crop_left", "crop_x"]).unwrap_or(0.0);

    let time_ids = build_time_ids(1, orig_h, orig_w, crop_y, crop_x, target_h, target_w, device)
        .map_err(|e| anyhow!("build_time_ids failed: {e}"))?;
    time_ids.reshape(&[6]).map_err(|e| anyhow!("reshape time_ids failed: {e}"))
}

fn infer_latent_hw(latents: &Tensor) -> Result<(i64, i64)> {
    let dims = latents.shape().dims();
    match dims.len() {
        3 => Ok((dims[1] as i64, dims[2] as i64)),
        4 => Ok((dims[2] as i64, dims[3] as i64)),
        other => Err(anyhow!("unexpected latent rank {} for SDXL cache: {:?}", other, dims)),
    }
}

fn metadata_f32(meta: &HashMap<String, Value>, keys: &[&str]) -> Option<f32> {
    for key in keys {
        if let Some(v) = meta.get(*key) {
            if let Some(val) = value_to_f32(v) {
                return Some(val);
            }
        }
    }
    None
}

fn extract_timestep(meta: &HashMap<String, Value>) -> f32 {
    metadata_f32(meta, &["timestep", "timesteps", "step", "sigma"]).unwrap_or(0.0)
}

fn value_to_f32(value: &Value) -> Option<f32> {
    match value {
        Value::Number(num) => num.as_f64().map(|v| v as f32),
        Value::String(s) => s.parse::<f32>().ok(),
        Value::Array(arr) => arr.first().and_then(value_to_f32),
        _ => None,
    }
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

fn compute_batch_stats(
    latents: &Tensor,
    clip_l: &Tensor,
    clip_g: &Tensor,
    pooled: &Tensor,
    samples: usize,
) -> FluxBatchStats {
    let mut stats = FluxBatchStats { samples, ..Default::default() };
    stats.bytes_latents = (latents.shape().elem_count() as u64)
        .saturating_mul(latents.dtype().size_in_bytes() as u64);
    stats.bytes_text = (clip_l.shape().elem_count() as u64)
        .saturating_mul(clip_l.dtype().size_in_bytes() as u64)
        .saturating_add(
            (clip_g.shape().elem_count() as u64)
                .saturating_mul(clip_g.dtype().size_in_bytes() as u64),
        );
    stats.bytes_pooled =
        (pooled.shape().elem_count() as u64).saturating_mul(pooled.dtype().size_in_bytes() as u64);
    stats
}

fn load_tensor(path: &Path, device: &Device) -> Result<Tensor> {
    let bytes = fs::read(path)?;
    let safetensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let view = safetensors.tensor("data")?;
    let shape_dims: Vec<usize> = view.shape().iter().copied().collect();

    let data_f32: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F32 => bytemuck::cast_slice::<u8, f32>(view.data()).to_vec(),
        safetensors::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        safetensors::Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        other => bail!("Unsupported safetensor dtype: {:?}", other),
    };

    let tensor_f32 = tensor_from_slice_on(&data_f32, Shape::from_dims(&shape_dims), device, F32_)?;
    Ok(tensor_f32.to_dtype(BF16)?)
}

#[cfg(test)]
mod tests {
    use super::shuffle_deterministic;

    #[test]
    fn deterministic_shuffle() {
        let mut a = vec![0, 1, 2, 3, 4, 5];
        let mut b = a.clone();
        shuffle_deterministic(&mut a, 1234);
        shuffle_deterministic(&mut b, 1234);
        assert_eq!(a, b);
    }
}
