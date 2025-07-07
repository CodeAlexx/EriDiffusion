//! WebDataset support for large-scale distributed training

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType, Device};
use std::path::{Path, PathBuf};
use std::io::{Read, BufReader};
use tar::Archive;
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use std::sync::Arc;
use dashmap::DashMap;
use tracing::{info, debug, warn};

/// WebDataset shard containing multiple samples
pub struct WebDatasetShard {
    path: PathBuf,
    samples: Vec<WebDatasetSample>,
}

/// Individual sample in WebDataset format
#[derive(Debug, Clone)]
pub struct WebDatasetSample {
    /// Unique key for the sample
    pub key: String,
    /// Image data (JPEG/PNG bytes)
    pub image_data: Vec<u8>,
    /// Text caption
    pub caption: String,
    /// Optional metadata
    pub metadata: Option<SampleMetadata>,
}

/// Sample metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleMetadata {
    pub width: u32,
    pub height: u32,
    pub original_width: Option<u32>,
    pub original_height: Option<u32>,
    pub aesthetic_score: Option<f32>,
    pub tags: Option<Vec<String>>,
}

/// WebDataset reader for streaming large datasets
pub struct WebDatasetReader {
    shard_paths: Vec<PathBuf>,
    current_shard: usize,
    buffer_size: usize,
    shuffle_buffer: Option<usize>,
    cache: Arc<DashMap<String, WebDatasetSample>>,
}

impl WebDatasetReader {
    /// Create new WebDataset reader
    pub fn new(shard_pattern: &str, buffer_size: usize) -> Result<Self> {
        let shard_paths = Self::find_shards(shard_pattern)?;
        
        if shard_paths.is_empty() {
            return Err(Error::DataError(format!(
                "No shards found matching pattern: {}",
                shard_pattern
            )));
        }
        
        info!("Found {} WebDataset shards", shard_paths.len());
        
        Ok(Self {
            shard_paths,
            current_shard: 0,
            buffer_size,
            shuffle_buffer: None,
            cache: Arc::new(DashMap::new()),
        })
    }
    
    /// Enable shuffle buffer
    pub fn with_shuffle(mut self, buffer_size: usize) -> Self {
        self.shuffle_buffer = Some(buffer_size);
        self
    }
    
    /// Find all shard files matching pattern
    fn find_shards(pattern: &str) -> Result<Vec<PathBuf>> {
        use glob::glob;
        
        let mut paths = Vec::new();
        for entry in glob(pattern).map_err(|e| Error::DataError(format!("Invalid pattern: {}", e)))? {
            match entry {
                Ok(path) => paths.push(path),
                Err(e) => warn!("Error reading path: {}", e),
            }
        }
        
        paths.sort();
        Ok(paths)
    }
    
    /// Read samples from current shard
    pub fn read_shard(&mut self) -> Result<Vec<WebDatasetSample>> {
        if self.current_shard >= self.shard_paths.len() {
            return Ok(Vec::new());
        }
        
        let shard_path = &self.shard_paths[self.current_shard];
        debug!("Reading shard: {}", shard_path.display());
        
        let samples = if shard_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            self.read_tar_gz(shard_path)?
        } else {
            self.read_tar(shard_path)?
        };
        
        self.current_shard += 1;
        Ok(samples)
    }
    
    /// Read tar.gz archive
    fn read_tar_gz(&self, path: &Path) -> Result<Vec<WebDatasetSample>> {
        let file = std::fs::File::open(path)?;
        let decoder = GzDecoder::new(BufReader::new(file));
        let mut archive = Archive::new(decoder);
        self.read_archive(&mut archive)
    }
    
    /// Read tar archive
    fn read_tar(&self, path: &Path) -> Result<Vec<WebDatasetSample>> {
        let file = std::fs::File::open(path)?;
        let mut archive = Archive::new(BufReader::new(file));
        self.read_archive(&mut archive)
    }
    
    /// Read samples from archive
    fn read_archive<R: Read>(&self, archive: &mut Archive<R>) -> Result<Vec<WebDatasetSample>> {
        let mut samples_map: std::collections::HashMap<String, WebDatasetSample> = 
            std::collections::HashMap::new();
        
        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_string_lossy().to_string();
            
            // Parse filename to get key and extension
            let parts: Vec<&str> = path.rsplitn(2, '.').collect();
            if parts.len() != 2 {
                continue;
            }
            
            let ext = parts[0];
            let key = parts[1].to_string();
            
            // Read entry data
            let mut data = Vec::new();
            entry.read_to_end(&mut data)?;
            
            // Get or create sample
            let sample = samples_map.entry(key.clone()).or_insert_with(|| {
                WebDatasetSample {
                    key: key.clone(),
                    image_data: Vec::new(),
                    caption: String::new(),
                    metadata: None,
                }
            });
            
            // Update sample based on file type
            match ext {
                "jpg" | "jpeg" | "png" | "webp" => {
                    sample.image_data = data;
                }
                "txt" => {
                    sample.caption = String::from_utf8_lossy(&data).trim().to_string();
                }
                "json" => {
                    if let Ok(metadata) = serde_json::from_slice::<SampleMetadata>(&data) {
                        sample.metadata = Some(metadata);
                    }
                }
                _ => {}
            }
        }
        
        // Filter out incomplete samples
        let complete_samples: Vec<_> = samples_map
            .into_values()
            .filter(|s| !s.image_data.is_empty() && !s.caption.is_empty())
            .collect();
        
        info!("Loaded {} complete samples from shard", complete_samples.len());
        Ok(complete_samples)
    }
    
    /// Create async iterator over samples
    pub async fn iter_samples(&mut self) -> WebDatasetIterator {
        let (tx, rx) = mpsc::channel(self.buffer_size);
        
        // Spawn background task to read shards
        let shard_paths = self.shard_paths.clone();
        let shuffle_buffer = self.shuffle_buffer;
        
        tokio::spawn(async move {
            for shard_path in shard_paths {
                match Self::read_shard_static(&shard_path) {
                    Ok(mut samples) => {
                        // Shuffle within shard if requested
                        if shuffle_buffer.is_some() {
                            use rand::seq::SliceRandom;
                            samples.shuffle(&mut rand::thread_rng());
                        }
                        
                        for sample in samples {
                            if tx.send(sample).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Error reading shard {}: {}", shard_path.display(), e);
                    }
                }
            }
        });
        
        WebDatasetIterator { receiver: rx }
    }
    
    /// Static method to read a shard
    fn read_shard_static(path: &Path) -> Result<Vec<WebDatasetSample>> {
        let reader = WebDatasetReader {
            shard_paths: vec![path.to_path_buf()],
            current_shard: 0,
            buffer_size: 1,
            shuffle_buffer: None,
            cache: Arc::new(DashMap::new()),
        };
        
        if path.extension().and_then(|s| s.to_str()) == Some("gz") {
            reader.read_tar_gz(path)
        } else {
            reader.read_tar(path)
        }
    }
}

/// Async iterator over WebDataset samples
pub struct WebDatasetIterator {
    receiver: mpsc::Receiver<WebDatasetSample>,
}

impl WebDatasetIterator {
    /// Get next sample
    pub async fn next(&mut self) -> Option<WebDatasetSample> {
        self.receiver.recv().await
    }
}

/// WebDataset loader that converts samples to tensors
pub struct WebDatasetLoader {
    reader: WebDatasetReader,
    image_size: usize,
    device: Device,
    dtype: DType,
}

impl WebDatasetLoader {
    /// Create new WebDataset loader
    pub fn new(
        shard_pattern: &str,
        image_size: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let reader = WebDatasetReader::new(shard_pattern, 64)?;
        
        Ok(Self {
            reader,
            image_size,
            device,
            dtype,
        })
    }
    
    /// Load batch of samples
    pub async fn load_batch(&mut self, batch_size: usize) -> Result<DataBatch> {
        let mut images = Vec::with_capacity(batch_size);
        let mut captions = Vec::with_capacity(batch_size);
        let mut metadata = Vec::with_capacity(batch_size);
        
        let mut iter = self.reader.iter_samples().await;
        
        for _ in 0..batch_size {
            if let Some(sample) = iter.next().await {
                // Decode image
                let image = self.decode_image(&sample.image_data)?;
                images.push(image);
                captions.push(sample.caption);
                metadata.push(sample.metadata);
            } else {
                break;
            }
        }
        
        if images.is_empty() {
            return Err(Error::DataError("No more samples".into()));
        }
        
        // Stack into batch
        let candle_device = candle_core::Device::cuda_if_available(0)?;
        
        let image_batch = Tensor::stack(&images, 0)?;
        
        Ok(DataBatch {
            images: image_batch,
            captions,
            metadata,
        })
    }
    
    /// Decode image from bytes
    fn decode_image(&self, data: &[u8]) -> Result<Tensor> {
        use image::io::Reader as ImageReader;
        use std::io::Cursor;
        
        // Decode image
        let reader = ImageReader::new(Cursor::new(data))
            .with_guessed_format()
            .map_err(|e| Error::DataError(format!("Failed to guess image format: {}", e)))?;
        
        let img = reader.decode()
            .map_err(|e| Error::DataError(format!("Failed to decode image: {}", e)))?;
        
        // Resize to target size
        let resized = img.resize_exact(
            self.image_size as u32,
            self.image_size as u32,
            image::imageops::FilterType::Lanczos3,
        );
        
        // Convert to tensor
        let rgb = resized.to_rgb8();
        let (width, height) = rgb.dimensions();
        let pixels: Vec<f32> = rgb.into_raw()
            .into_iter()
            .map(|p| p as f32 / 255.0)
            .collect();
        
        // Convert to CHW format
        let mut chw_data = vec![0.0f32; 3 * (width * height) as usize];
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let src_idx = ((y * width + x) * 3 + c) as usize;
                    let dst_idx = (c * height * width + y * width + x) as usize;
                    chw_data[dst_idx] = pixels[src_idx];
                }
            }
        }
        
        let candle_device = candle_core::Device::cuda_if_available(0)?;
        
        Tensor::from_vec(
            chw_data,
            &[3, height as usize, width as usize],
            &candle_device,
        )?.to_dtype(self.dtype)
            .map_err(|e| Error::TensorError(e.to_string()))
    }
}

/// Batch of data from WebDataset
pub struct DataBatch {
    pub images: Tensor,
    pub captions: Vec<String>,
    pub metadata: Vec<Option<SampleMetadata>>,
}

/// Create WebDataset shards from a directory of images
pub struct WebDatasetWriter {
    shard_size: usize,
    output_dir: PathBuf,
    current_shard: usize,
    current_samples: Vec<WebDatasetSample>,
}

impl WebDatasetWriter {
    /// Create new WebDataset writer
    pub fn new(output_dir: PathBuf, shard_size: usize) -> Result<Self> {
        std::fs::create_dir_all(&output_dir)?;
        
        Ok(Self {
            shard_size,
            output_dir,
            current_shard: 0,
            current_samples: Vec::new(),
        })
    }
    
    /// Add sample to dataset
    pub fn add_sample(
        &mut self,
        key: String,
        image_path: &Path,
        caption: &str,
        metadata: Option<SampleMetadata>,
    ) -> Result<()> {
        // Read image data
        let image_data = std::fs::read(image_path)?;
        
        let sample = WebDatasetSample {
            key,
            image_data,
            caption: caption.to_string(),
            metadata,
        };
        
        self.current_samples.push(sample);
        
        // Write shard if full
        if self.current_samples.len() >= self.shard_size {
            self.write_shard()?;
        }
        
        Ok(())
    }
    
    /// Write current samples to shard
    fn write_shard(&mut self) -> Result<()> {
        if self.current_samples.is_empty() {
            return Ok(());
        }
        
        let shard_path = self.output_dir.join(format!("shard-{:06}.tar", self.current_shard));
        info!("Writing shard {} with {} samples", shard_path.display(), self.current_samples.len());
        
        let file = std::fs::File::create(&shard_path)?;
        let mut builder = tar::Builder::new(file);
        
        for sample in &self.current_samples {
            // Write image
            let image_name = format!("{}.jpg", sample.key);
            let mut header = tar::Header::new_gnu();
            header.set_path(&image_name)?;
            header.set_size(sample.image_data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, &sample.image_data[..])?;
            
            // Write caption
            let caption_name = format!("{}.txt", sample.key);
            let caption_data = sample.caption.as_bytes();
            let mut header = tar::Header::new_gnu();
            header.set_path(&caption_name)?;
            header.set_size(caption_data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, caption_data)?;
            
            // Write metadata if present
            if let Some(ref metadata) = sample.metadata {
                let metadata_name = format!("{}.json", sample.key);
                let metadata_data = serde_json::to_vec(metadata)?;
                let mut header = tar::Header::new_gnu();
                header.set_path(&metadata_name)?;
                header.set_size(metadata_data.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder.append(&header, &metadata_data[..])?;
            }
        }
        
        builder.finish()?;
        
        // Optionally compress
        if cfg!(feature = "gzip") {
            use flate2::write::GzEncoder;
            use flate2::Compression;
            
            let tar_data = std::fs::read(&shard_path)?;
            let gz_path = shard_path.with_extension("tar.gz");
            
            let gz_file = std::fs::File::create(&gz_path)?;
            let mut encoder = GzEncoder::new(gz_file, Compression::default());
            std::io::Write::write_all(&mut encoder, &tar_data)?;
            encoder.finish()?;
            
            // Remove uncompressed tar
            std::fs::remove_file(&shard_path)?;
        }
        
        self.current_samples.clear();
        self.current_shard += 1;
        
        Ok(())
    }
    
    /// Finalize dataset
    pub fn finalize(mut self) -> Result<()> {
        self.write_shard()?;
        info!("WebDataset creation complete: {} shards written", self.current_shard);
        Ok(())
    }
}