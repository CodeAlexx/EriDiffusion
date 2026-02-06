use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use flame_core::cuda::utils::cuda_mem_get_free_mb;
use flame_core::cuda_ops_ffi::CudaStream;
use flame_core::memory_pool::MEMORY_POOL;
use flame_core::staging::{arena_record_and_release, arena_reset, arena_reset_stats, arena_stats};
use flame_core::{device::Device, DType, Tensor};

use crate::loaders::lazy_safetensors::{LazyPrefixedLoader, LazySafetensorsLoader};
#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
use crate::loaders::mmdit_weights::capture_mmdit_cpu_snapshot_from_file;
use crate::loaders::mmdit_weights::{
    load_context_embedder, load_final_layer, load_patch_embed, load_timestep_embedder,
    load_vector_embedder, LoadMode, MmditLoadReport,
};
use crate::loaders::weight_loader::MMDiTMetadata;
use crate::models::mmdit_blocks::{
    DismantledBlock, FinalLayer, JointTransformerBlock, MMDiTConfig, PatchEmbed, QkNormKind,
    RoPE2D, SelfAttention, TimestepEmbedder, VectorEmbedder, MLP,
};
#[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
use crate::models::mmdit_cpu::MmditCpuSnapshot;
#[cfg(not(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots")))]
struct MmditCpuSnapshot;
use crate::ops::{LayerNorm, Linear, RMSNorm};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct BlockArenaProfile {
    pub block_index: usize,
    pub arena_peak_bytes: u64,
    pub arena_allocations: u64,
    pub arena_releases: u64,
    pub arena_bytes_requested: u64,
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn query_free_vram_mb() -> Option<u64> {
    cuda_mem_get_free_mb().map(|mb| mb as u64)
}

#[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]
fn query_free_vram_mb() -> Option<u64> {
    None
}

struct EnvVarGuard {
    key: &'static str,
    active: bool,
}

impl EnvVarGuard {
    fn set_if_missing(key: &'static str, value: &str) -> Option<Self> {
        if env::var_os(key).is_some() {
            None
        } else {
            env::set_var(key, value);
            Some(Self { key, active: true })
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if self.active {
            env::remove_var(self.key);
        }
    }
}

/// Build an `MMDiTConfig` suitable for streaming inference using the checkpoint metadata.
pub fn build_streaming_config(
    metadata: &MMDiTMetadata,
    loader: &LazySafetensorsLoader,
    device: &Device,
) -> Result<MMDiTConfig> {
    let mut config = MMDiTConfig::default();

    let hidden_size = if let Some(hidden) = metadata.hidden_size {
        hidden
    } else {
        infer_hidden_size(loader, device)?
    };
    let depth = if let Some(depth) = metadata.depth { depth } else { infer_depth(loader)? };
    let num_heads = if let Some(heads) = metadata.num_heads {
        heads
    } else {
        hidden_size.checked_div(64).ok_or_else(|| anyhow!("hidden size not divisible by 64"))?
    };

    config.hidden_size = hidden_size;
    config.num_heads = num_heads;
    config.depth = depth;
    if let Some(ratio) = metadata.mlp_ratio {
        config.mlp_ratio = ratio;
    }
    config.qk_norm = metadata.qk_norm;
    config.x_self_attn_layers = metadata.x_self_attn_layers;

    Ok(config)
}

fn infer_depth(loader: &LazySafetensorsLoader) -> Result<usize> {
    loader
        .keys()
        .filter_map(|key| {
            key.strip_prefix("model.diffusion_model.joint_blocks.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|idx| idx.parse::<usize>().ok())
        })
        .max()
        .map(|max_idx| max_idx + 1)
        .ok_or_else(|| anyhow!("unable to infer depth from checkpoint"))
}

fn infer_hidden_size(loader: &LazySafetensorsLoader, device: &Device) -> Result<usize> {
    let key = "model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight";
    let tensor =
        loader.load_tensor(key, device).with_context(|| format!("failed to load tensor {key}"))?;
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(anyhow!("unexpected qkv weight shape {:?}", dims));
    }
    Ok(dims[1])
}

enum WeightSource {
    Lazy(LazySafetensorsLoader),
    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    Snapshot(Arc<MmditCpuSnapshot>),
}

pub struct StreamingMMDiT {
    pub config: MMDiTConfig,
    device: Device,
    rope: RoPE2D,
    weight_source: WeightSource,
    block_cache: RefCell<HashMap<(bool, bool), JointTransformerBlock>>,
}

impl StreamingMMDiT {
    pub fn new(config: MMDiTConfig, loader: LazySafetensorsLoader, device: Device) -> Result<Self> {
        let rope =
            RoPE2D::new(config.hidden_size, config.num_heads, config.pos_embed_max_size, &device)?;
        Ok(Self {
            config,
            device,
            rope,
            weight_source: WeightSource::Lazy(loader),
            block_cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn from_checkpoint(config: MMDiTConfig, path: &Path, device: Device) -> Result<Self> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
        {
            match capture_mmdit_cpu_snapshot_from_file(path) {
                Ok((snapshot, report)) => {
                    report.log_summary("mmdit.snapshot");
                    let snapshot = Arc::new(snapshot);
                    let mut model = Self::from_snapshot(snapshot, device.clone())?;
                    model.config = config;
                    return Ok(model);
                }
                Err(err) => {
                    log::warn!(
                        "StreamingMMDiT::from_checkpoint: snapshot capture failed ({err}); falling back to lazy loader"
                    );
                }
            }
        }
        #[cfg(not(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots")))]
        let _ = path; // silence unused warning when snapshot path unused
        let loader = LazySafetensorsLoader::new(path)?;
        Self::new(config, loader, device)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    pub fn from_snapshot(snapshot: Arc<MmditCpuSnapshot>, device: Device) -> Result<Self> {
        let config = snapshot.config.clone();
        let rope =
            RoPE2D::new(config.hidden_size, config.num_heads, config.pos_embed_max_size, &device)?;
        Ok(Self {
            config,
            device,
            rope,
            weight_source: WeightSource::Snapshot(snapshot),
            block_cache: RefCell::new(HashMap::new()),
        })
    }

    #[cfg(not(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots")))]
    pub fn from_snapshot(_: Arc<MmditCpuSnapshot>, _: Device) -> Result<Self> {
        bail!("Snapshot-backed streaming requires the `cuda` and `bf16_u16` features")
    }

    fn lazy_loader(&self) -> Option<&LazySafetensorsLoader> {
        match &self.weight_source {
            WeightSource::Lazy(loader) => Some(loader),
            #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
            WeightSource::Snapshot(_) => None,
        }
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
    fn snapshot(&self) -> Option<&MmditCpuSnapshot> {
        if let WeightSource::Snapshot(snapshot) = &self.weight_source {
            Some(snapshot.as_ref())
        } else {
            None
        }
    }

    #[cfg(not(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots")))]
    fn snapshot(&self) -> Option<&MmditCpuSnapshot> {
        None
    }

    pub fn forward(
        &self,
        latents: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
        pooled: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_impl(latents, timesteps, context, pooled, None)
    }

    pub fn forward_profiled(
        &self,
        latents: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
        pooled: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<BlockArenaProfile>)> {
        let mut profile = Vec::with_capacity(self.config.depth);
        let output = self.forward_impl(latents, timesteps, context, pooled, Some(&mut profile))?;
        Ok((output, profile))
    }

    fn forward_impl(
        &self,
        latents: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
        pooled: Option<&Tensor>,
        mut profile: Option<&mut Vec<BlockArenaProfile>>,
    ) -> Result<Tensor> {
        let device = &self.device;
        let batch = latents.shape().dims()[0];
        let height = latents.shape().dims()[2];
        let width = latents.shape().dims()[3];
        let stream = CudaStream::from_raw(self.device.cuda_stream_raw_ptr());
        let ordinal = self.device.cuda_device().ordinal() as i32;

        let chunk_value = env::var("STREAMING_SDPA_CHUNK_MAX").unwrap_or_else(|_| "96".to_string());
        let _sdpa_chunk_guard = EnvVarGuard::set_if_missing("FLAME_SDPA_CHUNK_MAX", &chunk_value);
        let _sdpa_qchunk_guard = EnvVarGuard::set_if_missing("FLAME_SDPA_QCHUNK", &chunk_value);

        log::info!(
            "StreamingMMDiT::forward start: batch={} height={} width={} tokens={}",
            batch,
            height,
            width,
            (height * width) / (self.config.patch_size * self.config.patch_size)
        );

        // Patch embed
        let mut x_tokens = {
            let mut patch_embed = PatchEmbed::new(
                None,
                self.config.patch_size,
                self.config.in_channels,
                self.config.hidden_size,
                true,
                true,
                true,
                device,
            )?;
            self.load_patch_embed(&mut patch_embed)?;
            let mut latents_nhwc = latents.permute(&[0, 2, 3, 1])?;
            if latents_nhwc.dtype() != DType::BF16 || latents_nhwc.storage_dtype() != DType::BF16 {
                latents_nhwc = latents_nhwc.to_dtype(DType::BF16)?;
            }
            log::info!(
                "Streaming patch_embed latents_nhwc shape {:?}, weight shape {:?}",
                latents_nhwc.shape().dims(),
                patch_embed.weight_shape()
            );
            patch_embed.forward(&latents_nhwc)?
        };
        log::info!("StreamingMMDiT::forward patch embed done: shape {:?}", x_tokens.shape().dims());

        let grid_h = (height + self.config.patch_size - 1) / self.config.patch_size;
        let grid_w = (width + self.config.patch_size - 1) / self.config.patch_size;
        let pos_embed = self.rope.embed(
            grid_h,
            grid_w,
            batch,
            &Device::from(latents.device().clone()),
            x_tokens.dtype(),
        )?;
        x_tokens = x_tokens.add(&pos_embed)?;

        // Timestep embedder
        let mut cond = {
            let mut t_embedder = TimestepEmbedder::new(
                self.config.hidden_size,
                self.config.frequency_embedding_size,
                device,
            )?;
            self.load_timestep_embedder(&mut t_embedder)?;
            t_embedder.forward(timesteps)?
        };
        log::info!(
            "StreamingMMDiT::forward timestep embedder done: shape {:?}",
            cond.shape().dims()
        );

        // Optional pooled embedding
        if let (Some(dim), Some(pooled_tensor)) = (self.config.pooled_dim, pooled) {
            let mut vector_embedder = VectorEmbedder::new(dim, self.config.hidden_size, device)?;
            self.load_vector_embedder(&mut vector_embedder)?;
            let mut pooled_proj = vector_embedder.forward(pooled_tensor)?;
            let cond_dims = cond.shape().dims();
            let pooled_dims = pooled_proj.shape().dims();
            if pooled_dims.len() == 2
                && pooled_dims[0] == 1
                && cond_dims.len() == 2
                && cond_dims[0] != 1
            {
                pooled_proj = pooled_proj.expand(&[cond_dims[0], pooled_dims[1]])?;
            }
            cond = cond.add(&pooled_proj)?;
        }
        log::info!("StreamingMMDiT::forward pooled/context added");

        // Context embedder
        let mut context_tokens = {
            let mut context_embedder = Linear::new_zeroed(
                self.config.context_dim,
                self.config.hidden_size,
                true,
                &device.cuda_device(),
            )?;
            self.load_context_embedder(&mut context_embedder)?;
            context_embedder.forward(context)?
        };

        // Blocks
        let stop_at_block = env::var("STOP_AT_BLOCK").ok().and_then(|v| v.parse::<usize>().ok());
        let keep_blocks_resident = env::var("STREAMING_KEEP_BLOCKS")
            .ok()
            .map(|raw| {
                let lowered = raw.trim().to_ascii_lowercase();
                matches!(lowered.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false);
        let sync_after_block = env::var("STREAMING_SYNC_AFTER_BLOCK")
            .ok()
            .map(|raw| {
                let lowered = raw.trim().to_ascii_lowercase();
                matches!(lowered.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false);

        if !keep_blocks_resident {
            let mut cache = self.block_cache.borrow_mut();
            if !cache.is_empty() {
                log::debug!(
                    "StreamingMMDiT::forward clearing {} cached blocks before streaming run",
                    cache.len()
                );
                cache.clear();
            }
        }

        if profile.is_some() {
            arena_reset_stats(ordinal, &stream)?;
        }

        for idx in 0..self.config.depth {
            log::info!("StreamingMMDiT::forward block {}", idx);
            let context_pre_only = idx == self.config.depth - 1;
            let x_self_attn =
                self.config.x_self_attn_layers.map(|limit| idx <= limit).unwrap_or(false);
            match query_free_vram_mb() {
                Some(free_mb) => log::info!(
                    "StreamingMMDiT::forward block {} pre-alloc free_vram={} MiB",
                    idx,
                    free_mb
                ),
                None => log::warn!(
                    "StreamingMMDiT::forward block {} pre-alloc free_vram unavailable",
                    idx
                ),
            }
            MEMORY_POOL.clear_all_caches();
            let (cache_key, reused, mut block) = {
                let cache_key = (context_pre_only, x_self_attn);
                let cached = self.block_cache.borrow_mut().remove(&cache_key);
                let reused = cached.is_some();
                let block = if let Some(existing) = cached {
                    log::debug!(
                        "StreamingMMDiT::forward reusing cached block context_pre_only={} x_self_attn={}",
                        context_pre_only,
                        x_self_attn
                    );
                    existing
                } else {
                    log::debug!(
                        "StreamingMMDiT::forward constructing transient block context_pre_only={} x_self_attn={}",
                        context_pre_only,
                        x_self_attn
                    );
                    let mut created = JointTransformerBlock::with_flags(
                        self.config.hidden_size,
                        self.config.num_heads,
                        self.config.mlp_ratio,
                        self.config.qkv_bias,
                        self.config.qk_norm,
                        self.config.hidden_size,
                        context_pre_only,
                        x_self_attn,
                        device,
                    )?;
                    created.disable_grads();
                    created
                };
                (cache_key, reused, block)
            };

            self.load_block(idx, &mut block)
                .with_context(|| format!("load_block failed for layer {}", idx))?;
            block.disable_grads();
            if let Some(free_mb) = query_free_vram_mb() {
                log::debug!(
                    "StreamingMMDiT::forward block {} after load ({}) free_vram={} MiB",
                    idx,
                    if reused { "cached" } else { "transient" },
                    free_mb
                );
            }
            if let Some(limit) = stop_at_block {
                if idx >= limit {
                    anyhow::bail!("STOP_AT_BLOCK={} triggered", limit);
                }
            }
            let (new_x, new_context) = block
                .forward(&x_tokens, &context_tokens, &cond)
                .with_context(|| format!("forward failed for layer {}", idx))?;
            if keep_blocks_resident {
                let mut cache = self.block_cache.borrow_mut();
                log::debug!(
                    "StreamingMMDiT::forward retaining block {} for reuse (cache size -> {})",
                    idx,
                    cache.len() + 1
                );
                cache.insert(cache_key, block);
            } else {
                log::debug!(
                    "StreamingMMDiT::forward releasing block {} weights back to allocator",
                    idx
                );
                drop(block);
            }
            arena_record_and_release(ordinal, &stream)?;
            arena_reset(ordinal, &stream)?;
            MEMORY_POOL.clear_all_caches();
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            {
                flame_core::tensor_storage::clear_bf16_pool();
            }
            if let Some(profile_vec) = profile.as_mut() {
                let stats = arena_stats(ordinal, &stream)?;
                profile_vec.push(BlockArenaProfile {
                    block_index: idx,
                    arena_peak_bytes: stats.bytes_peak,
                    arena_allocations: stats.allocations,
                    arena_releases: stats.releases,
                    arena_bytes_requested: stats.bytes_requested,
                });
                arena_reset_stats(ordinal, &stream)?;
            }
            x_tokens = new_x;
            context_tokens = new_context;
            if sync_after_block {
                log::debug!("StreamingMMDiT::forward synchronizing device after block {}", idx);
                self.device
                    .synchronize()
                    .context("cudaDeviceSynchronize failed after streaming block")?;
            }
            match query_free_vram_mb() {
                Some(free_mb) => {
                    log::info!(
                        "StreamingMMDiT::forward block {} post-free_vram={} MiB",
                        idx,
                        free_mb
                    );
                }
                None => {
                    log::warn!("StreamingMMDiT::forward block {} post-free_vram unavailable", idx);
                }
            }
            log::debug!(
                "StreamingMMDiT::forward cache size after block {}: {}",
                idx,
                self.block_cache.borrow().len()
            );
        }

        // Final layer
        let output = {
            let mut final_layer = FinalLayer::new(
                self.config.hidden_size,
                self.config.patch_size,
                self.config.out_channels,
                device,
            )?;
            self.load_final_layer(&mut final_layer)?;
            final_layer.forward(&x_tokens, &cond, height, width)?
        };
        log::debug!(
            "StreamingMMDiT::forward completed with {} cached blocks",
            self.block_cache.borrow().len()
        );
        log::info!("StreamingMMDiT::forward completed: output shape {:?}", output.shape().dims());
        Ok(output)
    }

    fn load_patch_embed(&self, patch: &mut PatchEmbed) -> Result<()> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
        if let Some(snapshot) = self.snapshot() {
            patch.apply_cpu_snapshot(&snapshot.patch_embed)?;
            return Ok(());
        }

        let loader = self.root_loader();
        let mut report = MmditLoadReport::default();
        load_patch_embed(patch, &loader, LoadMode::Copy, &mut report)
            .context("failed to load patch embed weights")
    }

    fn load_timestep_embedder(&self, embedder: &mut TimestepEmbedder) -> Result<()> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
        if let Some(snapshot) = self.snapshot() {
            embedder.apply_cpu_snapshot(&snapshot.timestep_embedder)?;
            return Ok(());
        }

        let loader = self.root_loader();
        let mut report = MmditLoadReport::default();
        load_timestep_embedder(embedder, &loader, LoadMode::Copy, &mut report)
            .context("failed to load timestep embedder")
    }

    fn load_vector_embedder(&self, embedder: &mut VectorEmbedder) -> Result<()> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
        if let Some(snapshot) = self.snapshot() {
            if let Some(vec_snapshot) = snapshot.vector_embedder.as_ref() {
                embedder.apply_cpu_snapshot(vec_snapshot)?;
            }
            return Ok(());
        }

        let loader = self.root_loader();
        let mut report = MmditLoadReport::default();
        load_vector_embedder(embedder, &loader, LoadMode::Copy, &mut report)
            .context("failed to load vector embedder")
    }

    fn load_context_embedder(&self, linear: &mut Linear) -> Result<()> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
        if let Some(snapshot) = self.snapshot() {
            snapshot.context_embedder.apply_to(linear)?;
            return Ok(());
        }

        let loader = self.root_loader();
        let mut report = MmditLoadReport::default();
        load_context_embedder(linear, &loader, LoadMode::Copy, &mut report)
            .context("failed to load context embedder")
    }

    fn load_final_layer(&self, final_layer: &mut FinalLayer) -> Result<()> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
        if let Some(snapshot) = self.snapshot() {
            final_layer.apply_cpu_snapshot(&snapshot.final_layer)?;
            return Ok(());
        }

        let loader = self.root_loader();
        let mut report = MmditLoadReport::default();
        load_final_layer(final_layer, &loader, LoadMode::Copy, &mut report)
            .context("failed to load final layer")
    }

    fn load_block(&self, index: usize, block: &mut JointTransformerBlock) -> Result<()> {
        #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
        if let Some(snapshot) = self.snapshot() {
            let block_snapshot = snapshot
                .blocks
                .get(index)
                .ok_or_else(|| anyhow!("snapshot missing block {}", index))?;
            block.apply_cpu_snapshot(block_snapshot)?;
            return Ok(());
        }

        self.load_block_lazy(index, block)
    }

    fn load_block_lazy(&self, index: usize, block: &mut JointTransformerBlock) -> Result<()> {
        let base = format!("model.diffusion_model.joint_blocks.{}", index);
        self.load_dismantled_block_lazy(
            &format!("{}.context_block", base),
            block.context_block_mut(),
        )
        .with_context(|| format!("failed to load context block {index}"))?;
        self.load_dismantled_block_lazy(&format!("{}.x_block", base), block.x_block_mut())
            .with_context(|| format!("failed to load x block {index}"))?;
        Ok(())
    }

    fn load_dismantled_block_lazy(&self, prefix: &str, block: &mut DismantledBlock) -> Result<()> {
        self.load_layer_norm_lazy(&format!("{}.norm1", prefix), block.norm1_mut())?;

        if let Some(norm2) = block.norm2_mut() {
            self.load_layer_norm_lazy(&format!("{}.norm2", prefix), norm2)?;
        }

        self.load_linear_lazy(&format!("{}.adaLN_modulation.1", prefix), block.modulation_mut())?;

        self.load_self_attention_lazy(&format!("{}.attn", prefix), block.attn_mut())?;

        if let Some(attn2) = block.attn2_mut() {
            self.load_self_attention_lazy(&format!("{}.attn2", prefix), attn2)?;
        }

        if let Some(mlp) = block.mlp_mut() {
            let (fc1, fc2) = mlp.fc_layers_mut();
            self.load_linear_lazy(&format!("{}.mlp.fc1", prefix), fc1)?;
            self.load_linear_lazy(&format!("{}.mlp.fc2", prefix), fc2)?;
        }

        Ok(())
    }

    fn load_linear_lazy(&self, key_prefix: &str, linear: &mut Linear) -> Result<()> {
        let loader = self.lazy_loader().expect("lazy loader required");
        let weight_key = format!("{}.weight", key_prefix);
        let weight_data = loader
            .load_tensor_bf16(&weight_key)
            .with_context(|| format!("failed to load {weight_key}"))?;

        if weight_data.len() != linear.weight.shape().elem_count() {
            anyhow::bail!(
                "{}: weight length {} mismatches expected {}",
                weight_key,
                weight_data.len(),
                linear.weight.shape().elem_count()
            );
        }
        linear
            .weight
            .copy_from_bf16_slice(&weight_data)
            .with_context(|| format!("failed to copy {weight_key}"))?;

        if let Some(bias) = linear.bias.as_mut() {
            let bias_key = format!("{}.bias", key_prefix);
            if loader.has_key(&bias_key) {
                let bias_data = loader
                    .load_tensor_bf16(&bias_key)
                    .with_context(|| format!("failed to load {bias_key}"))?;
                if bias_data.len() != bias.shape().elem_count() {
                    anyhow::bail!(
                        "{}: bias length {} mismatches expected {}",
                        bias_key,
                        bias_data.len(),
                        bias.shape().elem_count()
                    );
                }
                bias.copy_from_bf16_slice(&bias_data)
                    .with_context(|| format!("failed to copy {bias_key}"))?;
            }
        }

        Ok(())
    }

    fn load_layer_norm_lazy(&self, prefix: &str, norm: &mut LayerNorm) -> Result<()> {
        let loader = self.lazy_loader().expect("lazy loader required");
        if let Some(weight) = norm.weight.as_mut() {
            let key = format!("{}.weight", prefix);
            if loader.has_key(&key) {
                let data = loader
                    .load_tensor_bf16(&key)
                    .with_context(|| format!("failed to load {key}"))?;
                if data.len() != weight.shape().elem_count() {
                    anyhow::bail!(
                        "{}: weight length {} mismatches expected {}",
                        key,
                        data.len(),
                        weight.shape().elem_count()
                    );
                }
                weight
                    .copy_from_bf16_slice(&data)
                    .with_context(|| format!("failed to copy {key}"))?;
            }
        }

        if let Some(bias) = norm.bias.as_mut() {
            let key = format!("{}.bias", prefix);
            if loader.has_key(&key) {
                let data = loader
                    .load_tensor_bf16(&key)
                    .with_context(|| format!("failed to load {key}"))?;
                if data.len() != bias.shape().elem_count() {
                    anyhow::bail!(
                        "{}: bias length {} mismatches expected {}",
                        key,
                        data.len(),
                        bias.shape().elem_count()
                    );
                }
                bias.copy_from_bf16_slice(&data)
                    .with_context(|| format!("failed to copy {key}"))?;
            }
        }

        Ok(())
    }

    fn load_rms_norm_lazy(&self, prefix: &str, norm: &mut RMSNorm) -> Result<()> {
        let loader = self.lazy_loader().expect("lazy loader required");
        if let Some(weight) = norm.weight.as_mut() {
            let key = format!("{}.weight", prefix);
            if loader.has_key(&key) {
                let data = loader
                    .load_tensor_bf16(&key)
                    .with_context(|| format!("failed to load {key}"))?;
                if data.len() != weight.shape().elem_count() {
                    anyhow::bail!(
                        "{}: weight length {} mismatches expected {}",
                        key,
                        data.len(),
                        weight.shape().elem_count()
                    );
                }
                weight
                    .copy_from_bf16_slice(&data)
                    .with_context(|| format!("failed to copy {key}"))?;
            }
        }
        Ok(())
    }

    fn load_self_attention_lazy(&self, prefix: &str, attn: &mut SelfAttention) -> Result<()> {
        let loader = self.lazy_loader().expect("lazy loader required");
        let weight_key = format!("{}.qkv.weight", prefix);
        let weight_data = loader
            .load_tensor_bf16(&weight_key)
            .with_context(|| format!("failed to load {weight_key}"))?;
        let hidden = attn.hidden_size();
        let chunk = hidden * attn.in_features();
        if weight_data.len() != chunk * 3 {
            anyhow::bail!(
                "{}: expected {} elements, got {}",
                weight_key,
                chunk * 3,
                weight_data.len()
            );
        }
        {
            let q_proj = attn.q_proj_mut();
            q_proj
                .weight
                .copy_from_bf16_slice(&weight_data[..chunk])
                .with_context(|| format!("failed to copy {weight_key} (q)"))?;
        }
        {
            let k_proj = attn.k_proj_mut();
            k_proj
                .weight
                .copy_from_bf16_slice(&weight_data[chunk..chunk * 2])
                .with_context(|| format!("failed to copy {weight_key} (k)"))?;
        }
        {
            let v_proj = attn.v_proj_mut();
            v_proj
                .weight
                .copy_from_bf16_slice(&weight_data[chunk * 2..])
                .with_context(|| format!("failed to copy {weight_key} (v)"))?;
        }

        let has_bias = attn.q_proj_mut().bias.is_some();
        if has_bias {
            let bias_key = format!("{}.qkv.bias", prefix);
            if loader.has_key(&bias_key) {
                let bias_data = loader
                    .load_tensor_bf16(&bias_key)
                    .with_context(|| format!("failed to load {bias_key}"))?;
                if bias_data.len() != hidden * 3 {
                    anyhow::bail!(
                        "{}: expected {} elements, got {}",
                        bias_key,
                        hidden * 3,
                        bias_data.len()
                    );
                }
                {
                    let q_bias = attn.q_proj_mut().bias.as_mut();
                    if let Some(bias) = q_bias {
                        bias.copy_from_bf16_slice(&bias_data[..hidden])
                            .with_context(|| format!("failed to copy {bias_key} (q)"))?;
                    }
                }
                {
                    let k_bias = attn.k_proj_mut().bias.as_mut();
                    if let Some(bias) = k_bias {
                        bias.copy_from_bf16_slice(&bias_data[hidden..hidden * 2])
                            .with_context(|| format!("failed to copy {bias_key} (k)"))?;
                    }
                }
                {
                    let v_bias = attn.v_proj_mut().bias.as_mut();
                    if let Some(bias) = v_bias {
                        bias.copy_from_bf16_slice(&bias_data[hidden * 2..])
                            .with_context(|| format!("failed to copy {bias_key} (v)"))?;
                    }
                }
            }
        }

        let qk_norm = attn.qk_norm_mut();
        match qk_norm.kind() {
            QkNormKind::Layer => {
                let (norm_q, norm_k) = qk_norm.layer_norms_mut();
                if let Some(norm_q) = norm_q {
                    self.load_layer_norm_lazy(&format!("{}.ln_q", prefix), norm_q)?;
                }
                if let Some(norm_k) = norm_k {
                    self.load_layer_norm_lazy(&format!("{}.ln_k", prefix), norm_k)?;
                }
            }
            QkNormKind::Rms => {
                let (rms_q, rms_k) = qk_norm.rms_norms_mut();
                if let Some(rms_q) = rms_q {
                    self.load_rms_norm_lazy(&format!("{}.ln_q", prefix), rms_q)?;
                }
                if let Some(rms_k) = rms_k {
                    self.load_rms_norm_lazy(&format!("{}.ln_k", prefix), rms_k)?;
                }
            }
            QkNormKind::Disabled => {}
        }

        if let Some(proj) = attn.proj_mut() {
            self.load_linear_lazy(&format!("{}.proj", prefix), proj)?;
        }

        Ok(())
    }

    fn root_loader(&self) -> LazyPrefixedLoader<'_> {
        match &self.weight_source {
            WeightSource::Lazy(loader) => {
                LazyPrefixedLoader::new(loader, "model.diffusion_model", self.device.clone())
            }
            #[cfg(all(feature = "cuda", feature = "bf16_u16", feature = "cpu_snapshots"))]
            WeightSource::Snapshot(_) => {
                panic!("lazy loader requested for snapshot-backed StreamingMMDiT")
            }
        }
    }
}
