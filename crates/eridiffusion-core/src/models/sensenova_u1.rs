//! SenseNova-U1-8B-MoT — pure-Rust T2I inference.
//!
//! SCAFFOLD ONLY. Multi-session port. Layer + sampler bodies are stubs that
//! `todo!` with specific references back to the Python source under
//! `/home/alex/SenseNova-U1/src/sensenova_u1/models/neo_unify/`.
//!
//! ============================================================================
//! ARCHITECTURE (verified against reference 2026-04-28; supersedes any prior
//! handoff doc on /home/alex/EriDiffusion/HANDOFF_2026-04-28_SENSENOVA_U1_PORT.md)
//! ============================================================================
//!
//! T2I FLOW (modeling_neo_chat.py::t2i_generate, lines 1578-1730):
//!
//!   1. Build system+user prompt query → tokenize → text indexes (t-axis only).
//!   2. Build empty-prompt query for CFG-uncond → tokenize → uncond indexes.
//!   3. Run **prefix forward** for both queries through 42-layer Qwen3 with
//!      base weights only. This populates a KV cache (per-layer, K and V).
//!      `_t2i_prefix_forward` returns `(past_key_values, last_hidden_state)`.
//!      Two parallel caches: cache_cond + cache_uncond.
//!   4. Initialize image_prediction = noise_scale * randn(B, 3, H, W).
//!   5. Compute Euler ODE timesteps with exponential time-shift schedule:
//!         sigma = 1 - t
//!         shift = timestep_shift             # CLI default 3.0 (NOT 1.0)
//!         sigma = shift*sigma / (1 + (shift-1)*sigma)
//!         t = 1 - sigma
//!   6. For step in 0..num_steps:
//!         a. patchify image at patch*merge=32 → z (B, L, 32*32*3=3072).
//!         b. patchify image at patch=16, channel_first → image_input (B, N=4*L, 768).
//!         c. extract_feature(image_input, gen_model=True) — runs vision_model_mot_gen
//!            (Conv2d k16s16 + 2D-RoPE + Conv2d k2s2 dense_embedding) → (B, L, 4096).
//!         d. timestep_embedder(t) + (optional) noise_scale_embedder(scale) added to gen tokens.
//!         e. Run **gen step forward**: per-token gen path through 42 layers with
//!            `_mot_gen` weights. Each layer concatenates current K/V with
//!            cached text K/V (no cache update) before SDPA. Two passes:
//!            cond (with text cache) and uncond (with empty-text cache).
//!         f. fm_head([B, L, 4096]) → x_pred [B, L, 3072].
//!         g. v = (x_pred - z) / max(1 - t, t_eps=0.05).
//!         h. CFG: v = v_uncond + cfg_scale*(v_cond - v_uncond)  (cfg_norm='none').
//!         i. Euler step: z_next = z + (t_next - t) * v.
//!         j. Unpatchify z_next at patch*merge=32 → image_prediction.
//!   7. Denormalize: image = (image * 0.5 + 0.5).clamp(0, 1) → PNG.
//!
//! ============================================================================
//! 3D ROPE (modeling_qwen3.py::Qwen3Attention.forward_*, lines 422-736)
//! ============================================================================
//!
//! Each attention head has head_dim=128, split as:
//!     |      t (64 dims)      |   h (32 dims)   |   w (32 dims)   |
//!     ↑                       ↑                 ↑
//!     q_norm  → 1D RoPE θ=5e6 q_norm_hw[h half] q_norm_hw[w half]
//!     k_norm  → on t-axis     → 1D RoPE θ=1e4   → 1D RoPE θ=1e4
//!     k_norm                  on h-axis index   on w-axis index
//!
//! Concretely (forward_gen, lines 593-621):
//!     query_states = q_proj_mot_gen(hidden).view(*shape, num_heads, 128)
//!     query_t, query_hw = chunk(query, 2, dim=-1)            # [..., H, 64] each
//!     query_t = q_norm_mot_gen(query_t)                       # weight shape [64]
//!     query_hw = q_norm_hw_mot_gen(query_hw)                  # weight shape [64]
//!     query_h, query_w = chunk(query_hw, 2, dim=-1)           # [..., H, 32] each
//!     # Three independent RoPE applications on t/h/w slices:
//!     query_t = apply_rope(query_t, cos_t(idx_t), sin_t(idx_t))     # half-split, θ=5e6
//!     query_h = apply_rope(query_h, cos_h(idx_h), sin_h(idx_h))     # half-split, θ=1e4
//!     query_w = apply_rope(query_w, cos_w(idx_w), sin_w(idx_w))     # half-split, θ=1e4
//!     query  = concat([query_t, query_h, query_w], dim=-1)    # [..., H, 128]
//!     # Same for keys. Values are NOT RoPE'd.
//!
//! For text tokens: idx_t = position, idx_h = idx_w = 0 (they reside at row 0, col 0).
//! For image tokens (built in `_build_t2i_image_indexes`, lines 452-457):
//!     idx_t = text_len  (constant — gen tokens append at the same t-position)
//!     idx_h = patch_index // token_w
//!     idx_w = patch_index %  token_w
//!
//! ============================================================================
//! PER-LAYER ROUTING — TWO MODES, NEVER MIXED FOR T2I
//! ============================================================================
//!
//! Qwen3DecoderLayer.forward (modeling_qwen3.py:854) routes by the
//! image_gen_indicators mask:
//!   - all-text (prefix): forward_und → uses base weights {input_layernorm, q_proj,
//!     k_proj, v_proj, o_proj, q_norm, q_norm_hw, k_norm, k_norm_hw,
//!     post_attention_layernorm, mlp.{gate,up,down}_proj}. PASS update_cache=True.
//!   - all-image (per step): forward_gen → uses _mot_gen weights, PASS
//!     update_cache=False so the prefix K/V is preserved across all 50 steps.
//!     Current K/V are concatenated with the cached prefix K/V before SDPA.
//!
//! For T2I we only ever hit those two pure-mode branches. The mixed-mode
//! `forward` (used by it2i editing / interleaved gen) is OUT OF SCOPE here.
//!
//! ============================================================================
//! WEIGHT KEYS — VERIFIED FROM model.safetensors.index.json
//! ============================================================================
//!
//! Per layer i ∈ [0, 42), 26 tensors (13 base + 13 _mot_gen):
//!   language_model.model.layers.{i}.input_layernorm.weight                        [4096]
//!   language_model.model.layers.{i}.input_layernorm_mot_gen.weight                [4096]
//!   language_model.model.layers.{i}.post_attention_layernorm.weight               [4096]
//!   language_model.model.layers.{i}.post_attention_layernorm_mot_gen.weight       [4096]
//!   language_model.model.layers.{i}.self_attn.q_proj.weight                       [4096, 4096]
//!   language_model.model.layers.{i}.self_attn.q_proj_mot_gen.weight               [4096, 4096]
//!   language_model.model.layers.{i}.self_attn.k_proj.weight                       [1024, 4096]
//!   language_model.model.layers.{i}.self_attn.k_proj_mot_gen.weight               [1024, 4096]
//!   language_model.model.layers.{i}.self_attn.v_proj.weight                       [1024, 4096]
//!   language_model.model.layers.{i}.self_attn.v_proj_mot_gen.weight               [1024, 4096]
//!   language_model.model.layers.{i}.self_attn.o_proj.weight                       [4096, 4096]
//!   language_model.model.layers.{i}.self_attn.o_proj_mot_gen.weight               [4096, 4096]
//!   language_model.model.layers.{i}.self_attn.q_norm.weight                       [64]
//!   language_model.model.layers.{i}.self_attn.q_norm_mot_gen.weight               [64]
//!   language_model.model.layers.{i}.self_attn.q_norm_hw.weight                    [64]
//!   language_model.model.layers.{i}.self_attn.q_norm_hw_mot_gen.weight            [64]
//!   language_model.model.layers.{i}.self_attn.k_norm.weight                       [64]
//!   language_model.model.layers.{i}.self_attn.k_norm_mot_gen.weight               [64]
//!   language_model.model.layers.{i}.self_attn.k_norm_hw.weight                    [64]
//!   language_model.model.layers.{i}.self_attn.k_norm_hw_mot_gen.weight            [64]
//!   language_model.model.layers.{i}.mlp.gate_proj.weight                          [12288, 4096]
//!   language_model.model.layers.{i}.mlp.up_proj.weight                            [12288, 4096]
//!   language_model.model.layers.{i}.mlp.down_proj.weight                          [4096, 12288]
//!   language_model.model.layers.{i}.mlp_mot_gen.gate_proj.weight                  [12288, 4096]
//!   language_model.model.layers.{i}.mlp_mot_gen.up_proj.weight                    [12288, 4096]
//!   language_model.model.layers.{i}.mlp_mot_gen.down_proj.weight                  [4096, 12288]
//! NB: q_norm/k_norm shapes are [head_dim/2 = 64], NOT [128]. Per
//! Qwen3RMSNorm(self.head_dim // 2) at modeling_qwen3.py:400-408.
//!
//! Shared (resident, total 24+ tensors):
//!   language_model.model.embed_tokens.weight                                       [151936, 4096]
//!   language_model.model.norm.weight                                                [4096]
//!   language_model.model.norm_mot_gen.weight                                        [4096]
//!   language_model.lm_head.weight                                                   [151936, 4096]
//!   fm_modules.timestep_embedder.mlp.{0,2}.{weight,bias}                           in: 256, hidden+out: 4096
//!   fm_modules.noise_scale_embedder.mlp.{0,2}.{weight,bias}                        in: 256, hidden+out: 4096
//!   fm_modules.fm_head.{0,2}.{weight,bias}                                         [4096→1536→3072]
//!   fm_modules.vision_model_mot_gen.embeddings.patch_embedding.{weight,bias}       Conv2d(3, 1024, k=16, s=16)
//!   fm_modules.vision_model_mot_gen.embeddings.dense_embedding.{weight,bias}       Conv2d(1024, 4096, k=2, s=2)
//!   vision_model.embeddings.patch_embedding.{weight,bias}                          (UNUSED for T2I — understanding only)
//!   vision_model.embeddings.dense_embedding.{weight,bias}                          (UNUSED for T2I)

// Vendored from `inference-flame/src/models/sensenova_u1.rs` 2026-05-10.
// Two imports swapped to EDv2-internal equivalents; everything else is
// byte-equivalent to the inference-flame version. flame_diffusion's
// BlockFacilitator + BlockOffloader were the only flame-diffusion types
// the file referenced; EDv2 carries the same trait + struct under
// `crate::training::block_offload::*`.
use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use crate::training::block_offload::{BlockFacilitator, BlockOffloader};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Per-layer key router for the BlockOffloader: anything under
/// `language_model.model.layers.{i}.` belongs to block `i` (covers both base
/// and `_mot_gen` variants — they ride the same `layers.{i}.` prefix).
struct SenseNovaFacilitator {
    num_blocks: usize,
}

impl BlockFacilitator for SenseNovaFacilitator {
    fn block_count(&self) -> usize {
        self.num_blocks
    }
    fn classify_key(&self, key: &str) -> Option<usize> {
        classify_layer_key(key)
    }
}

// ---------------------------------------------------------------------------
// Config (parsed from /home/alex/.serenity/models/sensenova_u1/config.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SenseNovaU1Config {
    // ---- LLM (Qwen3 backbone) ----
    pub vocab_size: usize,                 // 151936
    pub hidden_size: usize,                // 4096
    pub num_layers: usize,                 // 42
    pub intermediate_size: usize,          // 12288
    pub num_heads: usize,                  // 32
    pub num_kv_heads: usize,               // 8
    pub head_dim: usize,                   // 128
    pub rms_norm_eps: f32,                 // 1e-6
    pub rope_theta: f64,                   // 5_000_000.0  (1D, t-axis, text/temporal)
    pub rope_theta_hw: f64,                // 10_000.0     (1D each, h-axis & w-axis)
    pub max_position_embeddings: usize,    // 262144 (t-axis)
    pub max_position_embeddings_hw: usize, // 10000  (h/w axes)
    // Token IDs (sourced from config.json + tokenizer_config.json):
    pub bos_token_id: i64,                 // 151643
    pub eos_token_id: i64,                 // 151645
    pub pad_token_id: i64,                 // 151643

    // ---- Image / patching ----
    pub patch_size: usize,        // 16
    pub downsample_ratio: f32,    // 0.5  ⇒ merge_size = 1/0.5 = 2 (2×2 patch merge)

    // ---- Vision-model gen path (NEOVisionEmbeddings under fm_modules.vision_model_mot_gen) ----
    pub vision_hidden_size: usize,        // 1024
    pub rope_theta_vision: f64,           // 10_000.0
    pub max_position_embeddings_vision: usize, // 10000

    // ---- Flow-matching ----
    pub timestep_shift_train: f32,        // 1.0  (config "timestep_shift")  — training default
    pub time_schedule: TimeSchedule,      // "standard"
    pub time_shift_type: TimeShiftType,   // "exponential"
    pub base_shift: f32,                  // 0.5
    pub max_shift: f32,                   // 1.15
    pub base_image_seq_len: usize,        // 64
    pub max_image_seq_len: usize,         // 4096
    pub noise_scale_mode: NoiseScaleMode, // "resolution"
    pub noise_scale_base_image_seq_len: usize, // 64
    pub add_noise_scale_embedding: bool,  // true
    pub noise_scale_max_value: f32,       // 8.0
    pub noise_scale: f32,                 // 1.0
    pub t_eps: f32,                       // 0.05

    // ---- fm_head ----
    pub fm_head_dim: usize,        // 1536
    pub fm_head_layers: usize,     // 2
    pub use_pixel_head: bool,      // false
    pub use_deep_fm_head: bool,    // false (config is silent → defaults to false)
    pub use_adaln: bool,           // false
}

impl Default for SenseNovaU1Config {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 4096,
            num_layers: 42,
            intermediate_size: 12288,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            rope_theta_hw: 10_000.0,
            max_position_embeddings: 262144,
            max_position_embeddings_hw: 10_000,
            bos_token_id: 151643,
            eos_token_id: 151645,
            pad_token_id: 151643,

            patch_size: 16,
            downsample_ratio: 0.5,

            vision_hidden_size: 1024,
            rope_theta_vision: 10_000.0,
            max_position_embeddings_vision: 10_000,

            timestep_shift_train: 1.0,
            time_schedule: TimeSchedule::Standard,
            time_shift_type: TimeShiftType::Exponential,
            base_shift: 0.5,
            max_shift: 1.15,
            base_image_seq_len: 64,
            max_image_seq_len: 4096,
            noise_scale_mode: NoiseScaleMode::Resolution,
            noise_scale_base_image_seq_len: 64,
            add_noise_scale_embedding: true,
            noise_scale_max_value: 8.0,
            noise_scale: 1.0,
            t_eps: 0.05,

            fm_head_dim: 1536,
            fm_head_layers: 2,
            use_pixel_head: false,
            use_deep_fm_head: false,
            use_adaln: false,
        }
    }
}

impl SenseNovaU1Config {
    /// `merge_size = round(1 / downsample_ratio)` — 2 for the 8B-MoT checkpoint.
    /// Used at every patchify/unpatchify call site and to derive the gen-token
    /// grid from the pixel grid.
    #[inline]
    pub fn merge_size(&self) -> usize {
        (1.0 / self.downsample_ratio).round() as usize
    }

    /// fm_head output dimension = (patch_size * merge_size)^2 * 3.
    /// At default (16 * 2)^2 * 3 = 3072.
    #[inline]
    pub fn fm_head_out_dim(&self) -> usize {
        let p = self.patch_size * self.merge_size();
        p * p * 3
    }

    /// Per-axis RoPE dim split. From modeling_qwen3.py:
    ///   t-half   = head_dim / 2     (64 dims, RoPE θ=rope_theta)
    ///   h-half   = head_dim / 4     (32 dims, RoPE θ=rope_theta_hw on row idx)
    ///   w-half   = head_dim / 4     (32 dims, RoPE θ=rope_theta_hw on col idx)
    #[inline]
    pub fn rope_dims(&self) -> (usize, usize, usize) {
        (self.head_dim / 2, self.head_dim / 4, self.head_dim / 4)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSchedule {
    Standard,
    Dynamic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeShiftType {
    Exponential,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseScaleMode {
    Static,
    Resolution,
    Dynamic,
    DynamicSqrt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgNorm {
    None,
    Global,
    Channel,
    CfgZeroStar,
}

// ---------------------------------------------------------------------------
// KV cache (cross-step, per-layer; one cache per CFG stream — cond + uncond)
// ---------------------------------------------------------------------------
//
// Pattern reference: `acestep_dit.rs::cross_kv_cache` (Vec<(Tensor, Tensor)>).
// Difference here: the cache is populated ONCE by `forward_und` and then read
// (without update) by every per-step `forward_gen` call across 50 ODE steps.
//
// Reference Python: modeling_qwen3.py forward_und path that calls
// `past_key_values.update(K, V, layer_idx, cache_kwargs=None)`, and
// forward_gen path that, when update_cache=False, concatenates the cached
// (K, V) with the current step's (K, V) before attention.
//
// Shapes per layer:
//   K, V : [B, num_kv_heads=8, prefix_len, head_dim=128], BF16

/// Per-layer (K, V) cache populated by `forward_und` (text prefix) and read
/// across all ODE steps by `forward_gen` without modification.
#[derive(Clone)]
pub struct KvCache {
    /// One (K, V) entry per layer, in layer-index order. Shapes:
    /// `K`: [B, num_kv_heads, prefix_len, head_dim], BF16.
    /// `V`: [B, num_kv_heads, prefix_len, head_dim], BF16.
    pub layers: Vec<(Tensor, Tensor)>,
    /// Number of tokens that produced the cache (== K.shape(2)).
    pub prefix_len: usize,
    /// `t_index` (in the model's spatiotemporal RoPE sense) to assign to the
    /// FIRST decoded token after this prefix. For a text-only prefix
    /// `next_t_index == prefix_len`. For a mixed prefix it equals
    /// `max(t_indexes_of_prefix) + 1`, which is `prefix_len - L_image_context`
    /// because each `<IMG_CONTEXT>` block consumes only one t-axis slot.
    pub next_t_index: usize,
}

// ---------------------------------------------------------------------------
// Top-level model struct
// ---------------------------------------------------------------------------

/// SenseNova-U1-8B-MoT inference module.
///
/// SenseNova-U1 8B-MoT runtime model. Per-layer transformer weights stream
/// from pinned host RAM via `BlockOffloader`; shared weights (embed tokens,
/// final norms, fm_modules, vision_model_mot_gen embedder) are resident.
pub struct SenseNovaU1 {
    pub(crate) config: SenseNovaU1Config,
    pub(crate) shared: HashMap<String, Tensor>,
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) offloader: BlockOffloader,
}

impl SenseNovaU1 {
    pub fn config(&self) -> &SenseNovaU1Config { &self.config }
    pub fn device(&self) -> &Arc<CudaDevice> { &self.device }

    // -----------------------------------------------------------------------
    // Step A: Loader (Phase 5 will swap the resident layer storage for a
    // BlockOffloader; the public load() signature stays the same.)
    // -----------------------------------------------------------------------

    /// Load all weights from the canonical 8-shard checkpoint at `weights_dir`.
    ///
    /// The directory must contain `model.safetensors.index.json` plus the
    /// `model-{NNNNN}-of-{TOTAL}.safetensors` shards it references. Every key
    /// is loaded into either `shared` (matches `SHARED_PREFIXES`) or
    /// `layers[i]` (per-layer transformer weights, classified by
    /// `classify_layer_key`). Crashes with a clear message on any key that
    /// matches neither category, or on missing expected keys.
    pub fn load(weights_dir: &Path, device: &Arc<CudaDevice>) -> Result<Self> {
        let config = SenseNovaU1Config::default();

        // 1) Read the safetensors index to discover shards.
        let index_path = weights_dir.join("model.safetensors.index.json");
        let index_text = std::fs::read_to_string(&index_path).map_err(|e| {
            Error::Io(format!(
                "SenseNovaU1: cannot read index json at {:?}: {e}",
                index_path
            ))
        })?;
        let index: serde_json::Value = serde_json::from_str(&index_text).map_err(|e| {
            Error::InvalidInput(format!(
                "SenseNovaU1: malformed index json at {:?}: {e}",
                index_path
            ))
        })?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "SenseNovaU1: index json at {:?} missing 'weight_map'",
                    index_path
                ))
            })?;

        // Collect the set of unique shard filenames (stable sorted order).
        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        shard_names.sort();
        let shard_paths: Vec<PathBuf> = shard_names
            .iter()
            .map(|n| weights_dir.join(n))
            .collect();
        let shard_strs: Vec<String> = shard_paths
            .iter()
            .map(|p| {
                p.to_str()
                    .map(str::to_string)
                    .ok_or_else(|| Error::Io(format!("non-utf8 shard path: {:?}", p)))
            })
            .collect::<Result<Vec<_>>>()?;
        let shard_refs: Vec<&str> = shard_strs.iter().map(|s| s.as_str()).collect();

        // 2) Stream per-layer weights via BlockOffloader (pinned host RAM →
        //    H2D one block at a time during forward).
        let facilitator = SenseNovaFacilitator {
            num_blocks: config.num_layers,
        };
        let offloader = BlockOffloader::load(&shard_refs, &facilitator, device.clone())
            .map_err(|e| {
                Error::InvalidInput(format!("SenseNovaU1 BlockOffloader::load: {e}"))
            })?;

        // 3) Resident shared weights: filter every shard for SHARED_PREFIXES.
        let mut shared: HashMap<String, Tensor> = HashMap::new();
        for path in &shard_paths {
            let part = load_file_filtered(path, device, |key| {
                SHARED_PREFIXES.iter().any(|p| key.starts_with(p))
            })?;
            shared.extend(part);
        }

        // 4) Validate every expected shared key is present (per-layer keys
        //    are validated lazily on each await_block via the facilitator
        //    routing — missing keys would have been silently dropped, but
        //    expected_per_layer_keys is exhaustive so we'd notice quickly).
        let mut missing: Vec<String> = Vec::new();
        for expected in expected_shared_keys() {
            if !shared.contains_key(*expected) {
                missing.push(expected.to_string());
            }
        }
        if !missing.is_empty() {
            return Err(Error::InvalidInput(format!(
                "SenseNovaU1: {} missing shared weight key(s); first few: {:?}",
                missing.len(),
                &missing[..missing.len().min(8)]
            )));
        }

        log::info!(
            "[SenseNovaU1] loaded: {} resident shared tensors, {} blocks streaming via BlockOffloader",
            shared.len(),
            offloader.block_count()
        );

        Ok(Self {
            config,
            shared,
            offloader,
            device: device.clone(),
        })
    }

    /// Borrow the resident shared weights (e.g. for the embed_tokens lookup,
    /// fm_head, vision_model_mot_gen embedder, final norms, lm_head).
    pub fn shared(&self) -> &HashMap<String, Tensor> { &self.shared }

    /// `BlockOffloader::prepare_weights` pre-transposes 2D `.weight` tensors
    /// to `[Cin, Cout]` for its internal matmul fast path. `fused_linear3d_native`
    /// expects PyTorch `[Cout, Cin]`, so un-transpose 2D weights here. Same
    /// pattern Chroma + QwenImage use.
    fn untranspose_block_weights(
        raw: &Arc<HashMap<String, Tensor>>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::with_capacity(raw.len());
        for (k, v) in raw.iter() {
            if k.ends_with(".weight") && v.shape().dims().len() == 2 {
                out.insert(k.clone(), v.transpose()?);
            } else {
                out.insert(k.clone(), v.clone());
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Phase 3: forward_und — text prefix path that POPULATES the KV cache
    // -----------------------------------------------------------------------
    //
    // Reference: modeling_qwen3.py::Qwen3DecoderLayer.forward_und (line 869) and
    // ::Qwen3Attention.forward_und (line 422). For T2I, the text prefix never
    // mixes with image tokens — every layer runs purely through base weights.
    //
    // Inputs:
    //   token_ids       : &[i32]  (input_ids from tokenizer; first dim is sequence)
    //   indexes_t       : &Tensor [seq_len]  (positions along t-axis; modeling_neo_chat.py:444)
    // Outputs:
    //   KvCache for the prefix + final hidden state (last_hidden_state of `language_model.model`).
    //
    // Steps (per layer):
    //   x = embed_tokens[token_ids]
    //   for layer in 0..42:
    //       residual = x
    //       x = rms_norm(x, input_layernorm.weight)
    //       Q = q_proj(x); K = k_proj(x); V = v_proj(x)
    //       (Q_t, Q_h, Q_w) = split(Q.view(b, n, h, head_dim), 64/32/32)
    //       Q_t = head_rms_norm(Q_t, q_norm.weight, eps)              # weight=[64]
    //       Q_hw = head_rms_norm(Q_hw_full_64, q_norm_hw.weight, eps) # then chunk(Q_hw, 2) → Q_h, Q_w
    //       Q_t = apply_rope_halfsplit(Q_t, cos_t(idx_t), sin_t(idx_t))
    //       Q_h = apply_rope_halfsplit(Q_h, cos_h(0), sin_h(0))   # text rows = 0
    //       Q_w = apply_rope_halfsplit(Q_w, cos_w(0), sin_w(0))   # text cols = 0
    //       Q   = concat([Q_t, Q_h, Q_w], dim=-1)
    //       (same for K; V is NOT RoPE'd)
    //       cache.layers[layer] = (K, V)            # BEFORE GQA repeat
    //       K_g = repeat_kv(K, n_rep);  V_g = repeat_kv(V, n_rep)
    //       attn = sdpa(Q, K_g, V_g, mask=block_causal_from(indexes_t))
    //       x = residual + o_proj(attn.merge_heads())
    //       residual = x
    //       x = rms_norm(x, post_attention_layernorm.weight)
    //       x = down_proj( silu(gate_proj(x)) * up_proj(x) )
    //       x = residual + x
    //   x = rms_norm(x, language_model.model.norm.weight)   # final, BASE norm
    //   return (cache, x)
    pub fn forward_und(
        &mut self,
        token_ids: &[i32],
    ) -> Result<(KvCache, Tensor)> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(Error::InvalidInput(
                "forward_und: empty token_ids".into(),
            ));
        }

        // Split-borrow self so the offloader can be borrowed `&mut` while
        // shared/config/device stay `&`.
        let Self { config, shared, device, offloader } = self;
        let cfg = &*config;

        // ---- Embed tokens → [1, N, hidden] ----
        let embed_w = shared
            .get("language_model.model.embed_tokens.weight")
            .ok_or_else(|| Error::InvalidInput(
                "forward_und: missing embed_tokens.weight".into(),
            ))?;
        let ids = Tensor::from_vec(
            token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let mut hidden = embed_w.index_select0(&ids)?.unsqueeze(0)?;

        // ---- Build RoPE tables for the t-axis (h/w are identity for text). ----
        let (dim_t, _dim_h, _dim_w) = cfg.rope_dims();
        let (cos_t, sin_t) = build_rope_table_1d(seq_len, dim_t, cfg.rope_theta, device)?;

        // ---- Build causal mask: lower-triangular 0/1 BF16, [1,1,N,N]. ----
        let attn_mask = build_causal_mask(seq_len, seq_len, device)?;

        // ---- 42 Qwen3 layers, base path — streamed via offloader ----
        let total = cfg.num_layers;
        offloader.prefetch_block(0)
            .map_err(|e| Error::InvalidInput(format!("prefetch block 0: {e}")))?;
        let mut cache_layers: Vec<(Tensor, Tensor)> = Vec::with_capacity(total);
        for i in 0..total {
            let raw = offloader.await_block(i)
                .map_err(|e| Error::InvalidInput(format!("await block {i}: {e}")))?;
            if i + 1 < total {
                offloader.prefetch_block(i + 1)
                    .map_err(|e| Error::InvalidInput(format!("prefetch block {}: {e}", i + 1)))?;
            }
            let lw = Self::untranspose_block_weights(&raw)?;
            let (new_hidden, k_cache, v_cache) =
                Self::und_layer(cfg, i, &lw, &hidden, &cos_t, &sin_t, &attn_mask)?;
            cache_layers.push((k_cache, v_cache));
            hidden = new_hidden;
        }

        // ---- Final norm (BASE path: language_model.model.norm.weight) ----
        let final_norm = shared
            .get("language_model.model.norm.weight")
            .ok_or_else(|| Error::InvalidInput(
                "forward_und: missing language_model.model.norm.weight".into(),
            ))?;
        hidden = Self::rms_norm_apply(&hidden, final_norm, cfg.rms_norm_eps)?;

        Ok((
            KvCache {
                layers: cache_layers,
                prefix_len: seq_len,
                next_t_index: seq_len,
            },
            hidden,
        ))
    }

    // -----------------------------------------------------------------------
    // Per-layer base-path forward (used by forward_und).
    // -----------------------------------------------------------------------

    /// Returns `(new_hidden, k_cache, v_cache)` where the cache tensors are
    /// the per-layer K/V at shape `[B, num_kv_heads, N, head_dim]` (BEFORE
    /// the GQA repeat — matches what forward_gen will concat with later).
    fn und_layer(
        cfg: &SenseNovaU1Config,
        i: usize,
        lw: &HashMap<String, Tensor>,
        hidden: &Tensor,
        cos_t: &Tensor,
        sin_t: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let h_total = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h_total / h_kv;
        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        let lget = |k: &str| -> Result<&Tensor> {
            lw.get(k).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "SenseNovaU1: missing layer-{i} weight {k}"
                ))
            })
        };

        // ---- Self-attention ----
        let normed = Self::rms_norm_apply(
            hidden,
            lget(&format!(
                "language_model.model.layers.{i}.input_layernorm.weight"
            ))?,
            cfg.rms_norm_eps,
        )?;

        let q = Self::linear_no_bias(
            &normed,
            lget(&format!(
                "language_model.model.layers.{i}.self_attn.q_proj.weight"
            ))?,
        )?;
        let k = Self::linear_no_bias(
            &normed,
            lget(&format!(
                "language_model.model.layers.{i}.self_attn.k_proj.weight"
            ))?,
        )?;
        let v = Self::linear_no_bias(
            &normed,
            lget(&format!(
                "language_model.model.layers.{i}.self_attn.v_proj.weight"
            ))?,
        )?;

        // [B, N, H*D] → [B, H, N, D]
        let q = q.reshape(&[b, n, h_total, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // Per-head split-half RMSNorm: first 64 dims (t-axis) and last 64
        // dims (hw) get DIFFERENT learned norm weights (q_norm vs q_norm_hw).
        // After norms, only the t-axis half gets RoPE; the hw half is
        // identity-RoPE for text (positions = 0). We skip the hw RoPE entirely
        // — the layout `[Q_t_after_rope, Q_hw_after_norm]` is bit-equivalent
        // to applying the HW norms then a no-op RoPE.
        let q_norm = lget(&format!(
            "language_model.model.layers.{i}.self_attn.q_norm.weight"
        ))?;
        let q_norm_hw = lget(&format!(
            "language_model.model.layers.{i}.self_attn.q_norm_hw.weight"
        ))?;
        let k_norm = lget(&format!(
            "language_model.model.layers.{i}.self_attn.k_norm.weight"
        ))?;
        let k_norm_hw = lget(&format!(
            "language_model.model.layers.{i}.self_attn.k_norm_hw.weight"
        ))?;

        let (q_t, q_hw) = Self::chunk_last_half(&q)?; // each [B, H, N, 64]
        let q_t = Self::head_rms_norm(&q_t, q_norm, cfg.rms_norm_eps)?;
        let q_hw = Self::head_rms_norm(&q_hw, q_norm_hw, cfg.rms_norm_eps)?;
        let q_t = flame_core::bf16_ops::rope_halfsplit_bf16(&q_t, cos_t, sin_t)?;
        let q = Tensor::cat(&[&q_t, &q_hw], 3)?;

        let (k_t, k_hw) = Self::chunk_last_half(&k)?;
        let k_t = Self::head_rms_norm(&k_t, k_norm, cfg.rms_norm_eps)?;
        let k_hw = Self::head_rms_norm(&k_hw, k_norm_hw, cfg.rms_norm_eps)?;
        let k_t = flame_core::bf16_ops::rope_halfsplit_bf16(&k_t, cos_t, sin_t)?;
        let k = Tensor::cat(&[&k_t, &k_hw], 3)?;

        // V is NOT RoPE'd (matches modeling_qwen3.py:447).

        // Save (K, V) BEFORE GQA repeat — matches forward_gen's expectation
        // that cache K/V are at num_kv_heads.
        let k_cache = k.clone();
        let v_cache = v.clone();

        // GQA repeat for SDPA
        let k_g = Self::repeat_kv(&k, n_rep)?;
        let v_g = Self::repeat_kv(&v, n_rep)?;

        let attn = flame_core::attention::sdpa(&q, &k_g, &v_g, Some(attn_mask))?;
        // [B, H, N, D] → [B, N, H*D]
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h_total * d])?;

        let attn = Self::linear_no_bias(
            &attn,
            lget(&format!(
                "language_model.model.layers.{i}.self_attn.o_proj.weight"
            ))?,
        )?;
        let hidden = hidden.add(&attn)?;

        // ---- SwiGLU MLP ----
        let post_norm_w = lget(&format!(
            "language_model.model.layers.{i}.post_attention_layernorm.weight"
        ))?;
        let n2 = Self::rms_norm_apply(&hidden, post_norm_w, cfg.rms_norm_eps)?;
        let gate_w = lget(&format!(
            "language_model.model.layers.{i}.mlp.gate_proj.weight"
        ))?;
        let up_w = lget(&format!(
            "language_model.model.layers.{i}.mlp.up_proj.weight"
        ))?;
        let down_w = lget(&format!(
            "language_model.model.layers.{i}.mlp.down_proj.weight"
        ))?;
        let gate = Self::linear_no_bias(&n2, gate_w)?;
        let up = Self::linear_no_bias(&n2, up_w)?;
        let mlp = gate.silu()?.mul(&up)?;
        let mlp = Self::linear_no_bias(&mlp, down_w)?;
        let hidden = hidden.add(&mlp)?;

        Ok((hidden, k_cache, v_cache))
    }

    // -----------------------------------------------------------------------
    // Helpers (private; visible to forward_gen later)
    // -----------------------------------------------------------------------

    fn shared_get(&self, key: &str) -> Result<&Tensor> {
        self.shared.get(key).ok_or_else(|| {
            Error::InvalidInput(format!("SenseNovaU1: missing shared weight {key}"))
        })
    }

    /// `fused_linear3d_native(x, w, None)` — preserves the [..., last] shape
    /// while doing the matmul with weight in row-major `[out, in]` (cuBLASLt
    /// transposes inside the GEMM, so we don't pre-transpose).
    fn linear_no_bias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        flame_core::ops::fused_inference::fused_linear3d_native(x, weight, None)
    }

    /// Apply RMSNorm with weight: reshape to [batch, hidden], norm, reshape back.
    fn rms_norm_apply(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
        out.reshape(&dims)
    }

    /// Per-head RMSNorm on `[B, H, N, D]` with a `[D]` weight.
    fn head_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "head_rms_norm expects 4D input, got {dims:?}"
            )));
        }
        let last = *dims.last().unwrap();
        let prod: usize = dims[..3].iter().product();
        let flat = x.reshape(&[prod, last])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&flat, Some(weight), eps)?;
        out.reshape(&dims)
    }

    /// Split `[..., D]` into two `[..., D/2]` halves along the last dim.
    fn chunk_last_half(x: &Tensor) -> Result<(Tensor, Tensor)> {
        let dims = x.shape().dims().to_vec();
        let last = *dims.last().unwrap();
        if last % 2 != 0 {
            return Err(Error::InvalidInput(format!(
                "chunk_last_half: last dim must be even, got {last}"
            )));
        }
        let half = last / 2;
        // Reshape to [..., 2, half], split, reshape back.
        let mut new_dims = dims.clone();
        *new_dims.last_mut().unwrap() = 2;
        new_dims.push(half);
        let reshaped = x.reshape(&new_dims)?;

        // Take chunk 0 and chunk 1 along the second-to-last dim via narrow.
        let lo = reshaped.narrow(new_dims.len() - 2, 0, 1)?;
        let hi = reshaped.narrow(new_dims.len() - 2, 1, 1)?;
        // Squeeze the size-1 axis to recover [..., half] shape.
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = half;
        Ok((lo.reshape(&out_dims)?, hi.reshape(&out_dims)?))
    }

    /// Repeat KV heads to match Q head count for GQA. `[B, H_kv, N, D]` →
    /// `[B, H_kv*n_rep, N, D]`.
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims();
        let b = dims[0];
        let h_kv = dims[1];
        let n = dims[2];
        let d = dims[3];
        let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
        let stacked = Tensor::stack(&copies, 2)?;
        stacked.reshape(&[b, h_kv * n_rep, n, d])
    }

    // -----------------------------------------------------------------------
    // Phase 3: forward_gen — per-step image path that READS the KV cache
    // -----------------------------------------------------------------------
    //
    // Reference: modeling_qwen3.py::Qwen3DecoderLayer.forward_gen (line 908) and
    // ::Qwen3Attention.forward_gen (line 574).
    //
    // Inputs:
    //   image_embeds   : Tensor [B, L, hidden=4096]  (gen-token embeddings + timestep + noise_scale)
    //   indexes_image  : (idx_t, idx_h, idx_w)        all shape [L]; from `_build_t2i_image_indexes`
    //   cache          : &KvCache                     populated by forward_und (NEVER updated here)
    //   attn_mask      : Option<&Tensor>              None → full attention; Some → block-causal+pad
    // Outputs:
    //   Tensor [B, L, hidden=4096]  — hidden state of the gen tokens after final norm_mot_gen.
    //
    // Steps (per layer i):
    //   residual = x
    //   x = rms_norm(x, input_layernorm_mot_gen.weight)
    //   Q = q_proj_mot_gen(x); K_cur = k_proj_mot_gen(x); V_cur = v_proj_mot_gen(x)
    //   (Q_t, Q_h, Q_w) build like forward_und but using *_mot_gen norms
    //   Q_t = apply_rope on idx_t = text_len (constant for image tokens, see line 453)
    //   Q_h = apply_rope on idx_h = patch_row
    //   Q_w = apply_rope on idx_w = patch_col
    //   Q  = concat([Q_t, Q_h, Q_w], dim=-1)
    //   (same for K_cur)
    //   K = concat([cache.layers[i].K, K_cur], dim=2)     # along seq_len; NO update
    //   V = concat([cache.layers[i].V, V_cur], dim=2)
    //   K_g = repeat_kv(K, n_rep); V_g = repeat_kv(V, n_rep)
    //   attn = sdpa(Q, K_g, V_g, mask=attn_mask /*None=causal=False per ref line 696*/)
    //   x = residual + o_proj_mot_gen(attn.merge_heads())
    //   residual = x
    //   x = rms_norm(x, post_attention_layernorm_mot_gen.weight)
    //   x = down_proj_mg(silu(gate_proj_mg(x)) * up_proj_mg(x))
    //   x = residual + x
    //  Final: x = rms_norm(x, language_model.model.norm_mot_gen.weight)
    /// Per-step image forward.
    ///
    /// `image_embeds`: `[B, L, hidden=4096]` gen-token embeddings AFTER timestep
    /// + noise_scale embedding addition. `text_len`: number of tokens
    /// previously fed to `forward_und` (drives the t-axis RoPE). `grid_h`/
    /// `grid_w`: patch-token grid (rows × cols) — `L = grid_h * grid_w`.
    /// `cache`: populated by `forward_und` and never mutated here.
    /// `attn_mask`: usually `None` (full attention; gen tokens see prefix +
    /// each other bidirectionally — see modeling_qwen3.py:631-696).
    pub fn forward_gen(
        &mut self,
        image_embeds: &Tensor,
        text_len: usize,
        grid_h: usize,
        grid_w: usize,
        cache: &KvCache,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dims = image_embeds.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "forward_gen: image_embeds must be [B, L, hidden], got {dims:?}"
            )));
        }
        let Self { config, shared, device, offloader } = self;
        let cfg = &*config;
        let l = dims[1];
        if l != grid_h * grid_w {
            return Err(Error::InvalidInput(format!(
                "forward_gen: L={l} must equal grid_h*grid_w={}*{}={}",
                grid_h, grid_w, grid_h * grid_w
            )));
        }
        if cache.layers.len() != cfg.num_layers {
            return Err(Error::InvalidInput(format!(
                "forward_gen: cache has {} layers, expected {}",
                cache.layers.len(),
                cfg.num_layers
            )));
        }

        // ---- Build positional indices for gen tokens (CPU-side) ----
        let idx_t: Vec<i32> = vec![text_len as i32; l];
        let idx_h: Vec<i32> = (0..l).map(|i| (i / grid_w) as i32).collect();
        let idx_w: Vec<i32> = (0..l).map(|i| (i % grid_w) as i32).collect();

        // ---- Build RoPE tables for the 3 axes ----
        let (dim_t, dim_h, dim_w) = cfg.rope_dims();
        let (cos_t, sin_t) = build_rope_for_positions(&idx_t, dim_t, cfg.rope_theta, device)?;
        let (cos_h, sin_h) = build_rope_for_positions(&idx_h, dim_h, cfg.rope_theta_hw, device)?;
        let (cos_w, sin_w) = build_rope_for_positions(&idx_w, dim_w, cfg.rope_theta_hw, device)?;

        // ---- 42 layers, gen path — streamed via offloader ----
        let total = cfg.num_layers;
        offloader.prefetch_block(0)
            .map_err(|e| Error::InvalidInput(format!("prefetch block 0: {e}")))?;
        let mut hidden = image_embeds.clone();
        for i in 0..total {
            let raw = offloader.await_block(i)
                .map_err(|e| Error::InvalidInput(format!("await block {i}: {e}")))?;
            if i + 1 < total {
                offloader.prefetch_block(i + 1)
                    .map_err(|e| Error::InvalidInput(format!("prefetch block {}: {e}", i + 1)))?;
            }
            let lw = Self::untranspose_block_weights(&raw)?;
            hidden = Self::gen_layer(
                cfg, i, &lw, &hidden, &cos_t, &sin_t, &cos_h, &sin_h, &cos_w, &sin_w,
                cache, attn_mask,
            )?;
        }

        // ---- Final norm (GEN path: language_model.model.norm_mot_gen.weight) ----
        let final_norm = shared
            .get("language_model.model.norm_mot_gen.weight")
            .ok_or_else(|| Error::InvalidInput(
                "forward_gen: missing language_model.model.norm_mot_gen.weight".into(),
            ))?;
        Self::rms_norm_apply(&hidden, final_norm, cfg.rms_norm_eps)
    }

    /// Per-layer gen-path forward. Mirrors `und_layer` with three changes:
    /// 1. Uses `_mot_gen` weights everywhere.
    /// 2. Applies the FULL 3D RoPE (t at θ=5e6, h+w at θ=1e4 over patch grid).
    /// 3. K/V are concatenated with `cache.layers[i]` along seq dim BEFORE GQA
    ///    repeat (matching the python no-update path that just builds
    ///    `[past_k, k_cur]` along axis 2).
    #[allow(clippy::too_many_arguments)]
    fn gen_layer(
        cfg: &SenseNovaU1Config,
        i: usize,
        lw: &HashMap<String, Tensor>,
        hidden: &Tensor,
        cos_t: &Tensor,
        sin_t: &Tensor,
        cos_h: &Tensor,
        sin_h: &Tensor,
        cos_w: &Tensor,
        sin_w: &Tensor,
        cache: &KvCache,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let h_total = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h_total / h_kv;
        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let l = dims[1];

        let lget = |k: &str| -> Result<&Tensor> {
            lw.get(k).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "SenseNovaU1: missing layer-{i} weight {k}"
                ))
            })
        };

        // ---- Self-attention (gen path) ----
        let normed = Self::rms_norm_apply(
            hidden,
            lget(&format!(
                "language_model.model.layers.{i}.input_layernorm_mot_gen.weight"
            ))?,
            cfg.rms_norm_eps,
        )?;

        let q = Self::linear_no_bias(
            &normed,
            lget(&format!(
                "language_model.model.layers.{i}.self_attn.q_proj_mot_gen.weight"
            ))?,
        )?;
        let k = Self::linear_no_bias(
            &normed,
            lget(&format!(
                "language_model.model.layers.{i}.self_attn.k_proj_mot_gen.weight"
            ))?,
        )?;
        let v = Self::linear_no_bias(
            &normed,
            lget(&format!(
                "language_model.model.layers.{i}.self_attn.v_proj_mot_gen.weight"
            ))?,
        )?;

        let q = q.reshape(&[b, l, h_total, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, l, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, l, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // Full 3D RoPE: split t/hw, norm both halves, then split hw into h/w,
        // apply 3 separate RoPE tables, concat back. Per modeling_qwen3.py:593-621.
        let q_norm = lget(&format!(
            "language_model.model.layers.{i}.self_attn.q_norm_mot_gen.weight"
        ))?;
        let q_norm_hw = lget(&format!(
            "language_model.model.layers.{i}.self_attn.q_norm_hw_mot_gen.weight"
        ))?;
        let k_norm = lget(&format!(
            "language_model.model.layers.{i}.self_attn.k_norm_mot_gen.weight"
        ))?;
        let k_norm_hw = lget(&format!(
            "language_model.model.layers.{i}.self_attn.k_norm_hw_mot_gen.weight"
        ))?;

        let q = Self::apply_3d_rope(
            &q, q_norm, q_norm_hw, cfg.rms_norm_eps, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
        )?;
        let k = Self::apply_3d_rope(
            &k, k_norm, k_norm_hw, cfg.rms_norm_eps, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
        )?;

        // ---- Concat with cached K/V along seq dim, then GQA repeat ----
        let (past_k, past_v) = &cache.layers[i];
        let k_full = Tensor::cat(&[past_k, &k], 2)?; // [B, H_kv, prefix_len + L, D]
        let v_full = Tensor::cat(&[past_v, &v], 2)?;

        let k_g = Self::repeat_kv(&k_full, n_rep)?;
        let v_g = Self::repeat_kv(&v_full, n_rep)?;

        let attn = flame_core::attention::sdpa(&q, &k_g, &v_g, attn_mask)?;
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, l, h_total * d])?;

        let attn = Self::linear_no_bias(
            &attn,
            lget(&format!(
                "language_model.model.layers.{i}.self_attn.o_proj_mot_gen.weight"
            ))?,
        )?;
        let hidden = hidden.add(&attn)?;

        // ---- SwiGLU MLP (mlp_mot_gen) ----
        let post_norm_w = lget(&format!(
            "language_model.model.layers.{i}.post_attention_layernorm_mot_gen.weight"
        ))?;
        let n2 = Self::rms_norm_apply(&hidden, post_norm_w, cfg.rms_norm_eps)?;
        let gate_w = lget(&format!(
            "language_model.model.layers.{i}.mlp_mot_gen.gate_proj.weight"
        ))?;
        let up_w = lget(&format!(
            "language_model.model.layers.{i}.mlp_mot_gen.up_proj.weight"
        ))?;
        let down_w = lget(&format!(
            "language_model.model.layers.{i}.mlp_mot_gen.down_proj.weight"
        ))?;
        let gate = Self::linear_no_bias(&n2, gate_w)?;
        let up = Self::linear_no_bias(&n2, up_w)?;
        let mlp = gate.silu()?.mul(&up)?;
        let mlp = Self::linear_no_bias(&mlp, down_w)?;
        hidden.add(&mlp)
    }

    /// Apply the 3-axis RoPE-with-norms to a `[B, H, N, head_dim=128]` tensor.
    /// Splits `(t=64, h=32, w=32)`, RMSNorm-each-half (the `_hw` norm operates
    /// on the WHOLE 64-dim hw chunk before the h/w split — see Python:
    /// `q_hw = q_norm_hw(q_hw)` then `q_h, q_w = chunk(q_hw, 2)`), applies a
    /// distinct RoPE on each axis, concatenates back.
    #[allow(clippy::too_many_arguments)]
    fn apply_3d_rope(
        x: &Tensor,
        norm_t: &Tensor,
        norm_hw: &Tensor,
        eps: f32,
        cos_t: &Tensor,
        sin_t: &Tensor,
        cos_h: &Tensor,
        sin_h: &Tensor,
        cos_w: &Tensor,
        sin_w: &Tensor,
    ) -> Result<Tensor> {
        let (x_t, x_hw) = Self::chunk_last_half(x)?; // [B, H, N, 64] each
        let x_t = Self::head_rms_norm(&x_t, norm_t, eps)?;
        let x_hw = Self::head_rms_norm(&x_hw, norm_hw, eps)?;
        let (x_h, x_w) = Self::chunk_last_half(&x_hw)?; // [B, H, N, 32] each
        let x_t = flame_core::bf16_ops::rope_halfsplit_bf16(&x_t, cos_t, sin_t)?;
        let x_h = flame_core::bf16_ops::rope_halfsplit_bf16(&x_h, cos_h, sin_h)?;
        let x_w = flame_core::bf16_ops::rope_halfsplit_bf16(&x_w, cos_w, sin_w)?;
        Tensor::cat(&[&x_t, &x_h, &x_w], 3)
    }

    // -----------------------------------------------------------------------
    // Phase 3c: gen-side patch embedder
    // -----------------------------------------------------------------------
    //
    // Reference: modeling_neo_vit.py::NEOVisionEmbeddings.forward (line 160).
    //
    //   pixel_values arrives as [B*N, 3*16*16=768]   (already 16x16-patchified)
    //   reshape → [B*N, 3, 16, 16]
    //   patch_embedding: Conv2d(3, 1024, k=16, s=16) → [B*N, 1024, 1, 1] → squeeze → [B*N, 1024]
    //   GELU
    //   apply 2D RoPE on the [..., 1024] tensor using patch (x, y) coords:
    //       split [..., :512] uses RoPE on x-coord, [..., 512:] uses RoPE on y-coord,
    //       each with theta=10000, half-split, applied to even/odd interleave (see line 69-78)
    //   per-image: reshape [h, w, 1024] → permute [1024, h, w] → unsqueeze batch → [1, 1024, h, w]
    //   dense_embedding: Conv2d(1024, 4096, k=2, s=2) → [1, 4096, h/2, w/2]
    //   permute → [1, h/2, w/2, 4096] → flatten to [(h/2)*(w/2), 4096]
    //   concat across batch
    /// Gen-side patch + 2x2 spatial merge embedder.
    ///
    /// Reference: `modeling_neo_vit.py::NEOVisionEmbeddings.forward` (line 160).
    ///
    /// Inputs:
    ///   `pixel_values`: `[B*N, 768]` BF16 — already-patchified flat patches in
    ///                   `(C, kH, kW)` C-major order (the output of
    ///                   `patchify(img, patch=16, channel_first=True)`).
    ///   `grid_h`, `grid_w`: image patch-grid dimensions before merge.
    ///
    /// Output: `[B, token_h*token_w, 4096]` BF16 where `token_h = grid_h/2`,
    /// `token_w = grid_w/2`.
    ///
    /// Pipeline:
    ///   1. Conv2d k=s=16 collapsed to matmul: `pixel_values @ Wᵀ + b`
    ///      with `W` reshaped from `[1024, 3, 16, 16]` → `[1024, 768]`.
    ///   2. GELU.
    ///   3. 2D **interleaved** RoPE on `[..., 1024]`: first 512 dims rotated by
    ///      patch x-coord (θ=10000), second 512 dims by y-coord. Both axes
    ///      share the same θ. Uses `flame_core::bf16_ops::rope_fused_bf16`.
    ///   4. Conv2d k=s=2 (dense_embedding) collapsed to matmul. Spatial pack
    ///      via `[B, gh, gw, 1024] → [B, gh/2, 2, gw/2, 2, 1024] →
    ///      permute(0,1,3,5,2,4) → [B*tH*tW, 4096]`. Weight reshape from
    ///      `[4096, 1024, 2, 2]` → `[4096, 4096]` matches the `(Cin, kH, kW)`
    ///      C-major flatten of the input.
    ///
    /// Numerical note: the Python does RoPE in F32 then casts back; we do it
    /// in BF16 (the flame-core interleaved RoPE kernel is BF16-only). For
    /// max position ≤ ~64 the difference is negligible for inference.
    pub fn extract_feature_gen(
        &self,
        pixel_values: &Tensor,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        let dims = pixel_values.shape().dims();
        if dims.len() != 2 || dims[1] != 3 * self.config.patch_size * self.config.patch_size {
            return Err(Error::InvalidInput(format!(
                "extract_feature_gen: expected [B*N, {}], got {:?}",
                3 * self.config.patch_size * self.config.patch_size,
                dims
            )));
        }
        let n = grid_h * grid_w;
        let bn = dims[0];
        if n == 0 || bn % n != 0 {
            return Err(Error::InvalidInput(format!(
                "extract_feature_gen: B*N={bn} not divisible by grid_h*grid_w={n}"
            )));
        }
        let b = bn / n;
        let merge = self.config.merge_size();
        if grid_h % merge != 0 || grid_w % merge != 0 {
            return Err(Error::InvalidInput(format!(
                "extract_feature_gen: grid {grid_h}x{grid_w} must be divisible by merge_size {merge}"
            )));
        }
        let token_h = grid_h / merge;
        let token_w = grid_w / merge;

        let pe_w = self.shared_get(
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        )?;
        let pe_b = self.shared_get(
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias",
        )?;
        let de_w = self.shared_get(
            "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight",
        )?;
        let de_b = self.shared_get(
            "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias",
        )?;

        // (1) Conv2d-as-matmul patch embedding. pe_w is [1024, 3, 16, 16];
        //     reshape to [1024, 768] preserves C-major (Cin, kH, kW) order.
        //     fused_linear3d_native requires 3D input — reshape [B*N, 768] →
        //     [1, B*N, 768] for the call, then squeeze back.
        let patch_flat = 3 * self.config.patch_size * self.config.patch_size;
        let pe_w_flat = pe_w.reshape(&[self.config.vision_hidden_size, patch_flat])?;
        let pixel_3d = pixel_values.reshape(&[1, bn, patch_flat])?;
        let h = flame_core::ops::fused_inference::fused_linear3d_native(
            &pixel_3d,
            &pe_w_flat,
            Some(pe_b),
        )?
        .reshape(&[bn, self.config.vision_hidden_size])?; // [B*N, 1024]

        // (2) GELU.
        let h = h.gelu()?;

        // (3) 2D interleaved RoPE on [..., vision_hidden]:
        //     first half rotated by patch x-coord, second half by y-coord.
        //     Build positions for B*N tokens (tile per-image pattern across batch).
        let half = self.config.vision_hidden_size / 2; // 512
        let theta = self.config.rope_theta_vision;
        let mut pos_x: Vec<i32> = Vec::with_capacity(bn);
        let mut pos_y: Vec<i32> = Vec::with_capacity(bn);
        for _ in 0..b {
            for i in 0..n {
                pos_x.push((i % grid_w) as i32);
                pos_y.push((i / grid_w) as i32);
            }
        }
        let (cos_x, sin_x) = build_rope_for_positions(&pos_x, half, theta, &self.device)?;
        let (cos_y, sin_y) = build_rope_for_positions(&pos_y, half, theta, &self.device)?;

        // Split [B*N, 1024] into two [B*N, 512].
        let (h_x, h_y) = Self::chunk_last_half(&h)?;
        // Reshape to [B=1, H=1, N=B*N, D=half] for the kernel's [B, H, N, D] convention.
        let h_x = h_x.reshape(&[1, 1, bn, half])?;
        let h_y = h_y.reshape(&[1, 1, bn, half])?;
        let h_x = flame_core::bf16_ops::rope_fused_bf16(&h_x, &cos_x, &sin_x)?;
        let h_y = flame_core::bf16_ops::rope_fused_bf16(&h_y, &cos_y, &sin_y)?;
        let h_x = h_x.reshape(&[bn, half])?;
        let h_y = h_y.reshape(&[bn, half])?;
        let h = Tensor::cat(&[&h_x, &h_y], 1)?; // [B*N, 1024]

        // (4) 2x2 spatial merge via dense_embedding (Conv2d k=s=2 as matmul).
        //     Pack so the inner-most axis is (Cin, kH, kW) C-major to match the
        //     reshape of dense_embedding.weight.
        let h = h.reshape(&[b, grid_h, grid_w, self.config.vision_hidden_size])?;
        let h = h.reshape(&[b, token_h, merge, token_w, merge, self.config.vision_hidden_size])?;
        // Source axes 0=B 1=token_h 2=kH 3=token_w 4=kW 5=Cin
        // Target order: B token_h token_w Cin kH kW  →  permute [0, 1, 3, 5, 2, 4]
        let h = h.permute(&[0, 1, 3, 5, 2, 4])?;
        let merge_flat = self.config.vision_hidden_size * merge * merge;
        let h = h.reshape(&[1, b * token_h * token_w, merge_flat])?;
        let de_w_flat = de_w.reshape(&[self.config.hidden_size, merge_flat])?;
        let h = flame_core::ops::fused_inference::fused_linear3d_native(
            &h,
            &de_w_flat,
            Some(de_b),
        )?; // [1, B*token_h*token_w, hidden_size]

        // (5) Reshape to [B, L, hidden_size]
        h.reshape(&[b, token_h * token_w, self.config.hidden_size])
    }

    // -----------------------------------------------------------------------
    // Phase 2 (cap. 1/2): understanding-side image feature extractor.
    //
    // Mirror of `extract_feature_gen` using the BASE vision-embedder weights
    // at `vision_model.embeddings.*` (no `_mot_gen` suffix). Same structure:
    // Conv2d-as-matmul patch embedder, GELU, interleaved 2D RoPE, then
    // Conv2d-as-matmul 2×2 spatial merge into hidden_size=4096.
    //
    // Reference: `modeling_neo_vit.py::NEOVisionEmbeddings.forward` for the
    // patch+rope+merge math, and `modeling_neo_chat.py::extract_feature`
    // (line 304) which switches between the two (`gen_model=True` vs False).
    // -----------------------------------------------------------------------

    /// Extract understanding-side image features. Inputs and shapes match
    /// `extract_feature_gen` — only the loaded weight prefix differs.
    pub fn extract_feature_und(
        &self,
        pixel_values: &Tensor,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        let dims = pixel_values.shape().dims();
        if dims.len() != 2 || dims[1] != 3 * self.config.patch_size * self.config.patch_size {
            return Err(Error::InvalidInput(format!(
                "extract_feature_und: expected [B*N, {}], got {:?}",
                3 * self.config.patch_size * self.config.patch_size,
                dims
            )));
        }
        let n = grid_h * grid_w;
        let bn = dims[0];
        if n == 0 || bn % n != 0 {
            return Err(Error::InvalidInput(format!(
                "extract_feature_und: B*N={bn} not divisible by grid_h*grid_w={n}"
            )));
        }
        let b = bn / n;
        let merge = self.config.merge_size();
        if grid_h % merge != 0 || grid_w % merge != 0 {
            return Err(Error::InvalidInput(format!(
                "extract_feature_und: grid {grid_h}x{grid_w} must be divisible by merge_size {merge}"
            )));
        }
        let token_h = grid_h / merge;
        let token_w = grid_w / merge;

        let pe_w = self.shared_get("vision_model.embeddings.patch_embedding.weight")?;
        let pe_b = self.shared_get("vision_model.embeddings.patch_embedding.bias")?;
        let de_w = self.shared_get("vision_model.embeddings.dense_embedding.weight")?;
        let de_b = self.shared_get("vision_model.embeddings.dense_embedding.bias")?;

        let patch_flat = 3 * self.config.patch_size * self.config.patch_size;
        let pe_w_flat = pe_w.reshape(&[self.config.vision_hidden_size, patch_flat])?;
        let pixel_3d = pixel_values.reshape(&[1, bn, patch_flat])?;
        let h = flame_core::ops::fused_inference::fused_linear3d_native(
            &pixel_3d,
            &pe_w_flat,
            Some(pe_b),
        )?
        .reshape(&[bn, self.config.vision_hidden_size])?;

        let h = h.gelu()?;

        let half = self.config.vision_hidden_size / 2;
        let theta = self.config.rope_theta_vision;
        let mut pos_x: Vec<i32> = Vec::with_capacity(bn);
        let mut pos_y: Vec<i32> = Vec::with_capacity(bn);
        for _ in 0..b {
            for i in 0..n {
                pos_x.push((i % grid_w) as i32);
                pos_y.push((i / grid_w) as i32);
            }
        }
        let (cos_x, sin_x) = build_rope_for_positions(&pos_x, half, theta, &self.device)?;
        let (cos_y, sin_y) = build_rope_for_positions(&pos_y, half, theta, &self.device)?;
        let (h_x, h_y) = Self::chunk_last_half(&h)?;
        let h_x = h_x.reshape(&[1, 1, bn, half])?;
        let h_y = h_y.reshape(&[1, 1, bn, half])?;
        let h_x = flame_core::bf16_ops::rope_fused_bf16(&h_x, &cos_x, &sin_x)?;
        let h_y = flame_core::bf16_ops::rope_fused_bf16(&h_y, &cos_y, &sin_y)?;
        let h_x = h_x.reshape(&[bn, half])?;
        let h_y = h_y.reshape(&[bn, half])?;
        let h = Tensor::cat(&[&h_x, &h_y], 1)?;

        let h = h.reshape(&[b, grid_h, grid_w, self.config.vision_hidden_size])?;
        let h = h.reshape(&[b, token_h, merge, token_w, merge, self.config.vision_hidden_size])?;
        let h = h.permute(&[0, 1, 3, 5, 2, 4])?;
        let merge_flat = self.config.vision_hidden_size * merge * merge;
        let h = h.reshape(&[1, b * token_h * token_w, merge_flat])?;
        let de_w_flat = de_w.reshape(&[self.config.hidden_size, merge_flat])?;
        let h = flame_core::ops::fused_inference::fused_linear3d_native(
            &h,
            &de_w_flat,
            Some(de_b),
        )?;
        h.reshape(&[b, token_h * token_w, self.config.hidden_size])
    }

    // -----------------------------------------------------------------------
    // Phase 4: fm_modules
    // -----------------------------------------------------------------------
    //
    // Reference: modeling_fm_modules.py::TimestepEmbedder (line 23) and the
    // 2-layer fm_head MLP keyed at `fm_modules.fm_head.{0,2}.{weight,bias}`.

    /// Sinusoidal frequency embedding (256 dims) → `Linear(256, 4096)` →
    /// SiLU → `Linear(4096, 4096)`. Shared shape between `timestep_embedder`
    /// and `noise_scale_embedder` (different weights, same architecture).
    ///
    /// Reference: `modeling_fm_modules.py::TimestepEmbedder` (line 23). The
    /// sinusoidal layout is `cat([cos(args), sin(args)], dim=-1)` per the
    /// reference implementation (line 52) — note `cos` first, `sin` second.
    ///
    /// `t`: 1-D Tensor `[N]` of fractional timesteps (or noise scales).
    /// Returns `[N, 4096]` BF16.
    pub fn time_or_scale_embed(&self, t: &Tensor, which: TimeOrScale) -> Result<Tensor> {
        let prefix = match which {
            TimeOrScale::Timestep => "fm_modules.timestep_embedder",
            TimeOrScale::NoiseScale => "fm_modules.noise_scale_embedder",
        };
        let w0 = self.shared_get(&format!("{prefix}.mlp.0.weight"))?;
        let b0 = self.shared_get(&format!("{prefix}.mlp.0.bias"))?;
        let w2 = self.shared_get(&format!("{prefix}.mlp.2.weight"))?;
        let b2 = self.shared_get(&format!("{prefix}.mlp.2.bias"))?;

        // Build sinusoidal frequency embedding [N, 256] in BF16.
        let freq_embed = sinusoidal_freq_embed(t, 256, 10_000.0, &self.device)?;
        let n = freq_embed.shape().dims()[0];

        // 2-layer MLP with SiLU. fused_linear3d_native requires 3D input —
        // reshape [N, 256] → [1, N, 256], compute, return as [N, 4096] for
        // callers that re-reshape to [B, L, 4096].
        let f3d = freq_embed.reshape(&[1, n, 256])?;
        let h0 = flame_core::ops::fused_inference::fused_linear3d_native(&f3d, w0, Some(b0))?;
        let h0 = h0.silu()?;
        let h2 = flame_core::ops::fused_inference::fused_linear3d_native(&h0, w2, Some(b2))?;
        // h2 shape: [1, N, hidden]. Squeeze to [N, hidden].
        let hidden = h2.shape().dims()[2];
        h2.reshape(&[n, hidden])
    }

    /// fm_head: 2-layer MLP from 4096 → 4096 → fm_head_out_dim (3072 default).
    ///
    /// **Note:** `fm_head_dim` from config (1536) is IGNORED when
    /// `use_deep_fm_head=False` and `use_pixel_head=False` — see
    /// `modeling_neo_chat.py:183-187`:
    /// ```python
    /// fm_head = nn.Sequential(
    ///     nn.Linear(llm_hidden_size, 4096, bias=True),  # mlp.0
    ///     nn.GELU(),                                    # no params
    ///     nn.Linear(4096, output_dim, bias=True),       # mlp.2
    /// )
    /// ```
    /// The middle dim is hard-coded to 4096, not `config.fm_head_dim`.
    /// Activation is **GELU**, not SiLU.
    pub fn fm_head_forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let w0 = self.shared_get("fm_modules.fm_head.0.weight")?;
        let b0 = self.shared_get("fm_modules.fm_head.0.bias")?;
        let w2 = self.shared_get("fm_modules.fm_head.2.weight")?;
        let b2 = self.shared_get("fm_modules.fm_head.2.bias")?;
        let h0 = flame_core::ops::fused_inference::fused_linear3d_native(hidden, w0, Some(b0))?;
        let h0 = h0.gelu()?;
        flame_core::ops::fused_inference::fused_linear3d_native(&h0, w2, Some(b2))
    }

    // -----------------------------------------------------------------------
    // Phase 4: ODE sampler helpers
    // -----------------------------------------------------------------------

    /// Apply the time-shift schedule to a uniform `[0, 1]` grid (CPU-side).
    ///
    /// Reference: `_apply_time_schedule` modeling_neo_chat.py:409. We work in
    /// `f32` host space because the timestep grid has at most ~50 entries —
    /// not worth a CUDA kernel.
    ///
    ///   sigma = 1 - t
    ///   if time_schedule == "standard":
    ///       sigma = shift * sigma / (1 + (shift - 1) * sigma)    # shift = timestep_shift
    ///   elif time_schedule == "dynamic" + time_shift_type == "exponential":
    ///       mu    = base_shift + (max_shift - base_shift) / (max_seq - base_seq) * (seq - base_seq)
    ///       shift = exp(mu)
    ///       sigma = shift * sigma / (1 + (shift - 1) * sigma)
    ///   t = 1 - sigma
    ///
    /// **Note:** the python sets `self.time_schedule = "standard"` at the top
    /// of `_apply_time_schedule` whenever `timestep_shift != 1` — overriding
    /// the config. We follow the same precedence: if `timestep_shift != 1.0`,
    /// use Standard regardless of `cfg.time_schedule`.
    pub fn apply_time_schedule(
        &self,
        t_uniform: &[f32],
        image_seq_len: usize,
        timestep_shift: f32,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let mut out = Vec::with_capacity(t_uniform.len());
        let use_standard = timestep_shift != 1.0 || cfg.time_schedule == TimeSchedule::Standard;
        for &t in t_uniform {
            let sigma = 1.0 - t;
            let shifted = if use_standard {
                let shift = timestep_shift;
                shift * sigma / (1.0 + (shift - 1.0) * sigma)
            } else {
                // Dynamic schedule
                let denom = cfg.max_image_seq_len as f32 - cfg.base_image_seq_len as f32;
                let mu = if denom == 0.0 {
                    cfg.base_shift
                } else {
                    let m = (cfg.max_shift - cfg.base_shift) / denom;
                    let b = cfg.base_shift - m * cfg.base_image_seq_len as f32;
                    image_seq_len as f32 * m + b
                };
                match cfg.time_shift_type {
                    TimeShiftType::Exponential => {
                        let shift = mu.exp();
                        shift * sigma / (1.0 + (shift - 1.0) * sigma)
                    }
                    TimeShiftType::Linear => mu / (mu + (1.0 / sigma - 1.0)),
                }
            };
            out.push(1.0 - shifted);
        }
        out
    }

    /// Resolution-aware noise scale.
    ///
    /// Reference: t2i_generate body, modeling_neo_chat.py:1656-1663:
    ///   base = noise_scale_base_image_seq_len   # 64
    ///   scale = sqrt((grid_h*grid_w) / merge_size^2 / base)
    ///   noise_scale = scale * config.noise_scale
    ///   noise_scale = min(noise_scale, noise_scale_max_value)
    pub fn compute_noise_scale(&self, grid_h: usize, grid_w: usize) -> f32 {
        let merge = self.config.merge_size() as f32;
        let base = self.config.noise_scale_base_image_seq_len as f32;
        let n_tokens = (grid_h * grid_w) as f32;
        let raw = ((n_tokens / (merge * merge) / base).sqrt()) * self.config.noise_scale;
        raw.min(self.config.noise_scale_max_value)
    }

    // -----------------------------------------------------------------------
    // Phase 5 (cap. 1/3/4): autoregressive decode primitives.
    // -----------------------------------------------------------------------
    //
    // Reference Python: `_generate_think` and `_append_text_tokens_to_cache`
    // in `modeling_neo_chat.py:507`/`:478`. The shared kernel is single-token
    // self-attention with past-K/V concat — we reuse the existing `und_layer`
    // shape contract (BF16, base path) and just append along the seq dim.

    /// Single-token (or short batch) base-mode forward through one decoder
    /// layer with past-K/V concat. Returns `(new_hidden, k_full, v_full)`
    /// where the K/V are at `num_kv_heads` (BEFORE GQA repeat) and span
    /// `past_len + cur_len` along the seq dim — i.e., the new cache entry.
    ///
    /// Mirrors `und_layer` (see line ~621) but takes a `past_k`/`past_v` to
    /// concatenate before SDPA, the same pattern `gen_layer` uses for the
    /// gen path.
    #[allow(clippy::too_many_arguments)]
    fn und_layer_step(
        cfg: &SenseNovaU1Config,
        i: usize,
        lw: &HashMap<String, Tensor>,
        hidden: &Tensor,
        cos_t: &Tensor,
        sin_t: &Tensor,
        past_k: &Tensor,
        past_v: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let h_total = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h_total / h_kv;
        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        let lget = |k: &str| -> Result<&Tensor> {
            lw.get(k).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "SenseNovaU1: missing layer-{i} weight {k}"
                ))
            })
        };

        let normed = Self::rms_norm_apply(
            hidden,
            lget(&format!("language_model.model.layers.{i}.input_layernorm.weight"))?,
            cfg.rms_norm_eps,
        )?;

        let q = Self::linear_no_bias(
            &normed,
            lget(&format!("language_model.model.layers.{i}.self_attn.q_proj.weight"))?,
        )?;
        let k = Self::linear_no_bias(
            &normed,
            lget(&format!("language_model.model.layers.{i}.self_attn.k_proj.weight"))?,
        )?;
        let v = Self::linear_no_bias(
            &normed,
            lget(&format!("language_model.model.layers.{i}.self_attn.v_proj.weight"))?,
        )?;

        let q = q.reshape(&[b, n, h_total, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        let q_norm = lget(&format!("language_model.model.layers.{i}.self_attn.q_norm.weight"))?;
        let q_norm_hw = lget(&format!("language_model.model.layers.{i}.self_attn.q_norm_hw.weight"))?;
        let k_norm = lget(&format!("language_model.model.layers.{i}.self_attn.k_norm.weight"))?;
        let k_norm_hw = lget(&format!("language_model.model.layers.{i}.self_attn.k_norm_hw.weight"))?;

        // Decoded text tokens carry h_idx = w_idx = 0 (identity RoPE) so we
        // skip the h/w rotations entirely. Same shortcut `und_layer` takes.
        let (q_t, q_hw) = Self::chunk_last_half(&q)?;
        let q_t = Self::head_rms_norm(&q_t, q_norm, cfg.rms_norm_eps)?;
        let q_hw = Self::head_rms_norm(&q_hw, q_norm_hw, cfg.rms_norm_eps)?;
        let q_t = flame_core::bf16_ops::rope_halfsplit_bf16(&q_t, cos_t, sin_t)?;
        let q = Tensor::cat(&[&q_t, &q_hw], 3)?;

        let (k_t, k_hw) = Self::chunk_last_half(&k)?;
        let k_t = Self::head_rms_norm(&k_t, k_norm, cfg.rms_norm_eps)?;
        let k_hw = Self::head_rms_norm(&k_hw, k_norm_hw, cfg.rms_norm_eps)?;
        let k_t = flame_core::bf16_ops::rope_halfsplit_bf16(&k_t, cos_t, sin_t)?;
        let k = Tensor::cat(&[&k_t, &k_hw], 3)?;

        // Concat with past — full attention, no mask needed (single new token
        // attending to everything is the autoregressive contract).
        let k_full = Tensor::cat(&[past_k, &k], 2)?;
        let v_full = Tensor::cat(&[past_v, &v], 2)?;

        let k_g = Self::repeat_kv(&k_full, n_rep)?;
        let v_g = Self::repeat_kv(&v_full, n_rep)?;
        let attn = flame_core::attention::sdpa(&q, &k_g, &v_g, None)?;
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h_total * d])?;
        let attn = Self::linear_no_bias(
            &attn,
            lget(&format!("language_model.model.layers.{i}.self_attn.o_proj.weight"))?,
        )?;
        let hidden = hidden.add(&attn)?;

        let post_norm_w = lget(&format!(
            "language_model.model.layers.{i}.post_attention_layernorm.weight"
        ))?;
        let n2 = Self::rms_norm_apply(&hidden, post_norm_w, cfg.rms_norm_eps)?;
        let gate_w = lget(&format!("language_model.model.layers.{i}.mlp.gate_proj.weight"))?;
        let up_w = lget(&format!("language_model.model.layers.{i}.mlp.up_proj.weight"))?;
        let down_w = lget(&format!("language_model.model.layers.{i}.mlp.down_proj.weight"))?;
        let gate = Self::linear_no_bias(&n2, gate_w)?;
        let up = Self::linear_no_bias(&n2, up_w)?;
        let mlp = gate.silu()?.mul(&up)?;
        let mlp = Self::linear_no_bias(&mlp, down_w)?;
        let hidden = hidden.add(&mlp)?;

        Ok((hidden, k_full, v_full))
    }

    /// Apply lm_head to a `[1, N, hidden]` tensor and return logits
    /// `[1, N, vocab_size]`. The lm_head weight in `shared` is in PyTorch
    /// `[vocab, hidden]` layout (`load_file_filtered` does not transpose).
    fn lm_head_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let w = self.shared_get("language_model.lm_head.weight")?;
        flame_core::ops::fused_inference::fused_linear3d_native(hidden, w, None)
    }

    /// Greedy argmax over the last dim of a `[1, 1, vocab]` BF16 logit tensor,
    /// returning the chosen token id as `i32`. Done host-side because flame-core
    /// has no argmax kernel and we only do one ~600 KB transfer per token.
    fn argmax_last_token(logits: &Tensor) -> Result<i32> {
        let dims = logits.shape().dims();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(Error::InvalidInput(format!(
                "argmax_last_token: expected [1, N, vocab], got {dims:?}"
            )));
        }
        let n = dims[1];
        let v = dims[2];
        // Take only the LAST row [1, 1, vocab] to minimize the dtoh copy.
        let last = logits.narrow(1, n - 1, 1)?;
        let host = last.to_vec_f32()?; // BF16 → F32 on device, then dtoh
        if host.len() != v {
            return Err(Error::InvalidInput(format!(
                "argmax_last_token: expected {v} elements, got {}",
                host.len()
            )));
        }
        let mut best_i: usize = 0;
        let mut best_x: f32 = host[0];
        for (i, &x) in host.iter().enumerate().skip(1) {
            if x > best_x {
                best_x = x;
                best_i = i;
            }
        }
        Ok(best_i as i32)
    }

    /// Embed a single token id into `[1, 1, hidden]` via `embed_tokens`.
    fn embed_one(&self, token_id: i32) -> Result<Tensor> {
        let embed_w = self.shared_get("language_model.model.embed_tokens.weight")?;
        let ids = Tensor::from_vec(
            vec![token_id as f32],
            Shape::from_dims(&[1]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        embed_w.index_select0(&ids)?.unsqueeze(0)
    }

    /// Greedy autoregressive decode after a base-mode prefix forward.
    ///
    /// `cache` is mutated in place — each layer's K/V is replaced with the
    /// post-decode (concat'd) tensor, and `cache.prefix_len` /
    /// `cache.next_t_index` advance by `out.len()`.
    ///
    /// `last_hidden` is the FINAL-NORMED hidden returned by the prefix forward
    /// (`forward_und` or `forward_mixed_prefix`). Only its last row is used.
    /// Returns the generated token ids in order. EOS tokens ARE included in
    /// the output so callers can record the literal stop token if desired.
    pub fn decode_autoregressive(
        &mut self,
        cache: &mut KvCache,
        last_hidden: &Tensor,
        max_new_tokens: usize,
        eos_token_ids: &[i32],
    ) -> Result<Vec<i32>> {
        if cache.layers.len() != self.config.num_layers {
            return Err(Error::InvalidInput(format!(
                "decode_autoregressive: cache has {} layers, expected {}",
                cache.layers.len(),
                self.config.num_layers
            )));
        }
        let dims = last_hidden.shape().dims().to_vec();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(Error::InvalidInput(format!(
                "decode_autoregressive: last_hidden must be [1, N, hidden], got {dims:?}"
            )));
        }
        let prefix_n = dims[1];
        // Take only the last position; it's already final-normed.
        let mut next_normed = last_hidden.narrow(1, prefix_n - 1, 1)?;

        let cfg_clone = self.config.clone();
        let cfg = &cfg_clone;
        let total = cfg.num_layers;
        let (dim_t, _dim_h, _dim_w) = cfg.rope_dims();
        let final_norm_w = self
            .shared
            .get("language_model.model.norm.weight")
            .ok_or_else(|| {
                Error::InvalidInput(
                    "decode_autoregressive: missing language_model.model.norm.weight".into(),
                )
            })?
            .clone();

        let mut out: Vec<i32> = Vec::with_capacity(max_new_tokens);
        for _step in 0..max_new_tokens {
            // 1) lm_head + greedy sample.
            let logits = self.lm_head_logits(&next_normed)?;
            let next_id = Self::argmax_last_token(&logits)?;
            out.push(next_id);
            if eos_token_ids.contains(&next_id) {
                break;
            }

            // 2) Embed and run through 42 layers; append K/V to cache.
            let mut hidden = self.embed_one(next_id)?; // [1, 1, hidden]
            // RoPE for the new token's t-position.
            let pos = cache.next_t_index as i32;
            let (cos_t, sin_t) = build_rope_for_positions(
                &[pos],
                dim_t,
                cfg.rope_theta,
                &self.device,
            )?;

            // Step the offloader through all 42 base-path blocks.
            self.offloader
                .prefetch_block(0)
                .map_err(|e| Error::InvalidInput(format!("prefetch block 0: {e}")))?;
            for i in 0..total {
                let raw = self
                    .offloader
                    .await_block(i)
                    .map_err(|e| Error::InvalidInput(format!("await block {i}: {e}")))?;
                if i + 1 < total {
                    self.offloader
                        .prefetch_block(i + 1)
                        .map_err(|e| Error::InvalidInput(format!("prefetch block {}: {e}", i + 1)))?;
                }
                let lw = Self::untranspose_block_weights(&raw)?;
                let (past_k, past_v) = &cache.layers[i];
                let (new_h, k_full, v_full) = Self::und_layer_step(
                    cfg, i, &lw, &hidden, &cos_t, &sin_t, past_k, past_v,
                )?;
                cache.layers[i] = (k_full, v_full);
                hidden = new_h;
            }

            // 3) Final norm → ready for next iteration's lm_head.
            next_normed = Self::rms_norm_apply(&hidden, &final_norm_w, cfg.rms_norm_eps)?;

            cache.prefix_len += 1;
            cache.next_t_index += 1;
        }
        Ok(out)
    }

    /// Force-extend `cache` with the literal sequence of `token_ids`. Used by
    /// think mode after `</think>` to push `\n\n<img>` into the cache without
    /// sampling, mirroring `_append_text_tokens_to_cache`. Returns the
    /// FINAL-NORMED last hidden after the appended sequence — useful when the
    /// caller wants to chain into another decode pass or into `forward_gen`.
    pub fn extend_cache_with_text_tokens(
        &mut self,
        cache: &mut KvCache,
        token_ids: &[i32],
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::InvalidInput(
                "extend_cache_with_text_tokens: empty token_ids".into(),
            ));
        }
        if cache.layers.len() != self.config.num_layers {
            return Err(Error::InvalidInput(format!(
                "extend_cache_with_text_tokens: cache has {} layers, expected {}",
                cache.layers.len(),
                self.config.num_layers
            )));
        }
        let cfg_clone = self.config.clone();
        let cfg = &cfg_clone;
        let total = cfg.num_layers;
        let (dim_t, _dim_h, _dim_w) = cfg.rope_dims();
        let final_norm_w = self
            .shared
            .get("language_model.model.norm.weight")
            .ok_or_else(|| {
                Error::InvalidInput(
                    "extend_cache_with_text_tokens: missing language_model.model.norm.weight".into(),
                )
            })?
            .clone();

        // Embed the run as a single [1, N, hidden] block (cheaper than N
        // separate single-token forwards). t-axis indexes are
        // arange(next_t_index, next_t_index + N).
        let n = token_ids.len();
        let embed_w = self.shared_get("language_model.model.embed_tokens.weight")?;
        let ids = Tensor::from_vec(
            token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[n]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let mut hidden = embed_w.index_select0(&ids)?.unsqueeze(0)?; // [1, N, h]

        let positions: Vec<i32> = (0..n)
            .map(|i| (cache.next_t_index + i) as i32)
            .collect();
        let (cos_t, sin_t) = build_rope_for_positions(
            &positions,
            dim_t,
            cfg.rope_theta,
            &self.device,
        )?;

        // Block-causal mask for the appended chunk attending to itself + the
        // entire past. The Python reference (`_append_text_tokens_to_cache`)
        // builds `[1, 1, n, past_len + n]` with the right half being a strict
        // lower-triangular causal block and the left half all-attend. Our SDPA
        // takes the [1, 1, q_len, kv_len] mask as 0/1 keep-mask BF16.
        let past_len = cache.prefix_len;
        let total_kv = past_len + n;
        let mut mask_host: Vec<f32> = vec![1.0; n * total_kv];
        for q in 0..n {
            for k in past_len..total_kv {
                let kpos = k - past_len; // 0..n
                if kpos > q {
                    mask_host[q * total_kv + k] = 0.0;
                }
            }
        }
        let attn_mask = Tensor::from_vec(
            mask_host,
            Shape::from_dims(&[1, 1, n, total_kv]),
            self.device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        self.offloader
            .prefetch_block(0)
            .map_err(|e| Error::InvalidInput(format!("prefetch block 0: {e}")))?;
        for i in 0..total {
            let raw = self
                .offloader
                .await_block(i)
                .map_err(|e| Error::InvalidInput(format!("await block {i}: {e}")))?;
            if i + 1 < total {
                self.offloader
                    .prefetch_block(i + 1)
                    .map_err(|e| Error::InvalidInput(format!("prefetch block {}: {e}", i + 1)))?;
            }
            let lw = Self::untranspose_block_weights(&raw)?;
            let (past_k, past_v) = &cache.layers[i];
            let (new_h, k_full, v_full) = Self::und_layer_step_chunk(
                cfg, i, &lw, &hidden, &cos_t, &sin_t, past_k, past_v, &attn_mask,
            )?;
            cache.layers[i] = (k_full, v_full);
            hidden = new_h;
        }

        let normed = Self::rms_norm_apply(&hidden, &final_norm_w, cfg.rms_norm_eps)?;

        cache.prefix_len += n;
        cache.next_t_index += n;
        Ok(normed)
    }

    /// Multi-token base-mode forward step with past-K/V concat. Same
    /// semantics as `und_layer_step` but accepts `n > 1` and a custom
    /// attention mask sized `[1, 1, n, past_len + n]`.
    #[allow(clippy::too_many_arguments)]
    fn und_layer_step_chunk(
        cfg: &SenseNovaU1Config,
        i: usize,
        lw: &HashMap<String, Tensor>,
        hidden: &Tensor,
        cos_t: &Tensor,
        sin_t: &Tensor,
        past_k: &Tensor,
        past_v: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let h_total = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h_total / h_kv;
        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        let lget = |k: &str| -> Result<&Tensor> {
            lw.get(k).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "SenseNovaU1: missing layer-{i} weight {k}"
                ))
            })
        };

        let normed = Self::rms_norm_apply(
            hidden,
            lget(&format!("language_model.model.layers.{i}.input_layernorm.weight"))?,
            cfg.rms_norm_eps,
        )?;
        let q = Self::linear_no_bias(
            &normed,
            lget(&format!("language_model.model.layers.{i}.self_attn.q_proj.weight"))?,
        )?;
        let k = Self::linear_no_bias(
            &normed,
            lget(&format!("language_model.model.layers.{i}.self_attn.k_proj.weight"))?,
        )?;
        let v = Self::linear_no_bias(
            &normed,
            lget(&format!("language_model.model.layers.{i}.self_attn.v_proj.weight"))?,
        )?;
        let q = q.reshape(&[b, n, h_total, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        let q_norm = lget(&format!("language_model.model.layers.{i}.self_attn.q_norm.weight"))?;
        let q_norm_hw = lget(&format!("language_model.model.layers.{i}.self_attn.q_norm_hw.weight"))?;
        let k_norm = lget(&format!("language_model.model.layers.{i}.self_attn.k_norm.weight"))?;
        let k_norm_hw = lget(&format!("language_model.model.layers.{i}.self_attn.k_norm_hw.weight"))?;

        let (q_t, q_hw) = Self::chunk_last_half(&q)?;
        let q_t = Self::head_rms_norm(&q_t, q_norm, cfg.rms_norm_eps)?;
        let q_hw = Self::head_rms_norm(&q_hw, q_norm_hw, cfg.rms_norm_eps)?;
        let q_t = flame_core::bf16_ops::rope_halfsplit_bf16(&q_t, cos_t, sin_t)?;
        let q = Tensor::cat(&[&q_t, &q_hw], 3)?;
        let (k_t, k_hw) = Self::chunk_last_half(&k)?;
        let k_t = Self::head_rms_norm(&k_t, k_norm, cfg.rms_norm_eps)?;
        let k_hw = Self::head_rms_norm(&k_hw, k_norm_hw, cfg.rms_norm_eps)?;
        let k_t = flame_core::bf16_ops::rope_halfsplit_bf16(&k_t, cos_t, sin_t)?;
        let k = Tensor::cat(&[&k_t, &k_hw], 3)?;

        let k_full = Tensor::cat(&[past_k, &k], 2)?;
        let v_full = Tensor::cat(&[past_v, &v], 2)?;
        let k_g = Self::repeat_kv(&k_full, n_rep)?;
        let v_g = Self::repeat_kv(&v_full, n_rep)?;
        let attn = flame_core::attention::sdpa(&q, &k_g, &v_g, Some(attn_mask))?;
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h_total * d])?;
        let attn = Self::linear_no_bias(
            &attn,
            lget(&format!("language_model.model.layers.{i}.self_attn.o_proj.weight"))?,
        )?;
        let hidden = hidden.add(&attn)?;

        let post_norm_w = lget(&format!(
            "language_model.model.layers.{i}.post_attention_layernorm.weight"
        ))?;
        let n2 = Self::rms_norm_apply(&hidden, post_norm_w, cfg.rms_norm_eps)?;
        let gate_w = lget(&format!("language_model.model.layers.{i}.mlp.gate_proj.weight"))?;
        let up_w = lget(&format!("language_model.model.layers.{i}.mlp.up_proj.weight"))?;
        let down_w = lget(&format!("language_model.model.layers.{i}.mlp.down_proj.weight"))?;
        let gate = Self::linear_no_bias(&n2, gate_w)?;
        let up = Self::linear_no_bias(&n2, up_w)?;
        let mlp = gate.silu()?.mul(&up)?;
        let mlp = Self::linear_no_bias(&mlp, down_w)?;
        let hidden = hidden.add(&mlp)?;

        Ok((hidden, k_full, v_full))
    }

    // -----------------------------------------------------------------------
    // Phase 6 (cap. 1/2/4): base-mode 3D-RoPE prefix forward.
    // -----------------------------------------------------------------------
    //
    // Reference: `Qwen3Model.forward` (modeling_qwen3.py:1027) — when
    // `image_gen_indicators is None` it sets `exist_image_gen_tokens = False`
    // and dispatches every layer to `forward_und`, i.e. pure base weights.
    // The 3D RoPE in `Qwen3Attention.forward_und` (line 422) is per-token,
    // so image-context tokens with non-zero h/w indexes pick up the right
    // spatial rotations even though projections stay on base weights.
    //
    // The earlier `forward_mixed_prefix` routed image-context positions to
    // `_mot_gen` weights and produced gibberish on the VQA smoke. We now
    // mirror the python and keep the call signature the same (still accepts
    // `image_mask` for forward-compat with future autonomous-mixed paths).

    /// Run a 3D-RoPE base-mode prefix forward over already-spliced
    /// embeddings.
    ///
    /// `hidden_in`: `[1, N, 4096]` BF16 — text tokens embedded via
    /// `embed_tokens`, image tokens (originally `<IMG_CONTEXT>`) replaced by
    /// features from `extract_feature_und`. `_image_mask` is currently unused
    /// (always-base routing) but kept on the signature so callers don't need
    /// to drop it. `t_indexes / h_indexes / w_indexes` come from
    /// `build_thw_indexes`. Returns `(KvCache, last_hidden)` with
    /// `last_hidden` FINAL-NORMED — same contract as `forward_und`.
    pub fn forward_mixed_prefix(
        &mut self,
        hidden_in: &Tensor,
        _image_mask: &[bool],
        t_indexes: &[i32],
        h_indexes: &[i32],
        w_indexes: &[i32],
    ) -> Result<(KvCache, Tensor)> {
        let dims = hidden_in.shape().dims().to_vec();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(Error::InvalidInput(format!(
                "forward_mixed_prefix: hidden must be [1, N, hidden], got {dims:?}"
            )));
        }
        let n = dims[1];
        if t_indexes.len() != n || h_indexes.len() != n || w_indexes.len() != n {
            return Err(Error::InvalidInput(format!(
                "forward_mixed_prefix: index arrays must all be length N={n}, got t={} h={} w={}",
                t_indexes.len(), h_indexes.len(), w_indexes.len()
            )));
        }

        let Self { config, shared, device, offloader } = self;
        let cfg = &*config;
        let total = cfg.num_layers;
        let (dim_t, dim_h, dim_w) = cfg.rope_dims();

        // Per-token 3D RoPE tables.
        let (cos_t, sin_t) = build_rope_for_positions(t_indexes, dim_t, cfg.rope_theta, device)?;
        let (cos_h, sin_h) = build_rope_for_positions(h_indexes, dim_h, cfg.rope_theta_hw, device)?;
        let (cos_w, sin_w) = build_rope_for_positions(w_indexes, dim_w, cfg.rope_theta_hw, device)?;

        // Block-causal attention mask from t-indexes (matches python
        // `create_block_causal_mask(indexes[0])`).
        let attn_mask = build_block_causal_mask(t_indexes, device)?;

        offloader
            .prefetch_block(0)
            .map_err(|e| Error::InvalidInput(format!("prefetch block 0: {e}")))?;
        let mut hidden = hidden_in.clone();
        let mut cache_layers: Vec<(Tensor, Tensor)> = Vec::with_capacity(total);
        for i in 0..total {
            let raw = offloader
                .await_block(i)
                .map_err(|e| Error::InvalidInput(format!("await block {i}: {e}")))?;
            if i + 1 < total {
                offloader
                    .prefetch_block(i + 1)
                    .map_err(|e| Error::InvalidInput(format!("prefetch block {}: {e}", i + 1)))?;
            }
            let lw = Self::untranspose_block_weights(&raw)?;
            let (new_h, k_cache, v_cache) = Self::und_layer_3d(
                cfg, i, &lw, &hidden,
                &cos_t, &sin_t, &cos_h, &sin_h, &cos_w, &sin_w, &attn_mask,
            )?;
            cache_layers.push((k_cache, v_cache));
            hidden = new_h;
        }

        // Final norm — base path only (matches python Qwen3Model.forward
        // line 1140, `not exist_image_gen_tokens` branch).
        let final_norm = shared
            .get("language_model.model.norm.weight")
            .ok_or_else(|| {
                Error::InvalidInput("forward_mixed_prefix: missing model.norm.weight".into())
            })?;
        let hidden = Self::rms_norm_apply(&hidden, final_norm, cfg.rms_norm_eps)?;

        let max_t = t_indexes.iter().copied().max().unwrap_or(0);
        let next_t_index = (max_t as usize) + 1;

        Ok((
            KvCache {
                layers: cache_layers,
                prefix_len: n,
                next_t_index,
            },
            hidden,
        ))
    }

    /// Per-layer base-mode body with full 3D RoPE. Same routing as
    /// `und_layer` (text prefix) but takes per-token h/w RoPE tables instead
    /// of skipping them. Used by `forward_mixed_prefix` for VQA / it2i
    /// prefixes where IMG_CONTEXT positions carry non-zero h/w indexes.
    #[allow(clippy::too_many_arguments)]
    fn und_layer_3d(
        cfg: &SenseNovaU1Config,
        i: usize,
        lw: &HashMap<String, Tensor>,
        hidden: &Tensor,
        cos_t: &Tensor,
        sin_t: &Tensor,
        cos_h: &Tensor,
        sin_h: &Tensor,
        cos_w: &Tensor,
        sin_w: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let h_total = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h_total / h_kv;
        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        let lget = |k: &str| -> Result<&Tensor> {
            lw.get(k).ok_or_else(|| {
                Error::InvalidInput(format!("SenseNovaU1: missing layer-{i} weight {k}"))
            })
        };

        let normed = Self::rms_norm_apply(
            hidden,
            lget(&format!("language_model.model.layers.{i}.input_layernorm.weight"))?,
            cfg.rms_norm_eps,
        )?;
        let q = Self::linear_no_bias(
            &normed,
            lget(&format!("language_model.model.layers.{i}.self_attn.q_proj.weight"))?,
        )?;
        let k = Self::linear_no_bias(
            &normed,
            lget(&format!("language_model.model.layers.{i}.self_attn.k_proj.weight"))?,
        )?;
        let v = Self::linear_no_bias(
            &normed,
            lget(&format!("language_model.model.layers.{i}.self_attn.v_proj.weight"))?,
        )?;

        let q = q.reshape(&[b, n, h_total, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        let q_norm = lget(&format!("language_model.model.layers.{i}.self_attn.q_norm.weight"))?;
        let q_norm_hw = lget(&format!("language_model.model.layers.{i}.self_attn.q_norm_hw.weight"))?;
        let k_norm = lget(&format!("language_model.model.layers.{i}.self_attn.k_norm.weight"))?;
        let k_norm_hw = lget(&format!("language_model.model.layers.{i}.self_attn.k_norm_hw.weight"))?;

        let q = Self::apply_3d_rope(
            &q, q_norm, q_norm_hw, cfg.rms_norm_eps,
            cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
        )?;
        let k = Self::apply_3d_rope(
            &k, k_norm, k_norm_hw, cfg.rms_norm_eps,
            cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
        )?;

        let k_cache = k.clone();
        let v_cache = v.clone();

        let k_g = Self::repeat_kv(&k, n_rep)?;
        let v_g = Self::repeat_kv(&v, n_rep)?;
        let attn = flame_core::attention::sdpa(&q, &k_g, &v_g, Some(attn_mask))?;
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h_total * d])?;
        let attn = Self::linear_no_bias(
            &attn,
            lget(&format!("language_model.model.layers.{i}.self_attn.o_proj.weight"))?,
        )?;
        let hidden = hidden.add(&attn)?;

        let post_norm_w = lget(&format!(
            "language_model.model.layers.{i}.post_attention_layernorm.weight"
        ))?;
        let n2 = Self::rms_norm_apply(&hidden, post_norm_w, cfg.rms_norm_eps)?;
        let gate_w = lget(&format!("language_model.model.layers.{i}.mlp.gate_proj.weight"))?;
        let up_w = lget(&format!("language_model.model.layers.{i}.mlp.up_proj.weight"))?;
        let down_w = lget(&format!("language_model.model.layers.{i}.mlp.down_proj.weight"))?;
        let gate = Self::linear_no_bias(&n2, gate_w)?;
        let up = Self::linear_no_bias(&n2, up_w)?;
        let mlp = gate.silu()?.mul(&up)?;
        let mlp = Self::linear_no_bias(&mlp, down_w)?;
        let hidden = hidden.add(&mlp)?;

        Ok((hidden, k_cache, v_cache))
    }

    // -----------------------------------------------------------------------
    // Index/embedding helpers used by VQA + it2i + interleaved bins.
    // -----------------------------------------------------------------------

    /// Build the per-token `(t_indexes, h_indexes, w_indexes)` arrays for a
    /// tokenized prefix, mirroring `get_thw_indexes` (modeling_neo_chat.py:1847).
    /// `img_context_token_id` and `img_start_token_id` mark the
    /// `<IMG_CONTEXT>` / `<img>` positions in `input_ids`. `image_grid` is a
    /// list of `(token_h, token_w)` post-merge grids in the same order they
    /// appear in `input_ids`. `image_mask` returns booleans `true` exactly at
    /// the `<IMG_CONTEXT>` slots.
    pub fn build_thw_indexes(
        &self,
        input_ids: &[i32],
        img_context_token_id: i32,
        img_start_token_id: i32,
        image_grid: &[(usize, usize)],
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<bool>)> {
        let n = input_ids.len();
        let img_start_shift: Vec<i32> = std::iter::once(0)
            .chain(input_ids.iter().take(n - 1).map(|&id| {
                if id == img_start_token_id { 1 } else { 0 }
            }))
            .collect();
        let not_img: Vec<i32> = input_ids
            .iter()
            .map(|&id| if id != img_context_token_id { 1 } else { 0 })
            .collect();
        let mut t_indexes: Vec<i32> = Vec::with_capacity(n);
        let mut acc: i32 = 0;
        for k in 0..n {
            acc += img_start_shift[k] + not_img[k];
            t_indexes.push(acc - 1);
        }
        let mut h_indexes: Vec<i32> = vec![0; n];
        let mut w_indexes: Vec<i32> = vec![0; n];
        let image_mask: Vec<bool> = input_ids.iter().map(|&id| id == img_context_token_id).collect();

        // Walk runs of `<IMG_CONTEXT>` and assign h/w from the next image grid.
        let mut img_idx = 0usize;
        let mut k = 0usize;
        while k < n {
            if image_mask[k] {
                let (token_h, token_w) = *image_grid.get(img_idx).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "build_thw_indexes: input_ids has more <IMG_CONTEXT> runs than image_grid entries ({})",
                        image_grid.len()
                    ))
                })?;
                let l = token_h * token_w;
                if k + l > n || !image_mask[k..k + l].iter().all(|&b| b) {
                    return Err(Error::InvalidInput(format!(
                        "build_thw_indexes: <IMG_CONTEXT> run at {k} length {l} doesn't match grid {token_h}x{token_w}"
                    )));
                }
                for j in 0..l {
                    h_indexes[k + j] = (j / token_w) as i32;
                    w_indexes[k + j] = (j % token_w) as i32;
                }
                k += l;
                img_idx += 1;
            } else {
                k += 1;
            }
        }
        if img_idx != image_grid.len() {
            return Err(Error::InvalidInput(format!(
                "build_thw_indexes: input_ids consumed {img_idx} images but image_grid had {}",
                image_grid.len()
            )));
        }
        Ok((t_indexes, h_indexes, w_indexes, image_mask))
    }

    /// Embed `input_ids` to `[1, N, hidden]` and splice each image's features
    /// in place of its `<IMG_CONTEXT>` slot run. `image_features` is one
    /// `[1, L_i, hidden]` tensor per image, in the same order they appear in
    /// `input_ids`. Mirrors the index-assignment block at
    /// `modeling_neo_chat.py:1810-1815` (chat) and `:614-622` (it2i).
    pub fn embed_with_image_splice(
        &self,
        input_ids: &[i32],
        img_context_token_id: i32,
        image_features: &[Tensor],
    ) -> Result<Tensor> {
        let n = input_ids.len();
        let embed_w = self.shared_get("language_model.model.embed_tokens.weight")?;
        let ids = Tensor::from_vec(
            input_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[n]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let base = embed_w.index_select0(&ids)?; // [N, hidden]
        let hidden_size = self.config.hidden_size;

        // Walk runs of <IMG_CONTEXT> and stitch a sequence of segments.
        let mut segs: Vec<Tensor> = Vec::new();
        let mut img_idx = 0usize;
        let mut k = 0usize;
        let is_ctx = |id: i32| id == img_context_token_id;
        while k < n {
            if is_ctx(input_ids[k]) {
                // Find the run length.
                let mut j = k;
                while j < n && is_ctx(input_ids[j]) {
                    j += 1;
                }
                let l = j - k;
                let feat = image_features.get(img_idx).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "embed_with_image_splice: ran out of image_features at run {img_idx}"
                    ))
                })?;
                let fdims = feat.shape().dims();
                if fdims.len() != 3 || fdims[0] != 1 || fdims[1] != l || fdims[2] != hidden_size {
                    return Err(Error::InvalidInput(format!(
                        "embed_with_image_splice: image_features[{img_idx}] expected [1, {l}, {hidden_size}], got {fdims:?}"
                    )));
                }
                segs.push(feat.reshape(&[l, hidden_size])?);
                img_idx += 1;
                k = j;
            } else {
                let mut j = k;
                while j < n && !is_ctx(input_ids[j]) {
                    j += 1;
                }
                let l = j - k;
                segs.push(base.narrow(0, k, l)?);
                k = j;
            }
        }
        if img_idx != image_features.len() {
            return Err(Error::InvalidInput(format!(
                "embed_with_image_splice: input_ids consumed {img_idx} images but image_features had {}",
                image_features.len()
            )));
        }
        let stitched = if segs.len() == 1 {
            segs.into_iter().next().unwrap()
        } else {
            let refs: Vec<&Tensor> = segs.iter().collect();
            Tensor::cat(&refs, 0)?
        };
        stitched.reshape(&[1, n, hidden_size])
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TimeOrScale {
    Timestep,
    NoiseScale,
}

// ---------------------------------------------------------------------------
// Free helpers (used by forward_und today; forward_gen will reuse).
// ---------------------------------------------------------------------------

/// Build 1D half-split RoPE tables `[1, 1, seq_len, dim/2]` (cos, sin) in BF16.
///
/// Half-split (HF convention): `cos`/`sin` are size `dim/2`, applied via
/// `flame_core::bf16_ops::rope_halfsplit_bf16` to a `[B, H, N, dim]` tensor.
fn build_rope_table_1d(
    seq_len: usize,
    dim: usize,
    theta: f64,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    if dim % 2 != 0 {
        return Err(Error::InvalidInput(format!(
            "build_rope_table_1d: dim must be even, got {dim}"
        )));
    }
    let half = dim / 2;
    let pos = Tensor::arange(0.0, seq_len as f32, 1.0, device.clone())?;
    let freq_idx = Tensor::arange(0.0, dim as f32, 2.0, device.clone())?;
    let log_theta = (theta as f32).ln();
    let scale = -log_theta / (dim as f32);
    let log_freqs = freq_idx.mul_scalar(scale)?.exp()?;
    let pos_col = pos.reshape(&[seq_len, 1])?;
    let freq_row = log_freqs.reshape(&[1, half])?;
    let angles = pos_col.matmul(&freq_row)?;
    let cos = angles.cos()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
    let sin = angles.sin()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

/// Sinusoidal frequency embedding: `[N]` scalar Tensor → `[N, dim]` BF16.
///
/// Reference: `modeling_fm_modules.py::TimestepEmbedder.timestep_embedding`
/// (line 37). Layout is `cat([cos(args), sin(args)], dim=-1)` — note `cos`
/// FIRST, `sin` second. `freqs[j] = exp(-log(max_period) * j / half)` for
/// `j ∈ [0, half)`.
fn sinusoidal_freq_embed(
    t: &Tensor,
    dim: usize,
    max_period: f32,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    if dim % 2 != 0 {
        return Err(Error::InvalidInput(format!(
            "sinusoidal_freq_embed: dim must be even, got {dim}"
        )));
    }
    let half = dim / 2;
    // freqs = exp(-log(max_period) * arange(half) / half)
    let idx = Tensor::arange(0.0, half as f32, 1.0, device.clone())?;
    let log_period = max_period.ln();
    let scale = -log_period / (half as f32);
    let freqs = idx.mul_scalar(scale)?.exp()?; // [half]

    // args[i, j] = t[i] * freqs[j] — matmul [N, 1] @ [1, half] = [N, half]
    let n = t.shape().dims()[0];
    let t_col = t.reshape(&[n, 1])?;
    let f_row = freqs.reshape(&[1, half])?;
    let args = t_col.matmul(&f_row)?;

    let cos = args.cos()?;
    let sin = args.sin()?;
    // cat([cos, sin], dim=-1) — matches reference line 52.
    Tensor::cat(&[&cos, &sin], 1)?.to_dtype(DType::BF16)
}

/// Build half-split RoPE tables for an explicit list of integer positions.
///
/// Generalization of `build_rope_table_1d` (which uses `arange(seq_len)`).
/// For the gen path, positions vary per token (constant t, variable h, variable
/// w over the patch grid). Output: `(cos, sin)` at `[1, 1, positions.len(), dim/2]` BF16.
fn build_rope_for_positions(
    positions: &[i32],
    dim: usize,
    theta: f64,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    if dim % 2 != 0 {
        return Err(Error::InvalidInput(format!(
            "build_rope_for_positions: dim must be even, got {dim}"
        )));
    }
    let n = positions.len();
    let half = dim / 2;
    let freq_idx = Tensor::arange(0.0, dim as f32, 2.0, device.clone())?;
    let log_theta = (theta as f32).ln();
    let scale = -log_theta / (dim as f32);
    let log_freqs = freq_idx.mul_scalar(scale)?.exp()?;
    let pos = Tensor::from_vec(
        positions.iter().map(|&p| p as f32).collect(),
        Shape::from_dims(&[n]),
        device.clone(),
    )?;
    let pos_col = pos.reshape(&[n, 1])?;
    let freq_row = log_freqs.reshape(&[1, half])?;
    let angles = pos_col.matmul(&freq_row)?;
    let cos = angles.cos()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
    let sin = angles.sin()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

/// Build a 0/1 BF16 causal mask `[1, 1, N, N]` consumable by
/// `flame_core::attention::sdpa`. `1` = keep, `0` = block.
///
/// Allows `i` to attend to `j` iff `j <= i AND j < real_len`.
fn build_causal_mask(
    seq_len: usize,
    real_len: usize,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i && j < real_len {
                data[i * seq_len + j] = 1.0;
            }
        }
    }
    let mask_f32 = Tensor::from_vec(
        data,
        Shape::from_dims(&[1, 1, seq_len, seq_len]),
        device.clone(),
    )?;
    mask_f32.to_dtype(DType::BF16)
}

/// Block-causal mask from a 1-D `t_indexes` array, mirroring
/// `create_block_causal_mask` (modeling_qwen3.py:152). Token i may attend to
/// token j when either `t_index[j] == t_index[i]` (same-t block, bidirectional)
/// or `j <= i` (sequence-order causal). Returns `[1, 1, L, L]` BF16 0/1.
fn build_block_causal_mask(
    t_indexes: &[i32],
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let l = t_indexes.len();
    let mut data = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            if j <= i || t_indexes[j] == t_indexes[i] {
                data[i * l + j] = 1.0;
            }
        }
    }
    let m = Tensor::from_vec(data, Shape::from_dims(&[1, 1, l, l]), device.clone())?;
    m.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// Weight-key generation (used by the future loader to filter shared vs per-layer)
// ---------------------------------------------------------------------------

/// Returns true iff `key` belongs to per-layer transformer weights (any of the
/// 22 tensors documented at the top of this file).
///
/// Used by the future BlockFacilitator to classify keys into layer indices.
/// The key shape is `language_model.model.layers.{i}.<...>` for ALL per-layer
/// weights — `_mot_gen` variants ride on the same `layers.{i}.` prefix.
pub fn classify_layer_key(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("language_model.model.layers.")?;
    rest.split('.').next()?.parse().ok()
}

/// Shared-weight prefix list used to filter the resident hashmap during load.
pub const SHARED_PREFIXES: &[&str] = &[
    "language_model.model.embed_tokens.",
    "language_model.model.norm.",
    "language_model.model.norm_mot_gen.",
    "language_model.lm_head.",
    "fm_modules.",
    // `vision_model.embeddings.*` is for understanding (VQA); we keep it
    // resident regardless so a future port to it2i / VQA modes can attach
    // without reloading.
    "vision_model.embeddings.",
];

/// All 22 expected per-layer weight keys for layer index `i`. Order matches
/// the documentation block at the top of this file.
///
/// Used both for load-time validation (every key must be present) and as a
/// canonical iterate-order for any test that wants to hit each weight.
pub fn expected_per_layer_keys(i: usize) -> Vec<String> {
    let p = format!("language_model.model.layers.{i}");
    vec![
        format!("{p}.input_layernorm.weight"),
        format!("{p}.input_layernorm_mot_gen.weight"),
        format!("{p}.post_attention_layernorm.weight"),
        format!("{p}.post_attention_layernorm_mot_gen.weight"),
        format!("{p}.self_attn.q_proj.weight"),
        format!("{p}.self_attn.q_proj_mot_gen.weight"),
        format!("{p}.self_attn.k_proj.weight"),
        format!("{p}.self_attn.k_proj_mot_gen.weight"),
        format!("{p}.self_attn.v_proj.weight"),
        format!("{p}.self_attn.v_proj_mot_gen.weight"),
        format!("{p}.self_attn.o_proj.weight"),
        format!("{p}.self_attn.o_proj_mot_gen.weight"),
        format!("{p}.self_attn.q_norm.weight"),
        format!("{p}.self_attn.q_norm_mot_gen.weight"),
        format!("{p}.self_attn.q_norm_hw.weight"),
        format!("{p}.self_attn.q_norm_hw_mot_gen.weight"),
        format!("{p}.self_attn.k_norm.weight"),
        format!("{p}.self_attn.k_norm_mot_gen.weight"),
        format!("{p}.self_attn.k_norm_hw.weight"),
        format!("{p}.self_attn.k_norm_hw_mot_gen.weight"),
        format!("{p}.mlp.gate_proj.weight"),
        format!("{p}.mlp.up_proj.weight"),
        format!("{p}.mlp.down_proj.weight"),
        format!("{p}.mlp_mot_gen.gate_proj.weight"),
        format!("{p}.mlp_mot_gen.up_proj.weight"),
        format!("{p}.mlp_mot_gen.down_proj.weight"),
    ]
}

/// All shared-weight keys required for SenseNova-U1 inference. Includes both
/// the gen-side `fm_modules.vision_model_mot_gen.embeddings.*` (T2I/it2i) and
/// the understanding-side `vision_model.embeddings.*` (VQA/it2i). The full
/// 8B-MoT checkpoint ships them all; loader fails fast if any is missing.
pub fn expected_shared_keys() -> &'static [&'static str] {
    &[
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.model.norm_mot_gen.weight",
        "language_model.lm_head.weight",
        // fm_modules MLPs: 2-layer with bias on both linear layers
        "fm_modules.timestep_embedder.mlp.0.weight",
        "fm_modules.timestep_embedder.mlp.0.bias",
        "fm_modules.timestep_embedder.mlp.2.weight",
        "fm_modules.timestep_embedder.mlp.2.bias",
        "fm_modules.noise_scale_embedder.mlp.0.weight",
        "fm_modules.noise_scale_embedder.mlp.0.bias",
        "fm_modules.noise_scale_embedder.mlp.2.weight",
        "fm_modules.noise_scale_embedder.mlp.2.bias",
        "fm_modules.fm_head.0.weight",
        "fm_modules.fm_head.0.bias",
        "fm_modules.fm_head.2.weight",
        "fm_modules.fm_head.2.bias",
        // Gen-side patch+merge embedder
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias",
        "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight",
        "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias",
        // Understanding-side patch+merge embedder (VQA + it2i image input)
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.patch_embedding.bias",
        "vision_model.embeddings.dense_embedding.weight",
        "vision_model.embeddings.dense_embedding.bias",
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_8b_mot_checkpoint() {
        let c = SenseNovaU1Config::default();
        assert_eq!(c.num_layers, 42);
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_heads, 32);
        assert_eq!(c.num_kv_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.merge_size(), 2);
        assert_eq!(c.fm_head_out_dim(), 32 * 32 * 3);
        assert_eq!(c.rope_dims(), (64, 32, 32));
    }

    #[test]
    fn classify_layer_key_handles_all_per_layer_variants() {
        for k in [
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.0.input_layernorm_mot_gen.weight",
            "language_model.model.layers.41.self_attn.q_norm_hw_mot_gen.weight",
            "language_model.model.layers.7.mlp_mot_gen.gate_proj.weight",
        ] {
            assert!(classify_layer_key(k).is_some(), "should classify: {k}");
        }
        for k in [
            "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
            "fm_modules.fm_head.0.weight",
            "vision_model.embeddings.patch_embedding.weight",
        ] {
            assert_eq!(classify_layer_key(k), None, "should NOT classify: {k}");
        }
        assert_eq!(
            classify_layer_key("language_model.model.layers.13.mlp.up_proj.weight"),
            Some(13)
        );
    }

    #[test]
    fn noise_scale_at_2048_is_capped_at_8() {
        // 2048×2048 → grid 128×128 = 16384 patches at p=16, merge=2 → tokens=4096
        // scale = sqrt(16384 / 4 / 64) = sqrt(64) = 8.0
        let cfg = SenseNovaU1Config::default();
        let merge = cfg.merge_size() as f32;
        let base = cfg.noise_scale_base_image_seq_len as f32;
        let raw = ((128.0 * 128.0 / (merge * merge) / base).sqrt()) * cfg.noise_scale;
        assert!((raw - 8.0).abs() < 1e-4);
        assert!(raw.min(cfg.noise_scale_max_value) <= cfg.noise_scale_max_value);
    }

    #[test]
    fn per_layer_key_count_is_26() {
        // 26 = 13 base + 13 _mot_gen, where 13 = 2 layer norms (input,
        // post_attention) + 4 attn projs (q,k,v,o) + 4 attn norms (q_norm,
        // q_norm_hw, k_norm, k_norm_hw) + 3 MLP projs (gate, up, down).
        let keys = expected_per_layer_keys(0);
        assert_eq!(keys.len(), 26);
    }

    #[test]
    fn per_layer_keys_all_classify_to_their_layer() {
        for i in [0usize, 7, 13, 41] {
            for k in expected_per_layer_keys(i) {
                assert_eq!(classify_layer_key(&k), Some(i), "key {k} should classify to layer {i}");
            }
        }
    }

    #[test]
    fn expected_shared_keys_are_disjoint_from_per_layer() {
        let layer0: std::collections::HashSet<String> =
            expected_per_layer_keys(0).into_iter().collect();
        for shared_key in expected_shared_keys() {
            assert!(
                !layer0.contains(*shared_key),
                "shared key {shared_key} must not appear in per-layer set"
            );
            assert_eq!(
                classify_layer_key(shared_key),
                None,
                "shared key {shared_key} must NOT classify to a layer"
            );
        }
    }

    #[test]
    fn shared_keys_match_shared_prefixes() {
        for shared_key in expected_shared_keys() {
            assert!(
                SHARED_PREFIXES.iter().any(|p| shared_key.starts_with(p)),
                "shared key {shared_key} doesn't match any SHARED_PREFIXES entry"
            );
        }
    }

    /// Smoke-test: verify the expected total tensor count matches the actual
    /// safetensors index json (1116 tensors per the 8B-MoT checkpoint).
    /// Uses ENV `SENSENOVA_U1_WEIGHTS` or the canonical local path; skipped
    /// silently if neither is present.
    #[test]
    fn index_json_total_count_matches_expectation() {
        let dir = std::env::var("SENSENOVA_U1_WEIGHTS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("/home/alex/.serenity/models/sensenova_u1"));
        let index = dir.join("model.safetensors.index.json");
        if !index.exists() {
            eprintln!("[skip] {index:?} not present");
            return;
        }
        let txt = std::fs::read_to_string(&index).unwrap();
        let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
        let map = v.get("weight_map").and_then(|x| x.as_object()).unwrap();
        let total = map.len();
        let cfg = SenseNovaU1Config::default();
        let expected_per_layer = expected_per_layer_keys(0).len() * cfg.num_layers;
        let expected_shared = expected_shared_keys().len();
        let computed = expected_per_layer + expected_shared;
        assert_eq!(
            total, computed,
            "index.json has {total} keys; computed expected {computed} \
             (per_layer={expected_per_layer}, shared={expected_shared})"
        );
    }
}
