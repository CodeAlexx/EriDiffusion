//! Shared LyCORIS bundle abstraction for all EDv2 trainers.
//!
//! Phase 2a: this module wraps `lycoris-rs` (LoCon / LoHa / LoKr / Full / Diag-OFT)
//! into a per-trainer collection that owns adapters keyed by dotted path
//! (e.g. `transformer.blocks.0.attn.q_proj`). Each trainer's wiring agent
//! constructs one [`LycorisBundle`], pushes adapters via [`add_linear`] /
//! [`add_conv2d`] for every targeted layer, then:
//!
//! 1. Calls [`LycorisBundle::to_parameters`] to register tensors with the
//!    optimizer.
//! 2. Calls [`LycorisBundle::get`] per layer in the model's forward pass,
//!    branching on the returned [`LycorisAdapter`] variant.
//! 3. Calls [`LycorisBundle::save_safetensors`] / [`LycorisBundle::load_safetensors`]
//!    around the trainer's checkpoint cycle.
//!
//! The legacy `lora.rs::LoRALinear` path remains for the [`LycorisAlgo::None`]
//! case — trainers should fall back to it when the user did not request a
//! LyCORIS algo. Drop-in replacement of `LoRALinear` with `LycorisAdapter::LoCon`
//! is intentionally **out of scope** here — that's per-trainer Phase 2b work.
//!
//! # DoRA support
//!
//! When `config.dora == true`, `add_linear` / `add_conv2d` requires `w_orig` to
//! be `Some(&Tensor)` (the pretrained weight) so the per-target magnitude can
//! be initialized from `||W_orig||_2`. The magnitude tensor is stored in
//! `dora_magnitudes[i]` parallel to `adapters[i]`. DoRA is supported on
//! LoCon / LoHa / LoKr / Full only — OFT support is deferred (the DoRA paper's
//! direction/magnitude split conflicts with OFT's multiplicative form).
//!
//! # Save/load conventions
//!
//! Adapter tensors are written with PEFT-style suffixes:
//!
//! | Algo  | Suffixes                                                            |
//! | ----- | ------------------------------------------------------------------- |
//! | LoCon | `.lora_A.weight`, `.lora_B.weight`, optional `.lora_mid.weight`     |
//! | LoHa  | `.hada_w1_a/b.weight`, `.hada_w2_a/b.weight`, optional `.hada_t1/2` |
//! | LoKr  | `.lokr_w1[.weight]` or `.lokr_w1_a/b`, `.lokr_w2[.weight]` or       |
//! |       | `.lokr_w2_a/b[.weight]`, optional `.lokr_t2.weight`                 |
//! | Full  | `.diff.weight`, optional `.diff_b`                                  |
//! | OFT   | `.oft_blocks.weight`                                                |
//! | DoRA  | `.dora_scale` appended for any of the above                         |
//!
//! `prefix` is prepended to every key (e.g., pass `"transformer"` to namespace
//! the entire DiT). `load_safetensors` detects algo from suffix presence and
//! validates against `self.config.algo` — bails on mismatch.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context};
use serde::Deserialize;
use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};

use lycoris_rs::{
    algorithms::{full::FullAdapter, locon::LoConModule, loha::LoHaModule, lokr::LoKrModule, oft::OFTModule},
    dora::{apply_weight_decompose, init_magnitude},
    LycorisAdapter, LycorisModule, StorageDtype,
};

/// PEFT / SimpleTuner `--lora_init_type` selector. Re-exported from
/// `lycoris-rs` so trainer binaries can import it via the same module they
/// already use for `LycorisAlgo` / `LycorisBundleConfig`.
pub use lycoris_rs::LoraInitType;

/// SimpleTuner `--lycoris_config preset.json` schema.
///
/// SimpleTuner uses a JSON file to map per-target-pattern algo configs onto
/// individual layer families: e.g. `Attention` blocks get one rank/factor,
/// `FeedForward` blocks get another. The file shape is:
///
/// ```json
/// {
///   "algo": "lokr",
///   "multiplier": 1.0,
///   "full_matrix": true,
///   "linear_dim": 10000,
///   "linear_alpha": 1,
///   "factor": 16,
///   "apply_preset": {
///     "target_module": ["Attention", "FeedForward"],
///     "module_algo_map": {
///       "Attention":    { "factor": 16 },
///       "FeedForward":  { "factor": 8  }
///     }
///   }
/// }
/// ```
///
/// **Pattern matching**: SimpleTuner's `target_module` lists module *class
/// names* and walks the PyTorch module tree. We don't have class names —
/// adapters are keyed by dotted-path strings (e.g.
/// `transformer.blocks.0.attn.q_proj`). The Rust port matches each pattern
/// as a **case-insensitive substring** against the adapter name. So a pattern
/// `"Attention"` hits any name containing `attention` / `attn` / `xattn`,
/// and a pattern `"FeedForward"` matches `feedforward` / `ffn` / `mlp`. To
/// target both attention and FFN under the SimpleTuner naming convention,
/// users typically write patterns such as `["attn", "ffn"]` for our trainers.
///
/// Top-level fields are applied as the bundle defaults; `module_algo_map`
/// entries override per-target. Unset override fields fall back to the
/// top-level value.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct LycorisConfigFile {
    /// Top-level algo. Used when `module_algo_map` doesn't override.
    #[serde(default)]
    pub algo: Option<String>,
    #[serde(default)]
    pub multiplier: Option<f32>,
    /// LoKr `full_matrix` (no rank decomposition; W1 and W2 stored full).
    #[serde(default)]
    pub full_matrix: Option<bool>,
    /// LoKr / generic rank. Maps to `LycorisBundleConfig::rank`.
    #[serde(default)]
    pub linear_dim: Option<usize>,
    /// LoKr / generic alpha. Maps to `LycorisBundleConfig::alpha`.
    #[serde(default)]
    pub linear_alpha: Option<f32>,
    /// LoKr factor.
    #[serde(default)]
    pub factor: Option<i32>,
    /// LoKr `decompose_both`.
    #[serde(default)]
    pub decompose_both: Option<bool>,
    /// LoKr `bypass_mode` (accepted for compatibility; currently a no-op
    /// in the Rust port).
    #[serde(default)]
    pub bypass_mode: Option<bool>,
    /// Tucker decomposition for conv variants.
    #[serde(default)]
    pub use_tucker: Option<bool>,
    /// DoRA enable flag (top-level).
    #[serde(default)]
    pub dora: Option<bool>,
    /// LoRA init type (top-level).
    #[serde(default)]
    pub lora_init_type: Option<String>,
    /// Per-target preset.
    #[serde(default)]
    pub apply_preset: Option<LycorisPreset>,
}

/// `apply_preset` block from a SimpleTuner-style lycoris_config.json.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct LycorisPreset {
    /// Module name patterns to target (case-insensitive substring match
    /// against the adapter's dotted path). Empty list = match nothing
    /// (caller falls back to the trainer's normal target picker).
    #[serde(default)]
    pub target_module: Vec<String>,
    /// Per-pattern overrides. Pattern matching is the same as
    /// `target_module`. The first matching pattern's override wins.
    #[serde(default)]
    pub module_algo_map: HashMap<String, LycorisModuleOverride>,
}

/// Per-pattern override from `module_algo_map`. Any field left as `None`
/// inherits from the top-level [`LycorisConfigFile`] / bundle config.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct LycorisModuleOverride {
    #[serde(default)]
    pub algo: Option<String>,
    #[serde(default)]
    pub linear_dim: Option<usize>,
    #[serde(default)]
    pub linear_alpha: Option<f32>,
    #[serde(default)]
    pub factor: Option<i32>,
    #[serde(default)]
    pub decompose_both: Option<bool>,
    #[serde(default)]
    pub use_tucker: Option<bool>,
    #[serde(default)]
    pub dora: Option<bool>,
    #[serde(default)]
    pub lora_init_type: Option<String>,
}

impl LycorisPreset {
    /// Find the first `module_algo_map` entry whose pattern matches `name`
    /// (case-insensitive substring). Returns `None` if no pattern matches.
    /// HashMap iteration is unordered — when multiple patterns match,
    /// the result is non-deterministic; callers should design patterns to
    /// be mutually exclusive.
    pub fn resolve(&self, name: &str) -> Option<&LycorisModuleOverride> {
        let lower = name.to_ascii_lowercase();
        self.module_algo_map
            .iter()
            .find(|(pat, _)| lower.contains(&pat.to_ascii_lowercase()))
            .map(|(_, ov)| ov)
    }

    /// Returns `true` if `name` is allowed by `target_module` (or the list
    /// is empty, which we treat as "match everything" so the file can be
    /// used as override-only without filtering).
    pub fn is_allowed(&self, name: &str) -> bool {
        if self.target_module.is_empty() {
            return true;
        }
        let lower = name.to_ascii_lowercase();
        self.target_module
            .iter()
            .any(|p| lower.contains(&p.to_ascii_lowercase()))
    }
}

impl LycorisConfigFile {
    /// Parse a SimpleTuner-style `lycoris_config.json`.
    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read lycoris_config: {}", path.display()))?;
        let cfg: LycorisConfigFile = serde_json::from_str(&raw)
            .with_context(|| format!("parse lycoris_config: {}", path.display()))?;
        Ok(cfg)
    }

    /// Build a [`LycorisBundleConfig`] from this file, overlaying top-level
    /// fields onto the supplied `base` config (CLI defaults). Fields absent
    /// from the file leave `base` untouched. The parsed `apply_preset` is
    /// returned alongside so `AdapterStore` can apply per-target overrides
    /// during construction.
    pub fn apply_to(
        &self,
        mut base: LycorisBundleConfig,
    ) -> anyhow::Result<(LycorisBundleConfig, Option<LycorisPreset>)> {
        if let Some(ref a) = self.algo {
            base.algo = LycorisAlgo::parse(a)?;
        }
        if let Some(d) = self.linear_dim {
            base.rank = d;
        }
        if let Some(a) = self.linear_alpha {
            base.alpha = a;
        }
        if let Some(f) = self.factor {
            base.factor = f;
        }
        if let Some(d) = self.decompose_both {
            base.decompose_both = d;
        }
        if let Some(t) = self.use_tucker {
            base.use_tucker = t;
        }
        if let Some(d) = self.dora {
            base.dora = d;
        }
        if let Some(ref t) = self.lora_init_type {
            base.init_type = LoraInitType::parse(t).map_err(|e| anyhow!("lora_init_type: {e}"))?;
        }
        Ok((base, self.apply_preset.clone()))
    }
}

/// LyCORIS algorithm selector. `None` means "fall back to legacy LoRALinear".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LycorisAlgo {
    /// Disabled — caller should use `lora::LoRALinear` instead.
    None,
    LoCon,
    LoHa,
    LoKr,
    Full,
    Oft,
}

impl LycorisAlgo {
    /// Parse from string (case-insensitive). Accepts `"lora"` as an alias for
    /// `LoCon` since LoCon at conv-degenerate-1×1 / linear-only is exactly
    /// the canonical LoRA decomposition.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        let lower = s.trim().to_ascii_lowercase();
        Ok(match lower.as_str() {
            "none" | "off" | "" => LycorisAlgo::None,
            "lora" | "locon" => LycorisAlgo::LoCon,
            "loha" => LycorisAlgo::LoHa,
            "lokr" => LycorisAlgo::LoKr,
            "full" => LycorisAlgo::Full,
            "oft" | "diag-oft" | "diag_oft" => LycorisAlgo::Oft,
            other => bail!(
                "unknown lycoris algo '{other}'; expected one of: none, lora, locon, loha, lokr, full, oft"
            ),
        })
    }

    /// Stable string identifier (matches the `parse` lowercase form).
    pub fn as_str(&self) -> &'static str {
        match self {
            LycorisAlgo::None => "none",
            LycorisAlgo::LoCon => "locon",
            LycorisAlgo::LoHa => "loha",
            LycorisAlgo::LoKr => "lokr",
            LycorisAlgo::Full => "full",
            LycorisAlgo::Oft => "oft",
        }
    }
}

/// Static configuration for a [`LycorisBundle`]. Set once at trainer init,
/// stored alongside the adapters for save/load validation.
#[derive(Clone, Debug)]
pub struct LycorisBundleConfig {
    pub algo: LycorisAlgo,
    pub rank: usize,
    pub alpha: f32,
    /// LoKr Kronecker split factor (default 16). Ignored for non-LoKr algos.
    pub factor: i32,
    /// OFT block size (default 32). Ignored for non-OFT algos.
    pub block_size: usize,
    /// OFT Cayley-Neumann series term count (default 5). Ignored for non-OFT.
    pub neumann_terms: usize,
    /// LoCon / LoHa / LoKr conv variant — Tucker decomposition for non-1×1 kernels.
    pub use_tucker: bool,
    /// Per-LyCORIS conv-layer rank override. `0` (default) → fall back to
    /// the top-level `rank` for conv targets. Mirrors ai-toolkit /
    /// SimpleTuner `network.conv` setting.
    pub conv_rank: usize,
    /// Per-LyCORIS conv-layer alpha override. `0.0` (default) → fall back
    /// to top-level `alpha`. Pairs with `conv_rank`.
    pub conv_alpha: f32,
    /// LoKr only: factorize both W1 *and* W2 (default false: only W2).
    pub decompose_both: bool,
    /// LoKr scalar gating flag — accepted for API stability, currently ignored
    /// (Phase 1.1 deferred the `scalar` field on `LoKrModule`).
    pub use_scalar: bool,
    /// Enable DoRA (weight-decomposed LoRA). Applies to LoCon/LoHa/LoKr/Full,
    /// **not** OFT.
    pub dora: bool,
    /// DoRA axis convention. `true` = lycoris-upstream (norm over input dims,
    /// magnitude shape `[out, 1]`). `false` = OneTrainer (norm over output dim,
    /// magnitude shape `[1, in]`). Default `true` to match upstream.
    pub dora_wd_on_out: bool,
    /// DoRA epsilon added to `||WP||_2` denominator. Default `1e-6`.
    pub dora_eps: f32,
    /// Storage dtype for trainable leaves. F32 default for trainer (AdamW state),
    /// BF16 for inference / merge-only.
    pub storage: StorageDtype,
    /// PEFT / SimpleTuner `lora_init_type` parity. Applies to the LoCon (LoRA)
    /// path only — other algos retain their algorithm-specific upstream init.
    /// `Default` (current) leaves Phase-2b byte-identical; `Gaussian` switches
    /// LoCon's `down`-side init to `N(0, 1/rank)`.
    pub init_type: LoraInitType,
    /// Optional SimpleTuner-style `lycoris_config preset.json` —
    /// per-target overrides resolved at adapter construction time. `None`
    /// = bundle uses uniform top-level config. See [`LycorisPreset`] for
    /// the matching rules.
    pub preset: Option<LycorisPreset>,
}

impl Default for LycorisBundleConfig {
    fn default() -> Self {
        Self {
            algo: LycorisAlgo::None,
            rank: 16,
            alpha: 16.0,
            factor: 16,
            block_size: 32,
            neumann_terms: 5,
            use_tucker: false,
            conv_rank: 0,
            conv_alpha: 0.0,
            decompose_both: false,
            use_scalar: false,
            dora: false,
            dora_wd_on_out: true,
            dora_eps: 1e-6,
            storage: StorageDtype::F32,
            init_type: LoraInitType::Default,
            preset: None,
        }
    }
}

/// Owning collection of LyCORIS adapters for one trainer. `adapters[i]`,
/// `names[i]`, and `dora_magnitudes[i]` are kept in lock-step.
pub struct LycorisBundle {
    pub config: LycorisBundleConfig,
    pub adapters: Vec<LycorisAdapter>,
    /// Same length as `adapters`. Used for save/load key naming and `get` lookup.
    pub names: Vec<String>,
    /// Optional DoRA magnitudes, parallel to `adapters`. Entry is `None` when
    /// `config.dora == false`, OR when the user explicitly opted into the algo
    /// without `w_orig` (impossible: ctor errors). All-`Some` when DoRA is on.
    pub dora_magnitudes: Vec<Option<Tensor>>,
    pub device: Arc<CudaDevice>,
    /// `name -> index` cache. Built lazily on first `get` call.
    name_index: parking_lot::Mutex<Option<HashMap<String, usize>>>,
}

impl LycorisBundle {
    /// Build an empty bundle. Caller pushes one adapter per LoRA target via
    /// [`add_linear`] / [`add_conv2d`].
    pub fn new(config: LycorisBundleConfig, device: Arc<CudaDevice>) -> Self {
        Self {
            config,
            adapters: Vec::new(),
            names: Vec::new(),
            dora_magnitudes: Vec::new(),
            device,
            name_index: parking_lot::Mutex::new(None),
        }
    }

    /// Construct + push a single Linear adapter (and matching DoRA magnitude
    /// when configured).
    ///
    /// `w_orig` is required when `self.config.dora == true` (used to init
    /// `||W_orig||_2`). For non-DoRA bundles, pass `None` (or the base weight,
    /// either is fine — it's ignored).
    pub fn add_linear(
        &mut self,
        name: &str,
        in_features: usize,
        out_features: usize,
        w_orig: Option<&Tensor>,
    ) -> anyhow::Result<()> {
        if self.config.algo == LycorisAlgo::None {
            bail!("LycorisBundle::add_linear called with algo=None — caller should use legacy LoRALinear instead");
        }

        let alpha = Some(self.config.alpha);
        let dtype = self.config.storage;
        let device = self.device.clone();

        let adapter = match self.config.algo {
            LycorisAlgo::None => unreachable!(),
            LycorisAlgo::LoCon => LycorisAdapter::LoCon(
                LoConModule::new_linear_for_training_with_init(
                    in_features,
                    out_features,
                    self.config.rank,
                    alpha,
                    device,
                    dtype,
                    self.config.init_type,
                )
                .map_err(|e| anyhow!("LoCon::new_linear_for_training({name}): {e}"))?,
            ),
            LycorisAlgo::LoHa => LycorisAdapter::LoHa(
                LoHaModule::new_linear_for_training(
                    in_features,
                    out_features,
                    self.config.rank,
                    alpha,
                    device,
                    dtype,
                )
                .map_err(|e| anyhow!("LoHa::new_linear_for_training({name}): {e}"))?,
            ),
            LycorisAlgo::LoKr => LycorisAdapter::LoKr(
                LoKrModule::new_linear(
                    in_features,
                    out_features,
                    self.config.rank,
                    self.config.alpha,
                    self.config.factor,
                    self.config.decompose_both,
                    self.config.use_scalar,
                    device,
                    dtype,
                )
                .map_err(|e| anyhow!("LoKr::new_linear({name}): {e}"))?,
            ),
            LycorisAlgo::Full => {
                // Full adapter for Linear: weight shape is [out, in] (PyTorch /
                // diffusers convention). bias is not modeled here — Phase 2b
                // can extend `add_linear` to take an optional `bias_size`.
                LycorisAdapter::Full(
                    FullAdapter::new_for_training(
                        Shape::from_dims(&[out_features, in_features]),
                        None,
                        device,
                        dtype,
                    )
                    .map_err(|e| anyhow!("Full::new_for_training({name}): {e}"))?,
                )
            }
            LycorisAlgo::Oft => LycorisAdapter::OFT(
                OFTModule::new_linear(
                    in_features,
                    out_features,
                    self.config.block_size,
                    self.config.alpha,
                    None,
                    dtype,
                    device,
                )
                .map_err(|e| anyhow!("OFT::new_linear({name}): {e}"))?
                .with_neumann_terms(self.config.neumann_terms),
            ),
        };

        // DoRA magnitude.
        let dora = self.build_dora_magnitude_linear(name, in_features, out_features, w_orig)?;

        self.push_entry(name.to_string(), adapter, dora);
        Ok(())
    }

    /// Construct + push a single Conv2d adapter.
    ///
    /// `kernel_size` is `(kh, kw)`. For 1×1 kernels and Tucker disabled the
    /// adapter degenerates to a 1×1-kernel LoRA. For DoRA `w_orig` must be the
    /// base weight in flame's NHWC-aware layout `[kh, kw, ic, oc]` — note the
    /// difference from PyTorch's `[oc, ic, kh, kw]`.
    pub fn add_conv2d(
        &mut self,
        name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        w_orig: Option<&Tensor>,
    ) -> anyhow::Result<()> {
        if self.config.algo == LycorisAlgo::None {
            bail!("LycorisBundle::add_conv2d called with algo=None — caller should use legacy LoRA instead");
        }

        let alpha = Some(self.config.alpha);
        let dtype = self.config.storage;
        let device = self.device.clone();

        let adapter = match self.config.algo {
            LycorisAlgo::None => unreachable!(),
            LycorisAlgo::LoCon => LycorisAdapter::LoCon(
                LoConModule::new_conv2d_for_training_with_init(
                    in_channels,
                    out_channels,
                    kernel_size,
                    self.config.rank,
                    alpha,
                    self.config.use_tucker,
                    device,
                    dtype,
                    self.config.init_type,
                )
                .map_err(|e| anyhow!("LoCon::new_conv2d_for_training({name}): {e}"))?,
            ),
            LycorisAlgo::LoHa => LycorisAdapter::LoHa(
                LoHaModule::new_conv2d_for_training(
                    in_channels,
                    out_channels,
                    kernel_size,
                    self.config.rank,
                    alpha,
                    self.config.use_tucker,
                    device,
                    dtype,
                )
                .map_err(|e| anyhow!("LoHa::new_conv2d_for_training({name}): {e}"))?,
            ),
            LycorisAlgo::LoKr => LycorisAdapter::LoKr(
                LoKrModule::new_conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    self.config.rank,
                    self.config.alpha,
                    self.config.factor,
                    self.config.decompose_both,
                    self.config.use_tucker,
                    self.config.use_scalar,
                    device,
                    dtype,
                )
                .map_err(|e| anyhow!("LoKr::new_conv2d({name}): {e}"))?,
            ),
            LycorisAlgo::Full => {
                // Flame conv layout: [kh, kw, ic, oc].
                let (kh, kw) = kernel_size;
                LycorisAdapter::Full(
                    FullAdapter::new_for_training(
                        Shape::from_dims(&[kh, kw, in_channels, out_channels]),
                        None,
                        device,
                        dtype,
                    )
                    .map_err(|e| anyhow!("Full::new_for_training({name}): {e}"))?,
                )
            }
            LycorisAlgo::Oft => bail!(
                "LycorisBundle::add_conv2d: OFT does not support conv2d (rotates input dim only)"
            ),
        };

        let dora =
            self.build_dora_magnitude_conv2d(name, in_channels, out_channels, kernel_size, w_orig)?;
        self.push_entry(name.to_string(), adapter, dora);
        Ok(())
    }

    /// Common DoRA-magnitude init for linear adapters. `wd_on_out=true` →
    /// magnitude shape `[out, 1]`; `false` → `[1, in]`.
    fn build_dora_magnitude_linear(
        &self,
        name: &str,
        _in_features: usize,
        _out_features: usize,
        w_orig: Option<&Tensor>,
    ) -> anyhow::Result<Option<Tensor>> {
        if !self.config.dora {
            return Ok(None);
        }
        if self.config.algo == LycorisAlgo::Oft {
            bail!("DoRA + OFT is not supported (multiplicative + decomposition conflict): {name}");
        }
        let w = w_orig.ok_or_else(|| {
            anyhow!(
                "DoRA enabled but w_orig is None for adapter '{name}' \
                 — pass the pretrained weight so ||W||_2 can initialize the magnitude"
            )
        })?;

        // DoRA magnitude must be F32 for stable Neumann-style accumulation
        // (per `dora.rs` docstring §5.2).
        let w_f32 = w.to_dtype(DType::F32).map_err(|e| {
            anyhow!("DoRA init_magnitude: cast w_orig to F32 for {name} failed: {e}")
        })?;
        let m = init_magnitude(&w_f32, self.config.dora_wd_on_out, 0.0).map_err(|e| {
            anyhow!("DoRA init_magnitude({name}): {e}")
        })?;
        // Trainable: requires_grad=true so optimizer collects it.
        Ok(Some(m.requires_grad_(true)))
    }

    /// Common DoRA-magnitude init for conv2d. Reshapes the flame NHWC layout
    /// `[kh, kw, ic, oc]` to PyTorch-style `[oc, ic, kh, kw]` before calling
    /// [`init_magnitude`], because the dora helper expects the latter.
    fn build_dora_magnitude_conv2d(
        &self,
        name: &str,
        _in_channels: usize,
        _out_channels: usize,
        _kernel_size: (usize, usize),
        w_orig: Option<&Tensor>,
    ) -> anyhow::Result<Option<Tensor>> {
        if !self.config.dora {
            return Ok(None);
        }
        if self.config.algo == LycorisAlgo::Oft {
            bail!("DoRA + OFT is not supported: {name}");
        }
        let w = w_orig.ok_or_else(|| {
            anyhow!(
                "DoRA enabled but w_orig is None for conv adapter '{name}' \
                 — pass the pretrained weight (flame layout [kh, kw, ic, oc])"
            )
        })?;

        // Flame conv layout is [kh, kw, ic, oc]; init_magnitude expects
        // PyTorch [oc, ic, kh, kw]. Permute (3,2,0,1) before the norm.
        let dims = w.dims();
        if dims.len() != 4 {
            bail!(
                "DoRA conv2d w_orig must be 4D (flame layout [kh, kw, ic, oc]), got rank {} for {name}",
                dims.len()
            );
        }
        let w_pt = w.permute(&[3, 2, 0, 1]).map_err(|e| {
            anyhow!("DoRA: permute w_orig flame→pytorch for {name} failed: {e}")
        })?;
        let w_f32 = w_pt.to_dtype(DType::F32).map_err(|e| {
            anyhow!("DoRA init_magnitude: cast w_orig to F32 for {name} failed: {e}")
        })?;
        let m = init_magnitude(&w_f32, self.config.dora_wd_on_out, 0.0).map_err(|e| {
            anyhow!("DoRA init_magnitude({name}): {e}")
        })?;
        Ok(Some(m.requires_grad_(true)))
    }

    fn push_entry(&mut self, name: String, adapter: LycorisAdapter, dora: Option<Tensor>) {
        self.adapters.push(adapter);
        self.names.push(name);
        self.dora_magnitudes.push(dora);
        // Invalidate the name->index cache.
        *self.name_index.lock() = None;
    }

    /// Flat list of leaf tensors for autograd lookup (adapter parameters +
    /// DoRA magnitudes, in `adapter[0].params... adapter[0].dora_scale,
    /// adapter[1].params... adapter[1].dora_scale, ...` order).
    ///
    /// Each `Tensor` is a clone of the live `Parameter` storage with the
    /// same `TensorId`. Use [`to_parameters`](Self::to_parameters) when the
    /// caller needs to apply optimizer updates — the `Tensor` clones here
    /// are read-only views.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut out: Vec<Tensor> = Vec::with_capacity(self.adapters.len() * 4);
        for (i, a) in self.adapters.iter().enumerate() {
            out.extend(a.parameters());
            if let Some(ref m) = self.dora_magnitudes[i] {
                out.push(m.clone());
            }
        }
        out
    }

    /// Live `Parameter` handles for the optimizer. The handles share the
    /// same `Arc<Mutex<Tensor>>` storage that the algorithm's forward path
    /// reads, so optimizer mutations are visible without any sync step.
    /// This is the path that fixes the gradient-isolation bug — wrapping
    /// `parameters().into_iter().map(Parameter::new)` would create FRESH
    /// `Arc<Mutex<Tensor>>` wrappers that the adapter can't see.
    pub fn to_parameters(&self) -> Vec<Parameter> {
        let mut out: Vec<Parameter> = Vec::with_capacity(self.adapters.len() * 4);
        for (i, a) in self.adapters.iter().enumerate() {
            out.extend(a.parameters_handles());
            if let Some(ref m) = self.dora_magnitudes[i] {
                // DoRA magnitude is currently a bare `Tensor`. Wrap it
                // (creates a fresh handle, matching prior behavior). If
                // DoRA gets in-place updates wired up, switch the field
                // type to `Parameter` and clone the handle here.
                out.push(Parameter::new(m.clone()));
            }
        }
        out
    }

    /// O(1) lookup by adapter name. Builds the index lazily on first call.
    pub fn get(&self, name: &str) -> Option<&LycorisAdapter> {
        let idx = {
            let mut cache = self.name_index.lock();
            if cache.is_none() {
                let map: HashMap<String, usize> = self
                    .names
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), i))
                    .collect();
                *cache = Some(map);
            }
            cache.as_ref().unwrap().get(name).copied()
        };
        idx.map(|i| &self.adapters[i])
    }

    /// Same as [`get`] but also returns the DoRA magnitude (when present)
    /// so the trainer's forward can apply DoRA inline. Returns `None` if the
    /// name isn't registered, or `(adapter, None)` if DoRA is off for this entry.
    pub fn get_with_dora(&self, name: &str) -> Option<(&LycorisAdapter, Option<&Tensor>)> {
        let idx = {
            let mut cache = self.name_index.lock();
            if cache.is_none() {
                let map: HashMap<String, usize> = self
                    .names
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), i))
                    .collect();
                *cache = Some(map);
            }
            cache.as_ref().unwrap().get(name).copied()
        }?;
        Some((&self.adapters[idx], self.dora_magnitudes[idx].as_ref()))
    }

    /// Bundle metadata: `(num_adapters, total_param_element_count, est_vram_mb)`.
    /// `est_vram_mb` sums each leaf's `numel * dtype.size_in_bytes()` (DoRA
    /// magnitudes included).
    pub fn stats(&self) -> (usize, usize, f64) {
        let leaves = self.parameters();
        let mut total_elems: usize = 0;
        let mut total_bytes: usize = 0;
        for t in &leaves {
            let elems = t.shape().elem_count();
            total_elems += elems;
            total_bytes += elems * t.dtype().size_in_bytes();
        }
        let mb = total_bytes as f64 / (1024.0 * 1024.0);
        (self.adapters.len(), total_elems, mb)
    }

    /// Save adapter tensors as PEFT-style safetensors. See module docstring
    /// for the suffix table. `prefix` is prepended to every key with a
    /// leading `.` separator (pass `""` to omit).
    pub fn save_safetensors(&self, prefix: &str, path: &Path) -> anyhow::Result<()> {
        let mut out: HashMap<String, Tensor> = HashMap::new();

        for (i, adapter) in self.adapters.iter().enumerate() {
            let name = &self.names[i];
            let full = qualify(prefix, name);
            self.collect_adapter_tensors(adapter, &full, &mut out)
                .with_context(|| format!("collect tensors for adapter '{name}'"))?;

            // DoRA magnitude.
            if let Some(ref m) = self.dora_magnitudes[i] {
                out.insert(format!("{full}.dora_scale"), m.clone());
            }
        }

        flame_core::serialization::save_file(&out, path)
            .map_err(|e| anyhow!("save_safetensors: {e}"))?;
        Ok(())
    }

    /// Load adapter tensors. Currently a stub that validates algo/key compatibility;
    /// in-place population of constructed adapters is per-trainer Phase 2b work.
    ///
    /// Why a stub: each algo holds its tensors in private fields with init policies
    /// that differ from on-disk storage (e.g. LoCon's `down`/`up` are F32 in-memory
    /// but BF16 on disk). A complete loader needs to dispatch into algo-specific
    /// setters that don't currently exist on the upstream structs. That work lands
    /// alongside the trainer's checkpoint-resume support.
    ///
    /// Bails on:
    /// - missing file
    /// - any key whose algo-suffix conflicts with `self.config.algo`
    /// - DoRA scale present in file but `config.dora == false` (or vice versa)
    pub fn load_safetensors(&mut self, prefix: &str, path: &Path) -> anyhow::Result<()> {
        let source = flame_core::serialization::load_file(path, &self.device)
            .map_err(|e| anyhow!("load_safetensors open '{}': {e}", path.display()))?;

        let detected = detect_algo_from_keys(&source, prefix)?;
        if detected.0 != self.config.algo {
            bail!(
                "load_safetensors: file algo '{}' does not match bundle algo '{}'",
                detected.0.as_str(),
                self.config.algo.as_str()
            );
        }
        if detected.1 != self.config.dora {
            bail!(
                "load_safetensors: file DoRA={} does not match bundle DoRA={}",
                detected.1,
                self.config.dora
            );
        }

        // Phase 2b will wire in-place tensor population per algo. For now,
        // we surface the validation result so trainers can trust the file
        // matches the bundle's config before they invoke their own loader.
        log::warn!(
            "LycorisBundle::load_safetensors: validation-only stub \
             (algo={}, dora={}, num_adapters_in_file≈{}); \
             per-trainer in-place loader is Phase 2b work",
            detected.0.as_str(),
            detected.1,
            detected.2
        );
        Ok(())
    }

    fn collect_adapter_tensors(
        &self,
        adapter: &LycorisAdapter,
        full_prefix: &str,
        out: &mut HashMap<String, Tensor>,
    ) -> anyhow::Result<()> {
        // Helper: pull the live tensor out of a `Parameter`. The clone
        // preserves `TensorId` so it stays paired with autograd state.
        let pt = |p: &flame_core::parameter::Parameter| -> anyhow::Result<Tensor> {
            p.tensor()
                .map_err(|e| anyhow::anyhow!("collect_adapter_tensors: parameter mutex poisoned: {e}"))
        };
        match adapter {
            LycorisAdapter::LoCon(m) => {
                out.insert(format!("{full_prefix}.lora_A.weight"), pt(&m.down)?);
                out.insert(format!("{full_prefix}.lora_B.weight"), pt(&m.up)?);
                if let Some(ref mid) = m.mid {
                    out.insert(format!("{full_prefix}.lora_mid.weight"), pt(mid)?);
                }
            }
            LycorisAdapter::LoHa(m) => {
                out.insert(format!("{full_prefix}.hada_w1_a.weight"), pt(&m.w1a)?);
                out.insert(format!("{full_prefix}.hada_w1_b.weight"), pt(&m.w1b)?);
                out.insert(format!("{full_prefix}.hada_w2_a.weight"), pt(&m.w2a)?);
                out.insert(format!("{full_prefix}.hada_w2_b.weight"), pt(&m.w2b)?);
                if let Some(ref t) = m.t1 {
                    out.insert(format!("{full_prefix}.hada_t1.weight"), pt(t)?);
                }
                if let Some(ref t) = m.t2 {
                    out.insert(format!("{full_prefix}.hada_t2.weight"), pt(t)?);
                }
            }
            LycorisAdapter::LoKr(m) => {
                if let Some(ref w) = m.w1 {
                    out.insert(format!("{full_prefix}.lokr_w1.weight"), pt(w)?);
                }
                if let Some(ref w) = m.w1a {
                    out.insert(format!("{full_prefix}.lokr_w1_a.weight"), pt(w)?);
                }
                if let Some(ref w) = m.w1b {
                    out.insert(format!("{full_prefix}.lokr_w1_b.weight"), pt(w)?);
                }
                if let Some(ref w) = m.w2 {
                    out.insert(format!("{full_prefix}.lokr_w2.weight"), pt(w)?);
                }
                if let Some(ref w) = m.w2a {
                    out.insert(format!("{full_prefix}.lokr_w2_a.weight"), pt(w)?);
                }
                if let Some(ref w) = m.w2b {
                    out.insert(format!("{full_prefix}.lokr_w2_b.weight"), pt(w)?);
                }
                if let Some(ref t) = m.t2 {
                    out.insert(format!("{full_prefix}.lokr_t2.weight"), pt(t)?);
                }
            }
            LycorisAdapter::Full(m) => {
                out.insert(format!("{full_prefix}.diff.weight"), pt(&m.diff)?);
                if let Some(ref b) = m.diff_b {
                    out.insert(format!("{full_prefix}.diff_b"), pt(b)?);
                }
            }
            LycorisAdapter::OFT(m) => {
                out.insert(format!("{full_prefix}.oft_blocks.weight"), pt(&m.blocks)?);
            }
            LycorisAdapter::BOFT(m) => {
                // Same `oft_blocks` suffix as upstream `boft.py` weight_list.
                // The 4D shape `[boft_m, num_blocks, b, b]` is what
                // disambiguates BOFT from OFT (3D `[num_blocks, b, b]`) at
                // load time — see `BOFTModule::algo_check` upstream.
                out.insert(format!("{full_prefix}.oft_blocks.weight"), pt(&m.blocks)?);
            }
        }
        Ok(())
    }

    /// Per-step forward helper for training: looks up the adapter by name and
    /// returns its scaled delta output (`alpha/rank · A·B(x)` for LoCon, etc.).
    /// Returns `Ok(None)` when no adapter is registered for `name` — caller
    /// then leaves the base projection untouched, preserving the byte-identity
    /// of the legacy LoRA path when this bundle is empty.
    ///
    /// Dtype handling: trainer storage is F32 (AdamW state) but model inputs
    /// arrive as BF16. The upstream `LycorisModule::forward` does no casting,
    /// so this wrapper casts the input to the adapter's storage dtype before
    /// the call and casts the output back to the input dtype before returning.
    /// This mirrors the cast pattern in `LoRALinear::forward_delta`.
    ///
    /// Last-dim contract: linear adapters expect input shape `[*, in_features]`
    /// and produce `[*, out_features]`. Z-Image's training forward feeds
    /// `[B, seq, dim]`; LoCon::forward dispatches via matmul on the last dim,
    /// preserving leading batch / sequence axes.
    pub fn forward_delta(
        &self,
        name: &str,
        input: &Tensor,
    ) -> anyhow::Result<Option<Tensor>> {
        let Some(adapter) = self.get(name) else {
            return Ok(None);
        };
        let storage_dt = self.config.storage.to_dtype();
        let input_dt = input.dtype();
        let cast_in = if input_dt != storage_dt {
            input.to_dtype(storage_dt).map_err(|e| {
                anyhow!("forward_delta({name}): cast input {:?}→{:?}: {e}", input_dt, storage_dt)
            })?
        } else {
            input.clone()
        };
        let out_storage = match adapter {
            LycorisAdapter::LoCon(m) => m.forward(&cast_in),
            LycorisAdapter::LoHa(m) => m.forward(&cast_in),
            LycorisAdapter::LoKr(m) => m.forward(&cast_in),
            // FullAdapter has no residual forward — pure weight delta.
            // Trainer should fuse `delta_weight()` into the base weight,
            // not call this path. Bail loudly.
            LycorisAdapter::Full(_) => {
                bail!(
                    "forward_delta({name}): Full adapter has no residual forward — \
                     merge `delta_weight()` into the base weight instead"
                );
            }
            // OFT is multiplicative — `apply_to_input` rotates the input
            // directly. For a residual-additive call site we return
            // `R·x − x` so caller can `base = base + delta`.
            LycorisAdapter::OFT(m) => m
                .apply_to_input(&cast_in)
                .and_then(|rx| rx.sub(&cast_in).map_err(lycoris_rs::Error::Flame)),
            // BOFT is the butterfly extension of OFT — same input-rotation
            // semantics. Return `R(x) − x` so additive call sites compose
            // correctly when the base linear is identity-like.
            LycorisAdapter::BOFT(m) => m
                .apply_to_input(&cast_in)
                .and_then(|rx| rx.sub(&cast_in).map_err(lycoris_rs::Error::Flame)),
        }
        .map_err(|e| anyhow!("forward_delta({name}): {e}"))?;
        let out = if out_storage.dtype() != input_dt {
            out_storage.to_dtype(input_dt).map_err(|e| {
                anyhow!(
                    "forward_delta({name}): cast output {:?}→{:?}: {e}",
                    out_storage.dtype(),
                    input_dt
                )
            })?
        } else {
            out_storage
        };
        Ok(Some(out))
    }

    /// Apply DoRA to a freshly-reconstructed weight `wp = W_orig + ΔW`.
    /// Convenience wrapper around [`apply_weight_decompose`] that pulls the
    /// per-adapter magnitude tensor from the bundle.
    ///
    /// Returns `Ok(None)` when DoRA is off for this name (so callers can use
    /// `.unwrap_or(wp)`-style wiring).
    pub fn apply_dora(
        &self,
        name: &str,
        wp: &Tensor,
    ) -> anyhow::Result<Option<Tensor>> {
        let (_, mag) = self.get_with_dora(name).ok_or_else(|| {
            anyhow!("apply_dora: no adapter registered for '{name}'")
        })?;
        let Some(m) = mag else { return Ok(None) };
        let result =
            apply_weight_decompose(wp, m, self.config.dora_wd_on_out, self.config.dora_eps)
                .map_err(|e| anyhow!("apply_weight_decompose({name}): {e}"))?;
        Ok(Some(result))
    }
}

/// Streaming `init_lokr_norm` for trainers whose base weights are not all
/// resident in a single map (BlockOffloader / sharded streaming trainers:
/// qwenimage, zimage, anima, sd35, ernie, acestep, wan22).
///
/// Walks the bundle's adapters, calls `lookup(adapter_name)` to fetch each
/// LoKr adapter's base weight on demand, and applies the perturbed-normal
/// init via `AdapterModule::init_perturbed_normal_lokr`.
///
/// The trainer provides `lookup` — a closure that maps an adapter's dotted
/// path to its base weight tensor. The closure can fetch the weight from a
/// memory-mapped safetensors file via [`flame_core::serialization::load_file_filtered`]
/// (paged-in, dropped after init) or from the trainer's resident-shared map
/// (head/embeds/etc.) without retaining the streamed block weights.
///
/// Returns the number of adapters skipped (non-LoKr, factored, or
/// `lookup` returned `None`).
///
/// # Example
///
/// ```rust,ignore
/// // Inside `train_qwenimage::main`, after model load:
/// let ckpt_path = args.transformer.clone();
/// let device_for_lookup = device.clone();
/// let skipped = streaming_init_lokr(
///     &model.bundle.adapters,
///     &model.bundle.names,
///     args.init_lokr_norm,
///     |name| {
///         // `name` is e.g. "transformer_blocks.0.attn.to_q";
///         // map to the safetensors key the checkpoint uses.
///         let key = adapter_name_to_ckpt_key(name);
///         let map = flame_core::serialization::load_file_filtered(
///             &ckpt_path,
///             &device_for_lookup,
///             |k| k == key,
///         )?;
///         Ok(map.into_iter().next().map(|(_, t)| t))
///     },
/// )?;
/// ```
pub fn streaming_init_lokr<F>(
    adapters: &[Box<dyn crate::adapter::AdapterModule>],
    names: &[String],
    scale: f32,
    mut lookup: F,
) -> anyhow::Result<usize>
where
    F: FnMut(&str) -> anyhow::Result<Option<Tensor>>,
{
    if scale <= 0.0 {
        return Ok(adapters.len()); // skip everything when disabled
    }
    if adapters.len() != names.len() {
        bail!(
            "streaming_init_lokr: adapters/names length mismatch ({} vs {})",
            adapters.len(),
            names.len()
        );
    }
    let mut skipped = 0usize;
    for (adapter, name) in adapters.iter().zip(names.iter()) {
        if adapter.kind() != "lokr" {
            skipped += 1;
            continue;
        }
        match lookup(name)? {
            Some(w) => {
                let applied = adapter
                    .init_perturbed_normal_lokr(&w, scale)
                    .map_err(|e| anyhow!("init_perturbed_normal_lokr({name}): {e}"))?;
                if !applied {
                    log::warn!(
                        "streaming_init_lokr: '{name}' is LoKr but factored — \
                         init_perturbed_normal_lokr only supports full-form LoKr"
                    );
                    skipped += 1;
                }
            }
            None => {
                log::warn!(
                    "streaming_init_lokr: no base weight returned by lookup for '{name}' — \
                     skipping"
                );
                skipped += 1;
            }
        }
    }
    Ok(skipped)
}

/// Build `<prefix>.<name>` (or just `<name>` when prefix is empty).
fn qualify(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Inspect the safetensors keys and detect (algo, dora_present, adapter_count_estimate).
/// Returns an error when the file mixes incompatible suffixes.
fn detect_algo_from_keys(
    source: &HashMap<String, Tensor>,
    prefix: &str,
) -> anyhow::Result<(LycorisAlgo, bool, usize)> {
    let mut counts = [0usize; 5]; // [LoCon, LoHa, LoKr, Full, OFT]
    let mut dora_present = false;
    let mut adapter_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let strip = |k: &str| -> Option<String> {
        if prefix.is_empty() {
            Some(k.to_string())
        } else if let Some(rest) = k.strip_prefix(prefix) {
            rest.strip_prefix('.').map(str::to_string)
        } else {
            None
        }
    };

    for k in source.keys() {
        let Some(rest) = strip(k) else { continue };
        if rest.ends_with(".dora_scale") || rest.ends_with(".magnitude_vector") {
            dora_present = true;
            adapter_names.insert(rest.rsplit_once('.').map(|(p, _)| p.to_string()).unwrap_or(rest));
            continue;
        }
        // Remove trailing `.weight` if present, then test suffix.
        let core = rest.strip_suffix(".weight").unwrap_or(&rest);
        let (head, suffix) = match core.rsplit_once('.') {
            Some((h, s)) => (h, s),
            None => continue,
        };
        match suffix {
            "lora_A" | "lora_B" | "lora_mid" => counts[0] += 1,
            "hada_w1_a" | "hada_w1_b" | "hada_w2_a" | "hada_w2_b" | "hada_t1" | "hada_t2" => {
                counts[1] += 1
            }
            "lokr_w1" | "lokr_w1_a" | "lokr_w1_b" | "lokr_w2" | "lokr_w2_a" | "lokr_w2_b"
            | "lokr_t2" => counts[2] += 1,
            "diff" | "diff_b" => counts[3] += 1,
            "oft_blocks" => counts[4] += 1,
            "alpha" => continue, // companion scalar; not algo-determining
            _ => continue,
        }
        adapter_names.insert(head.to_string());
    }

    let nz: Vec<_> = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .collect();
    if nz.is_empty() {
        bail!("detect_algo_from_keys: no recognized LyCORIS suffixes under prefix '{prefix}'");
    }
    if nz.len() > 1 {
        bail!(
            "detect_algo_from_keys: file mixes multiple algo suffixes ({} kinds detected); \
             a LycorisBundle file must contain exactly one algo",
            nz.len()
        );
    }
    let algo = match nz[0].0 {
        0 => LycorisAlgo::LoCon,
        1 => LycorisAlgo::LoHa,
        2 => LycorisAlgo::LoKr,
        3 => LycorisAlgo::Full,
        4 => LycorisAlgo::Oft,
        _ => unreachable!(),
    };
    Ok((algo, dora_present, adapter_names.len()))
}

// --------------------------------------------------------------------------
// CLI helper
// --------------------------------------------------------------------------

/// Clap-friendly subset of the LyCORIS knobs. Each EDv2 trainer should
/// `#[command(flatten)] lycoris: LycorisCliArgs` into its `Args` struct, then
/// call `LycorisBundleConfig::from_cli(&args.lycoris)`.
#[derive(Clone, Debug, clap::Args)]
pub struct LycorisCliArgs {
    /// LyCORIS algo: none|lora|locon|loha|lokr|full|oft. Default `none` →
    /// legacy LoRA path (LoRALinear), no LyCORIS adapter constructed.
    #[arg(long, default_value = "none")]
    pub lycoris_algo: String,

    #[arg(long, default_value_t = 16)]
    pub lycoris_rank: usize,

    #[arg(long, default_value_t = 16.0)]
    pub lycoris_alpha: f32,

    #[arg(long, default_value_t = 16)]
    pub lycoris_factor: i32,

    #[arg(long, default_value_t = 32)]
    pub lycoris_block_size: usize,

    #[arg(long, default_value_t = 5)]
    pub lycoris_neumann_terms: usize,

    #[arg(long, default_value_t = false)]
    pub lycoris_tucker: bool,

    #[arg(long, default_value_t = false)]
    pub lycoris_decompose_both: bool,

    #[arg(long, default_value_t = false)]
    pub lycoris_use_scalar: bool,

    /// Enable DoRA (weight-decomposed LoRA). Applies to LoCon/LoHa/LoKr/Full only.
    #[arg(long, default_value_t = false)]
    pub lycoris_decompose: bool,

    /// DoRA magnitude axis. `true` (default) = lycoris-upstream
    /// (norm over input dims, magnitude `[out, 1]`).
    #[arg(long, default_value_t = true)]
    pub lycoris_dora_wd_on_out: bool,

    #[arg(long, default_value_t = 1e-6)]
    pub lycoris_dora_eps: f32,

    /// PEFT / SimpleTuner `--lora_init_type`. Applies to the LoCon (LoRA)
    /// path only. Choices: `default | gaussian | pissa | olora | loftq`.
    /// `pissa`/`olora`/`loftq` parse but error at adapter construction
    /// because flame-core does not yet expose SVD/QR.
    #[arg(long, default_value = "default")]
    pub lora_init_type: String,

    /// SimpleTuner-style `lycoris_config preset.json` — per-target
    /// `module_algo_map` overrides. See [`LycorisConfigFile`] for the
    /// schema. Top-level fields overlay the bundle defaults; per-target
    /// overrides apply at adapter construction time. Default unset → no
    /// preset (uniform top-level config).
    #[arg(long)]
    pub lycoris_config: Option<std::path::PathBuf>,
}

impl Default for LycorisCliArgs {
    fn default() -> Self {
        Self {
            lycoris_algo: "none".to_string(),
            lycoris_rank: 16,
            lycoris_alpha: 16.0,
            lycoris_factor: 16,
            lycoris_block_size: 32,
            lycoris_neumann_terms: 5,
            lycoris_tucker: false,
            lycoris_decompose_both: false,
            lycoris_use_scalar: false,
            lycoris_decompose: false,
            lycoris_dora_wd_on_out: true,
            lycoris_dora_eps: 1e-6,
            lora_init_type: "default".to_string(),
            lycoris_config: None,
        }
    }
}

impl LycorisBundleConfig {
    /// Build a [`LycorisBundleConfig`] from the parsed CLI args.
    /// Storage is hard-wired to F32 (training default — AdamW state needs F32).
    /// Inference / merge-only callers should construct the config directly.
    pub fn from_cli(args: &LycorisCliArgs) -> anyhow::Result<Self> {
        let algo = LycorisAlgo::parse(&args.lycoris_algo)?;
        let init_type = LoraInitType::parse(&args.lora_init_type)
            .map_err(|e| anyhow!("--lora_init_type: {e}"))?;
        let base = Self {
            algo,
            rank: args.lycoris_rank,
            alpha: args.lycoris_alpha,
            factor: args.lycoris_factor,
            block_size: args.lycoris_block_size,
            neumann_terms: args.lycoris_neumann_terms,
            use_tucker: args.lycoris_tucker,
            conv_rank: 0,
            conv_alpha: 0.0,
            decompose_both: args.lycoris_decompose_both,
            use_scalar: args.lycoris_use_scalar,
            dora: args.lycoris_decompose,
            dora_wd_on_out: args.lycoris_dora_wd_on_out,
            dora_eps: args.lycoris_dora_eps,
            storage: StorageDtype::F32,
            init_type,
            preset: None,
        };
        base.with_optional_lycoris_config_file(args.lycoris_config.as_deref())
    }

    /// Load and overlay a SimpleTuner-style `lycoris_config.json` onto this
    /// config (in place). Top-level fields in the JSON override the
    /// supplied defaults; the parsed `apply_preset` is stored in
    /// `self.preset` for per-target dispatch in the [`AdapterStore`].
    ///
    /// `path = None` is a no-op (idiomatic for trainers that pass
    /// `Option<&Path>` straight from CLI).
    pub fn with_optional_lycoris_config_file<P: AsRef<Path>>(
        mut self,
        path: Option<P>,
    ) -> anyhow::Result<Self> {
        let Some(p) = path else { return Ok(self) };
        let file = LycorisConfigFile::from_path(p.as_ref())?;
        let (cfg, preset) = file.apply_to(self)?;
        self = cfg;
        self.preset = preset;
        Ok(self)
    }
}

// ===========================================================================
// AdapterStore — kohya/SimpleTuner-style heterogeneous adapter container
// ===========================================================================
//
// This is the trait-object-based replacement for `LycorisBundle`. Where
// `LycorisBundle` keeps its adapters as `Vec<LycorisAdapter>` (an enum with
// per-algo arms), `AdapterStore` keeps them as
// `Vec<Box<dyn AdapterModule>>` so per-call-site model code is identical
// regardless of algo:
//
// ```rust,ignore
// let delta = self.adapters[idx].forward_delta(input)?;
// ```
//
// A model holding `Vec<Box<dyn AdapterModule>>` works for both `LoRALinear`
// (legacy) and `LycorisLinear` (new) without per-call-site match arms.
//
// `LycorisBundle` stays in this file for back-compat with `train_klein.rs`
// (Phase 2a wiring), but new trainers and Phase 2b refactors should target
// `AdapterStore`.

use crate::adapter::{AdapterModule, LycorisLinear};
use crate::lora::LoRALinear;

/// Owning collection of trainable adapters for one trainer. Each entry is
/// a per-target wrapper (`LoRALinear` for `LycorisAlgo::None`,
/// `LycorisLinear` otherwise).
///
/// Kohya / SimpleTuner pattern: model-side per-call-site code is identical
/// regardless of algo. See module-level doc and
/// [`crate::adapter::AdapterModule`].
pub struct AdapterStore {
    pub adapters: Vec<Box<dyn AdapterModule>>,
    pub names: Vec<String>,
    pub config: LycorisBundleConfig,
    pub device: Arc<CudaDevice>,
    /// `name -> index` cache. Built lazily on first `get` call.
    name_index: parking_lot::Mutex<Option<HashMap<String, usize>>>,
}

impl AdapterStore {
    /// Build an empty store. Caller pushes one adapter per LoRA target via
    /// [`build_and_push_linear`](Self::build_and_push_linear) /
    /// [`build_and_push_conv2d`](Self::build_and_push_conv2d).
    pub fn new(config: LycorisBundleConfig, device: Arc<CudaDevice>) -> Self {
        Self {
            adapters: Vec::new(),
            names: Vec::new(),
            config,
            device,
            name_index: parking_lot::Mutex::new(None),
        }
    }

    /// Resolve the effective per-target [`LycorisBundleConfig`] for adapter
    /// `name`, applying any preset filter and `module_algo_map` overrides.
    ///
    /// Returns `Ok(Some(cfg))` when the adapter should be built (with `cfg`
    /// reflecting any per-target overrides), `Ok(None)` when the preset's
    /// `target_module` filter excludes this name. Returns `Err` only on a
    /// malformed override (e.g. unknown algo string).
    fn resolve_effective_config(
        &self,
        name: &str,
    ) -> anyhow::Result<Option<LycorisBundleConfig>> {
        let Some(preset) = self.config.preset.as_ref() else {
            return Ok(Some(self.config.clone()));
        };
        if !preset.is_allowed(name) {
            return Ok(None);
        }
        let mut cfg = self.config.clone();
        if let Some(ov) = preset.resolve(name) {
            if let Some(ref a) = ov.algo {
                cfg.algo = LycorisAlgo::parse(a)
                    .map_err(|e| anyhow!("preset override algo for '{name}': {e}"))?;
            }
            if let Some(d) = ov.linear_dim {
                cfg.rank = d;
            }
            if let Some(a) = ov.linear_alpha {
                cfg.alpha = a;
            }
            if let Some(f) = ov.factor {
                cfg.factor = f;
            }
            if let Some(d) = ov.decompose_both {
                cfg.decompose_both = d;
            }
            if let Some(t) = ov.use_tucker {
                cfg.use_tucker = t;
            }
            if let Some(d) = ov.dora {
                cfg.dora = d;
            }
            if let Some(ref t) = ov.lora_init_type {
                cfg.init_type = LoraInitType::parse(t)
                    .map_err(|e| anyhow!("preset override lora_init_type for '{name}': {e}"))?;
            }
        }
        // Drop preset on the per-call config so the inner ctor doesn't recurse.
        cfg.preset = None;
        Ok(Some(cfg))
    }

    /// Build the right adapter for the configured algo and push it.
    ///
    /// `config.algo == None` → constructs a [`LoRALinear`] (legacy plain
    /// LoRA). Else → constructs the matching `lycoris_rs` module via the
    /// `*_for_training` ctors and wraps it in [`LycorisLinear`]. When
    /// `config.dora == true`, the per-target DoRA magnitude is initialized
    /// from `w_orig` (which must be `Some`) and stored on the wrapper.
    ///
    /// `seed` is forwarded to [`LoRALinear::new`] for the LoRA path. The
    /// LyCORIS path uses its internal RNG (kaiming/normal init).
    pub fn build_and_push_linear(
        &mut self,
        name: &str,
        in_features: usize,
        out_features: usize,
        w_orig: Option<&Tensor>,
    ) -> anyhow::Result<()> {
        // SimpleTuner-style per-target preset resolution. When `preset` is
        // set, find the first `module_algo_map` entry whose pattern matches
        // `name` (case-insensitive substring) and overlay its overrides on
        // the bundle config for this single adapter. `target_module` is
        // applied as a filter when present: a name that doesn't match any
        // `target_module` entry skips with `Ok(())` (caller's outer loop
        // moves on, no adapter created — this matches SimpleTuner's
        // `apply_preset` semantics).
        let effective_config = self.resolve_effective_config(name)?;
        if effective_config.is_none() {
            log::debug!(
                "AdapterStore: name='{name}' filtered out by lycoris_config target_module"
            );
            return Ok(());
        }
        let effective_config = effective_config.unwrap();
        let saved_config = std::mem::replace(&mut self.config, effective_config);
        let result = self.build_and_push_linear_inner(name, in_features, out_features, w_orig);
        self.config = saved_config;
        return result;
    }

    fn build_and_push_linear_inner(
        &mut self,
        name: &str,
        in_features: usize,
        out_features: usize,
        w_orig: Option<&Tensor>,
    ) -> anyhow::Result<()> {
        let boxed: Box<dyn AdapterModule> = match self.config.algo {
            LycorisAlgo::None => {
                // Seed: derive deterministically from adapter index so two
                // pushes don't collide. Trainers that need full control
                // should construct `LoRALinear` themselves and call
                // `push_lora_linear`.
                let seed = 1729u64.wrapping_add(self.adapters.len() as u64 * 2654435761);
                let lora = LoRALinear::new(
                    in_features,
                    out_features,
                    self.config.rank,
                    self.config.alpha,
                    self.device.clone(),
                    seed,
                )
                .map_err(|e| anyhow!("LoRALinear::new({name}): {e}"))?;
                Box::new(lora)
            }
            _ => {
                let alpha = Some(self.config.alpha);
                let dtype = self.config.storage;
                let device = self.device.clone();

                let adapter = match self.config.algo {
                    LycorisAlgo::None => unreachable!(),
                    LycorisAlgo::LoCon => LycorisAdapter::LoCon(
                        LoConModule::new_linear_for_training_with_init(
                            in_features,
                            out_features,
                            self.config.rank,
                            alpha,
                            device,
                            dtype,
                            self.config.init_type,
                        )
                        .map_err(|e| anyhow!("LoCon::new_linear_for_training({name}): {e}"))?,
                    ),
                    LycorisAlgo::LoHa => LycorisAdapter::LoHa(
                        LoHaModule::new_linear_for_training(
                            in_features, out_features, self.config.rank, alpha, device, dtype,
                        )
                        .map_err(|e| anyhow!("LoHa::new_linear_for_training({name}): {e}"))?,
                    ),
                    LycorisAlgo::LoKr => LycorisAdapter::LoKr(
                        LoKrModule::new_linear(
                            in_features,
                            out_features,
                            self.config.rank,
                            self.config.alpha,
                            self.config.factor,
                            self.config.decompose_both,
                            self.config.use_scalar,
                            device,
                            dtype,
                        )
                        .map_err(|e| anyhow!("LoKr::new_linear({name}): {e}"))?,
                    ),
                    LycorisAlgo::Full => LycorisAdapter::Full(
                        FullAdapter::new_for_training(
                            Shape::from_dims(&[out_features, in_features]),
                            None,
                            device,
                            dtype,
                        )
                        .map_err(|e| anyhow!("Full::new_for_training({name}): {e}"))?,
                    ),
                    LycorisAlgo::Oft => LycorisAdapter::OFT(
                        OFTModule::new_linear(
                            in_features,
                            out_features,
                            self.config.block_size,
                            self.config.alpha,
                            None,
                            dtype,
                            device,
                        )
                        .map_err(|e| anyhow!("OFT::new_linear({name}): {e}"))?
                        .with_neumann_terms(self.config.neumann_terms),
                    ),
                };

                let dora = build_dora_magnitude_linear(&self.config, name, w_orig)?;
                let wrapper = LycorisLinear::new(
                    adapter,
                    dora,
                    self.config.dora_wd_on_out,
                    self.config.dora_eps,
                    self.config.storage,
                );
                Box::new(wrapper)
            }
        };

        self.push_entry(name.to_string(), boxed);
        Ok(())
    }

    /// Build the right Conv2d adapter for the configured algo and push it.
    ///
    /// LoRA (`algo=None`) Conv2d isn't supported here — `LoRALinear` is
    /// linear-only. Bails with a clear error in that case so callers know
    /// to either disable Conv2d targeting for plain LoRA or pick a LoCon /
    /// LoHa / LoKr / Full variant.
    pub fn build_and_push_conv2d(
        &mut self,
        name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        w_orig: Option<&Tensor>,
    ) -> anyhow::Result<()> {
        // Same per-target preset overlay as `build_and_push_linear`.
        let effective_config = self.resolve_effective_config(name)?;
        if effective_config.is_none() {
            log::debug!(
                "AdapterStore: name='{name}' filtered out by lycoris_config target_module"
            );
            return Ok(());
        }
        let effective_config = effective_config.unwrap();
        let saved_config = std::mem::replace(&mut self.config, effective_config);
        let result = self.build_and_push_conv2d_inner(
            name,
            in_channels,
            out_channels,
            kernel_size,
            w_orig,
        );
        self.config = saved_config;
        return result;
    }

    fn build_and_push_conv2d_inner(
        &mut self,
        name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        w_orig: Option<&Tensor>,
    ) -> anyhow::Result<()> {
        if self.config.algo == LycorisAlgo::None {
            bail!(
                "AdapterStore::build_and_push_conv2d: plain-LoRA (algo=None) does not support \
                 Conv2d targets — pick locon/loha/lokr/full instead, or skip conv layers"
            );
        }
        if self.config.algo == LycorisAlgo::Oft {
            bail!(
                "AdapterStore::build_and_push_conv2d: OFT does not support conv2d \
                 (rotates input dim only)"
            );
        }

        let alpha = Some(self.config.alpha);
        let dtype = self.config.storage;
        let device = self.device.clone();

        let adapter = match self.config.algo {
            LycorisAlgo::LoCon => LycorisAdapter::LoCon(
                LoConModule::new_conv2d_for_training_with_init(
                    in_channels,
                    out_channels,
                    kernel_size,
                    self.config.rank,
                    alpha,
                    self.config.use_tucker,
                    device,
                    dtype,
                    self.config.init_type,
                )
                .map_err(|e| anyhow!("LoCon::new_conv2d_for_training({name}): {e}"))?,
            ),
            LycorisAlgo::LoHa => LycorisAdapter::LoHa(
                LoHaModule::new_conv2d_for_training(
                    in_channels,
                    out_channels,
                    kernel_size,
                    self.config.rank,
                    alpha,
                    self.config.use_tucker,
                    device,
                    dtype,
                )
                .map_err(|e| anyhow!("LoHa::new_conv2d_for_training({name}): {e}"))?,
            ),
            LycorisAlgo::LoKr => LycorisAdapter::LoKr(
                LoKrModule::new_conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    self.config.rank,
                    self.config.alpha,
                    self.config.factor,
                    self.config.decompose_both,
                    self.config.use_tucker,
                    self.config.use_scalar,
                    device,
                    dtype,
                )
                .map_err(|e| anyhow!("LoKr::new_conv2d({name}): {e}"))?,
            ),
            LycorisAlgo::Full => {
                let (kh, kw) = kernel_size;
                LycorisAdapter::Full(
                    FullAdapter::new_for_training(
                        Shape::from_dims(&[kh, kw, in_channels, out_channels]),
                        None,
                        device,
                        dtype,
                    )
                    .map_err(|e| anyhow!("Full::new_for_training({name}): {e}"))?,
                )
            }
            _ => unreachable!("Conv2d-incompatible algos rejected above"),
        };

        let dora = build_dora_magnitude_conv2d(&self.config, name, w_orig)?;
        let wrapper = LycorisLinear::new(
            adapter,
            dora,
            self.config.dora_wd_on_out,
            self.config.dora_eps,
            self.config.storage,
        );
        self.push_entry(name.to_string(), Box::new(wrapper));
        Ok(())
    }

    /// Push a pre-constructed `LoRALinear` (escape hatch for trainers that
    /// want full control over the per-target seed / shape).
    pub fn push_lora_linear(&mut self, name: &str, lora: LoRALinear) {
        self.push_entry(name.to_string(), Box::new(lora));
    }

    fn push_entry(&mut self, name: String, adapter: Box<dyn AdapterModule>) {
        self.adapters.push(adapter);
        self.names.push(name);
        *self.name_index.lock() = None; // invalidate name-index cache
    }

    /// Flat list of leaf tensors for the optimizer (in `adapter[0].leaves...
    /// adapter[1].leaves...` order).
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut out = Vec::with_capacity(self.adapters.len() * 4);
        for a in &self.adapters {
            out.extend(a.parameters());
        }
        out
    }

    /// Wrap each leaf in `Parameter`. Use this to hand the optimizer a
    /// flat parameter list.
    pub fn to_parameters(&self) -> Vec<Parameter> {
        self.adapters
            .iter()
            .flat_map(|a| a.to_parameters())
            .collect()
    }

    /// O(1) lookup by adapter name. Builds the index lazily on first call.
    pub fn get(&self, name: &str) -> Option<&dyn AdapterModule> {
        let idx = {
            let mut cache = self.name_index.lock();
            if cache.is_none() {
                let map: HashMap<String, usize> = self
                    .names
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), i))
                    .collect();
                *cache = Some(map);
            }
            cache.as_ref().unwrap().get(name).copied()
        };
        idx.map(|i| self.adapters[i].as_ref())
    }

    /// Bundle metadata: `(num_adapters, total_param_element_count, est_vram_mb)`.
    pub fn stats(&self) -> (usize, usize, f64) {
        let leaves = self.parameters();
        let mut total_elems: usize = 0;
        let mut total_bytes: usize = 0;
        for t in &leaves {
            let elems = t.shape().elem_count();
            total_elems += elems;
            total_bytes += elems * t.dtype().size_in_bytes();
        }
        let mb = total_bytes as f64 / (1024.0 * 1024.0);
        (self.adapters.len(), total_elems, mb)
    }

    /// Save adapter tensors as PEFT-style safetensors. Iterates each
    /// adapter's [`AdapterModule::named_tensors`] with `<prefix>.<name>.<suffix>`
    /// keys.
    pub fn save_safetensors(&self, prefix: &str, path: &Path) -> anyhow::Result<()> {
        let mut out: HashMap<String, Tensor> = HashMap::new();

        for (i, adapter) in self.adapters.iter().enumerate() {
            let name = &self.names[i];
            let full = qualify(prefix, name);
            for (suffix, tensor) in adapter.named_tensors() {
                out.insert(format!("{full}.{suffix}"), tensor);
            }
        }

        flame_core::serialization::save_file(&out, path)
            .map_err(|e| anyhow!("save_safetensors: {e}"))?;
        Ok(())
    }

    /// Load adapter tensors. Validates algo/key compatibility against
    /// `self.config.algo` and `self.config.dora`. Per-algo in-place tensor
    /// population is per-trainer Phase 2b work (same constraint as the
    /// legacy `LycorisBundle::load_safetensors`).
    pub fn load_safetensors(&mut self, prefix: &str, path: &Path) -> anyhow::Result<()> {
        let source = flame_core::serialization::load_file(path, &self.device)
            .map_err(|e| anyhow!("load_safetensors open '{}': {e}", path.display()))?;

        // For algo=None the file is plain LoRA; detect_algo_from_keys
        // expects LyCORIS suffixes. Skip the algo check for plain LoRA.
        if self.config.algo != LycorisAlgo::None {
            let detected = detect_algo_from_keys(&source, prefix)?;
            if detected.0 != self.config.algo {
                bail!(
                    "load_safetensors: file algo '{}' does not match store algo '{}'",
                    detected.0.as_str(),
                    self.config.algo.as_str()
                );
            }
            if detected.1 != self.config.dora {
                bail!(
                    "load_safetensors: file DoRA={} does not match store DoRA={}",
                    detected.1,
                    self.config.dora
                );
            }
            log::warn!(
                "AdapterStore::load_safetensors: validation-only stub \
                 (algo={}, dora={}, num_adapters_in_file≈{}); \
                 per-trainer in-place loader is Phase 2b work",
                detected.0.as_str(),
                detected.1,
                detected.2
            );
        }
        Ok(())
    }
}

/// Common DoRA-magnitude init for linear adapters. Extracted out of
/// `LycorisBundle` so `AdapterStore` and the legacy bundle can share the
/// same axis convention + dtype handling.
fn build_dora_magnitude_linear(
    config: &LycorisBundleConfig,
    name: &str,
    w_orig: Option<&Tensor>,
) -> anyhow::Result<Option<Tensor>> {
    if !config.dora {
        return Ok(None);
    }
    if config.algo == LycorisAlgo::Oft {
        bail!("DoRA + OFT is not supported (multiplicative + decomposition conflict): {name}");
    }
    let w = w_orig.ok_or_else(|| {
        anyhow!(
            "DoRA enabled but w_orig is None for adapter '{name}' \
             — pass the pretrained weight so ||W||_2 can initialize the magnitude"
        )
    })?;
    let w_f32 = w
        .to_dtype(DType::F32)
        .map_err(|e| anyhow!("DoRA init_magnitude: cast w_orig to F32 for {name} failed: {e}"))?;
    let m = init_magnitude(&w_f32, config.dora_wd_on_out, 0.0)
        .map_err(|e| anyhow!("DoRA init_magnitude({name}): {e}"))?;
    Ok(Some(m.requires_grad_(true)))
}

/// Conv2d DoRA-magnitude init. Same flame-NHWC → pytorch-NCHW permute as
/// `LycorisBundle::build_dora_magnitude_conv2d`.
fn build_dora_magnitude_conv2d(
    config: &LycorisBundleConfig,
    name: &str,
    w_orig: Option<&Tensor>,
) -> anyhow::Result<Option<Tensor>> {
    if !config.dora {
        return Ok(None);
    }
    if config.algo == LycorisAlgo::Oft {
        bail!("DoRA + OFT is not supported: {name}");
    }
    let w = w_orig.ok_or_else(|| {
        anyhow!(
            "DoRA enabled but w_orig is None for conv adapter '{name}' \
             — pass the pretrained weight (flame layout [kh, kw, ic, oc])"
        )
    })?;
    let dims = w.dims();
    if dims.len() != 4 {
        bail!(
            "DoRA conv2d w_orig must be 4D (flame layout [kh, kw, ic, oc]), got rank {} for {name}",
            dims.len()
        );
    }
    let w_pt = w
        .permute(&[3, 2, 0, 1])
        .map_err(|e| anyhow!("DoRA: permute w_orig flame→pytorch for {name} failed: {e}"))?;
    let w_f32 = w_pt
        .to_dtype(DType::F32)
        .map_err(|e| anyhow!("DoRA init_magnitude: cast w_orig to F32 for {name} failed: {e}"))?;
    let m = init_magnitude(&w_f32, config.dora_wd_on_out, 0.0)
        .map_err(|e| anyhow!("DoRA init_magnitude({name}): {e}"))?;
    Ok(Some(m.requires_grad_(true)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algo_parse_accepts_aliases() {
        assert_eq!(LycorisAlgo::parse("none").unwrap(), LycorisAlgo::None);
        assert_eq!(LycorisAlgo::parse("LoRA").unwrap(), LycorisAlgo::LoCon);
        assert_eq!(LycorisAlgo::parse("locon").unwrap(), LycorisAlgo::LoCon);
        assert_eq!(LycorisAlgo::parse("LoHa").unwrap(), LycorisAlgo::LoHa);
        assert_eq!(LycorisAlgo::parse("lokr").unwrap(), LycorisAlgo::LoKr);
        assert_eq!(LycorisAlgo::parse("Full").unwrap(), LycorisAlgo::Full);
        assert_eq!(LycorisAlgo::parse("oft").unwrap(), LycorisAlgo::Oft);
        assert_eq!(LycorisAlgo::parse("diag-oft").unwrap(), LycorisAlgo::Oft);
        assert!(LycorisAlgo::parse("bogus").is_err());
    }

    #[test]
    fn algo_as_str_roundtrip() {
        for a in [
            LycorisAlgo::None,
            LycorisAlgo::LoCon,
            LycorisAlgo::LoHa,
            LycorisAlgo::LoKr,
            LycorisAlgo::Full,
            LycorisAlgo::Oft,
        ] {
            let s = a.as_str();
            assert_eq!(LycorisAlgo::parse(s).unwrap(), a);
        }
    }

    #[test]
    fn default_config_is_disabled() {
        let c = LycorisBundleConfig::default();
        assert_eq!(c.algo, LycorisAlgo::None);
        assert_eq!(c.storage, StorageDtype::F32);
        assert!(!c.dora);
    }

    #[test]
    fn from_cli_default_is_none() {
        let cli = LycorisCliArgs::default();
        let cfg = LycorisBundleConfig::from_cli(&cli).unwrap();
        assert_eq!(cfg.algo, LycorisAlgo::None);
        assert_eq!(cfg.storage, StorageDtype::F32);
    }

    #[test]
    fn qualify_handles_empty_prefix() {
        assert_eq!(qualify("", "x.y"), "x.y");
        assert_eq!(qualify("transformer", "blocks.0.attn"), "transformer.blocks.0.attn");
    }
}
