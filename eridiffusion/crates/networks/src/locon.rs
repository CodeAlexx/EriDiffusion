//! LoCoN (LoRA for Convolution) — Flame-only adapter scaffold
//!
//! This file was using Candle-era constructs and incomplete code paths.
//! It is now aligned with eridiffusion_core's NetworkAdapter trait and Flame tensors.

use async_trait::async_trait;
use flame_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use eridiffusion_core::{Device, ModelArchitecture, NetworkAdapter, NetworkMetadata, NetworkType, Result};

/// LoCoN configuration (Flame-only)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoConConfig {
    pub rank: usize,
    pub alpha: f32,
    #[serde(default)]
    pub dropout: f32,
    #[serde(default)]
    pub target_modules: Vec<String>,
    #[serde(default)]
    pub conv_rank: Option<usize>,
    #[serde(default)]
    pub use_cp: bool,
}

impl Default for LoConConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            alpha: 1.0,
            dropout: 0.0,
            target_modules: vec![],
            conv_rank: None,
            use_cp: false,
        }
    }
}

/// Minimal LoCoN adapter wired to core traits (Flame only). No Candle.
pub struct LoConAdapter {
    config: LoConConfig,
    device: Device,
    architecture: ModelArchitecture,
    metadata: NetworkMetadata,
}

impl LoConAdapter {
    pub fn new(config: LoConConfig, architecture: ModelArchitecture, device: Device) -> anyhow::Result<Self> {
        let metadata = NetworkMetadata {
            name: "locon_adapter".to_string(),
            network_type: NetworkType::LoCoN,
            version: "1.0.0".to_string(),
            base_model: format!("{:?}", architecture),
            rank: Some(config.rank),
            alpha: Some(config.alpha),
            target_modules: config.target_modules.clone(),
            created_at: chrono::Utc::now(),
            config: HashMap::new(),
        };

        Ok(Self { config, device, architecture, metadata })
    }
}

#[async_trait]
impl NetworkAdapter for LoConAdapter {
    fn adapter_type(&self) -> NetworkType { NetworkType::LoCoN }

    fn metadata(&self) -> &NetworkMetadata { &self.metadata }

    fn target_modules(&self) -> &[String] { &self.config.target_modules }

    fn trainable_parameters(&self) -> Vec<&Tensor> {
        // LoCoN parameters not wired yet (future: conv/linear low-rank factors).
        Vec::new()
    }

    fn parameters(&self) -> HashMap<String, Tensor> {
        // No parameters exposed yet.
        HashMap::new()
    }

    fn apply_to_layer(&self, _layer_name: &str, input: &Tensor) -> anyhow::Result<Tensor> {
        // No-op until LoCoN weights are attached to targets.
        Ok(input.clone())
    }

    fn merge_weights(&mut self, _scale: f32) -> anyhow::Result<()> { Ok(()) }

    async fn save_weights(&self, path: &Path) -> anyhow::Result<()> {
        // Persist metadata only for now to keep IO paths working.
        let metadata_path = path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        tokio::fs::write(&metadata_path, metadata_json).await?;
        Ok(())
    }

    async fn load_weights(&mut self, path: &Path) -> anyhow::Result<()> {
        // Load metadata if present; no tensor weights yet.
        let metadata_path = path.with_extension("json");
        if metadata_path.exists() {
            let metadata_json = tokio::fs::read_to_string(&metadata_path).await?;
            self.metadata = serde_json::from_str(&metadata_json)?;
        }
        Ok(())
    }

    fn memory_usage(&self) -> usize { 0 }

    fn to_device(&mut self, device: &Device) -> anyhow::Result<()> { self.device = device.clone(); Ok(()) }
}

