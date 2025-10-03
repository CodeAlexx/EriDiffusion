use flame_core::device::Device;
use flame_core::GradientMap;
use flame_core::{DType, Shape, Tensor};
// backward is a method on Tensor, not a standalone function
use flame_core::optimizers::{Adam, SGD};
use serde::{Deserialize, Serialize};

/// Memory allocation precision mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PrecisionMode {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Mixed,
}

/// Memory format for tensor storage
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryFormat {
    Contiguous,
    ChannelsFirst,
    ChannelsLast,
    Patchified,
}

/// Model type for memory optimization  
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelType {
    SD15,
    SD21,
    SDXL,
    SD3,
    SD35,
    Flux,
    HunyuanDiT,
    Custom,
}

/// Quantization mode for reduced precision
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum QuantizationMode {
    None,
    INT8,
    INT4,
    NF4,      // NormalFloat4 for QLoRA
    GPTQ,     // GPTQ quantization
    AWQ,      // Activation-aware Weight Quantization
    FP8_E4M3, // FP8 with 4-bit exponent, 3-bit mantissa
    FP8_E5M2, // FP8 with 5-bit exponent, 2-bit mantissa
}

impl QuantizationMode {
    /// Get bytes per element for this quantization mode
    pub fn bytes_per_element(&self) -> f32 {
        match self {
            QuantizationMode::None => 4.0, // FP32
            QuantizationMode::INT8 => 1.0,
            QuantizationMode::INT4 => 0.5,
            QuantizationMode::NF4 => 0.5,
            QuantizationMode::GPTQ => 0.5, // 4-bit typically
            QuantizationMode::AWQ => 0.5,  // 4-bit typically
            QuantizationMode::FP8_E4M3 => 1.0,
            QuantizationMode::FP8_E5M2 => 1.0,
        }
    }

    /// Check if mode requires special alignment
    pub fn alignment_requirement(&self) -> usize {
        match self {
            QuantizationMode::INT4 | QuantizationMode::NF4 => 8, // Pack 8 4-bit values
            QuantizationMode::GPTQ | QuantizationMode::AWQ => 32, // Warp-aligned
            _ => 1,
        }
    }
}

/// Attention mechanism optimization strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AttentionStrategy {
    Standard,
    FlashAttention,
    FlashAttention2,
    MemoryEfficient,
    Chunked,
    GradientCheckpointing,
}

/// Diffusion-specific memory configuration
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    pub precision_mode: PrecisionMode,
    pub attention_strategy: AttentionStrategy,
    pub quantization_mode: QuantizationMode,
    pub model_type: ModelType,
    pub max_sequence_length: usize,
    pub batch_size: usize,
    pub enable_flash_attention: bool,
    pub gradient_checkpointing: bool,
    pub prefetch_factor: f32,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            precision_mode: PrecisionMode::BFloat16,
            attention_strategy: AttentionStrategy::FlashAttention2,
            quantization_mode: QuantizationMode::None,
            model_type: ModelType::SDXL,
            max_sequence_length: 4096,
            batch_size: 1,
            enable_flash_attention: true,
            gradient_checkpointing: true,
            prefetch_factor: 1.1,
        }
    }
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    pub initial_size: usize,
    pub max_size: usize,
    pub block_size: usize,
    pub growth_factor: f32,
    pub garbage_collection_threshold: f32,
    pub enable_defragmentation: bool,
    pub enable_mixed_precision: bool,
    pub attention_memory_efficient: bool,
    pub gradient_checkpointing: bool,
    pub prefetch_factor: f32,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024 * 128, // 128MB
            max_size: usize::MAX,
            block_size: 1024 * 1024, // 1MB
            growth_factor: 1.5,
            garbage_collection_threshold: 0.8,
            enable_defragmentation: true,
            enable_mixed_precision: true,
            attention_memory_efficient: true,
            gradient_checkpointing: false,
            prefetch_factor: 1.2,
        }
    }
}

impl MemoryPoolConfig {
    /// Create configuration optimized for Flux training on 24GB GPUs
    pub fn flux_24gb() -> Self {
        Self {
            initial_size: 1024 * 1024 * 512,   // 512MB initial
            max_size: 22 * 1024 * 1024 * 1024, // 22GB max (leave 2GB for system)
            block_size: 1024 * 1024 * 4,       // 4MB blocks
            growth_factor: 1.2,
            garbage_collection_threshold: 0.85,
            enable_defragmentation: true,
            enable_mixed_precision: true,
            attention_memory_efficient: true,
            gradient_checkpointing: true,
            prefetch_factor: 1.1,
        }
    }
}
