//! Model-specific memory block definitions from fluxmemory.txt v2

use std::ffi::c_void;
use super::{cuda_allocator::MemoryError, PrecisionMode, MemoryFormat};
use anyhow::Result;

/// MMDiT memory block for SD3.5
pub struct MMDiTMemoryBlock {
    pub patch_embed_ptr: *mut c_void,
    pub pos_embed_ptr: *mut c_void,
    pub q_ptr: *mut c_void,
    pub k_ptr: *mut c_void,
    pub v_ptr: *mut c_void,
    pub adaln_ptr: *mut c_void,
    pub resolution: (usize, usize),
    pub seq_len: usize,
    pub batch_size: usize,
}

/// WAN 2.1 video memory block
pub struct WAN21VideoMemoryBlock {
    pub video_latent_ptr: *mut c_void,      // 5D tensor: (B, T, C, H, W)
    pub temporal_embed_ptr: *mut c_void,     // Temporal position embeddings
    pub spatial_attn_ptr: *mut c_void,       // Spatial attention cache
    pub temporal_attn_ptr: *mut c_void,      // Temporal attention cache
    pub conv3d_workspace_ptr: *mut c_void,   // Workspace for 3D convolutions
    pub motion_vector_ptr: *mut c_void,      // Motion vectors for temporal coherence
    pub text_embed_ptr: *mut c_void,         // Text conditioning
    pub frame_buffer_ptr: *mut c_void,       // Frame buffer for cascaded generation
    pub batch_size: usize,
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

/// Flux-specific memory block
pub struct FluxMemoryBlock {
    // Double blocks (image and text)
    pub img_attn_q_ptr: *mut c_void,
    pub img_attn_k_ptr: *mut c_void,
    pub img_attn_v_ptr: *mut c_void,
    pub img_attn_out_ptr: *mut c_void,
    pub txt_attn_q_ptr: *mut c_void,
    pub txt_attn_k_ptr: *mut c_void,
    pub txt_attn_v_ptr: *mut c_void,
    pub txt_attn_out_ptr: *mut c_void,
    
    // MLP
    pub mlp_gate_ptr: *mut c_void,
    pub mlp_up_ptr: *mut c_void,
    pub mlp_down_ptr: *mut c_void,
    
    // Modulation
    pub mod_ptr: *mut c_void,
    
    // Metadata
    pub layer_idx: usize,
    pub is_double_block: bool,
    pub hidden_dim: usize,
    pub batch_size: usize,
}

/// Sharding strategy for multi-GPU tensors
#[derive(Debug, Clone)]
pub enum ShardingStrategy {
    Batch,           // Shard along batch dimension
    Channel,         // Shard along channel dimension  
    Spatial,         // Shard along spatial dimensions
    Temporal,        // Shard along temporal dimension (for video)
    Custom(usize),   // Shard along specified dimension
}

/// Attention cache key for prewarming
pub type AttentionCacheKey = (usize, usize); // (seq_len, batch_size)

/// Model type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    SD15,
    SD21,
    SDXL,
    SD3,
    SD35Medium,
    SD35Large,
    FluxDev,
    FluxSchnell,
    WAN21Video,
    LTXVideo,
    HunyuanVideo,
    Custom,
}

impl ModelType {
    /// Get recommended memory pool configuration
    pub fn recommended_pool_config(&self) -> super::MemoryPoolConfig {
        match self {
            ModelType::FluxDev | ModelType::FluxSchnell => {
                super::MemoryPoolConfig::flux_24gb()
            }
            ModelType::SD35Large => {
                super::MemoryPoolConfig {
                    initial_size: 16 * 1024 * 1024 * 1024, // 16GB
                    max_size: 20 * 1024 * 1024 * 1024,     // 20GB
                    block_size: 256 * 1024 * 1024,         // 256MB chunks
                    enable_defragmentation: true,
                    gradient_checkpointing: true,
                    ..Default::default()
                }
            }
            ModelType::WAN21Video | ModelType::LTXVideo | ModelType::HunyuanVideo => {
                super::MemoryPoolConfig {
                    initial_size: 18 * 1024 * 1024 * 1024,  // 18GB
                    max_size: 22 * 1024 * 1024 * 1024,      // 22GB
                    block_size: 512 * 1024 * 1024,          // 512MB chunks
                    enable_defragmentation: true,
                    gradient_checkpointing: true,
                    memory_format: MemoryFormat::ChannelsLast3d,
                    ..Default::default()
                }
            }
            _ => super::MemoryPoolConfig::default(),
        }
    }
    
    /// Get memory requirements estimate
    pub fn memory_requirements(&self) -> MemoryRequirements {
        match self {
            ModelType::FluxDev => MemoryRequirements {
                model_size: 12 * 1024 * 1024 * 1024,        // 12GB
                activation_size: 8 * 1024 * 1024 * 1024,    // 8GB
                gradient_size: 4 * 1024 * 1024 * 1024,      // 4GB
                optimizer_size: 8 * 1024 * 1024 * 1024,     // 8GB (AdamW)
                total_estimate: 32 * 1024 * 1024 * 1024,    // 32GB
            },
            ModelType::SD35Large => MemoryRequirements {
                model_size: 8 * 1024 * 1024 * 1024,         // 8GB
                activation_size: 6 * 1024 * 1024 * 1024,     // 6GB
                gradient_size: 3 * 1024 * 1024 * 1024,       // 3GB
                optimizer_size: 6 * 1024 * 1024 * 1024,      // 6GB
                total_estimate: 23 * 1024 * 1024 * 1024,     // 23GB
            },
            _ => MemoryRequirements::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    pub model_size: usize,
    pub activation_size: usize,
    pub gradient_size: usize,
    pub optimizer_size: usize,
    pub total_estimate: usize,
}

impl Default for MemoryRequirements {
    fn default() -> Self {
        Self {
            model_size: 4 * 1024 * 1024 * 1024,
            activation_size: 2 * 1024 * 1024 * 1024,
            gradient_size: 1 * 1024 * 1024 * 1024,
            optimizer_size: 2 * 1024 * 1024 * 1024,
            total_estimate: 9 * 1024 * 1024 * 1024,
        }
    }
}

/// Video-specific configuration
#[derive(Debug, Clone)]
pub struct VideoMemoryConfig {
    pub num_frames: usize,
    pub frame_height: usize,
    pub frame_width: usize,
    pub temporal_compression: usize,
    pub use_motion_vectors: bool,
    pub cascaded_generation: bool,
    pub frame_buffer_on_cpu: bool,
}

impl Default for VideoMemoryConfig {
    fn default() -> Self {
        Self {
            num_frames: 16,
            frame_height: 512,
            frame_width: 512,
            temporal_compression: 4,
            use_motion_vectors: true,
            cascaded_generation: false,
            frame_buffer_on_cpu: true,
        }
    }
}