//! FFI bindings for CUDA kernels
//! 
//! This module provides the Rust interface to the CUDA kernels
//! compiled from the .cu files.

#[cfg(feature = "cuda")]
pub mod cuda_kernels {
    use std::os::raw::{c_float, c_int, c_void};
    
    #[link(name = "flux_cuda_kernels", kind = "static")]
    extern "C" {
        // GroupNorm kernels
        pub fn group_norm_forward_f32(
            input: *const c_float,
            weight: *const c_float,
            bias: *const c_float,
            output: *mut c_float,
            mean: *mut c_float,
            rstd: *mut c_float,
            batch_size: c_int,
            num_channels: c_int,
            num_groups: c_int,
            spatial_size: c_int,
            eps: c_float,
        ) -> c_int;  // Returns cudaError_t
        
        // RoPE kernels
        pub fn rope_forward_f32(
            input: *const c_float,
            positions: *const c_int,
            output: *mut c_float,
            batch_size: c_int,
            seq_len: c_int,
            num_heads: c_int,
            head_dim: c_int,
            rotary_dim: c_int,
            theta_base: c_float,
            is_2d: c_int,
        ) -> c_int;  // Returns cudaError_t
        
        pub fn rope_forward_cached_f32(
            input: *const c_float,
            cos_cache: *const c_float,
            sin_cache: *const c_float,
            positions: *const c_int,
            output: *mut c_float,
            batch_size: c_int,
            seq_len: c_int,
            num_heads: c_int,
            head_dim: c_int,
            rotary_dim: c_int,
            max_cached_len: c_int,
            is_2d: c_int,
        ) -> c_int;  // Returns cudaError_t
        
        pub fn precompute_rope_cache_f32(
            cos_cache: *mut c_float,
            sin_cache: *mut c_float,
            max_seq_len: c_int,
            rotary_dim: c_int,
            theta_base: c_float,
        ) -> c_int;  // Returns cudaError_t
        
        // RMSNorm kernels (for completeness)
        pub fn rms_norm_f32(
            x: *const c_float,
            weight: *const c_float,
            out: *mut c_float,
            batch_size: c_int,
            seq_len: c_int,
            hidden_size: c_int,
            eps: c_float,
        ) -> c_int;
        
        pub fn rms_norm_f16(
            x: *const c_void,  // half*
            weight: *const c_void,  // half*
            out: *mut c_void,  // half*
            batch_size: c_int,
            seq_len: c_int,
            hidden_size: c_int,
            eps: c_float,
        ) -> c_int;
    }
}