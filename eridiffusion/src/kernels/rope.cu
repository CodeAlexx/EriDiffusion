#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

template<typename T>
struct Vec2 {
    T x, y;
    __device__ Vec2(T x_, T y_) : x(x_), y(y_) {}
};

// Fixed RoPE kernel with correct indexing
template<typename T>
__global__ void rope_forward_kernel(
    const T* __restrict__ input,
    const int* __restrict__ positions,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim,
    float theta_base,
    int is_2d  // Changed from bool to int for FFI
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * num_heads * head_dim;
    
    if (idx >= total_elements) return;
    
    // Decompose index
    int b = idx / (seq_len * num_heads * head_dim);
    int s = (idx / (num_heads * head_dim)) % seq_len;
    int h = (idx / head_dim) % num_heads;
    int d = idx % head_dim;
    
    // Copy non-rotated dimensions
    if (d >= rotary_dim) {
        output[idx] = input[idx];
        return;
    }
    
    // Ensure we're working with pairs
    if (d % 2 == 1) return;  // Process pairs in even indices only
    
    // Get position
    float pos;
    int d_rot = d;  // Keep original d for indexing
    
    if (is_2d) {
        // For 2D: first half of rotary_dim for height, second half for width
        if (d < rotary_dim / 2) {
            pos = (float)positions[b * seq_len * 2 + s * 2];
        } else {
            pos = (float)positions[b * seq_len * 2 + s * 2 + 1];
            d_rot = d - rotary_dim / 2;
        }
    } else {
        // 1D position
        pos = (float)positions[b * seq_len + s];
    }
    
    // Calculate rotation
    int d_pair = d_rot / 2;
    float freq = 1.0f / powf(theta_base, 2.0f * d_pair / (float)rotary_dim);
    float angle = pos * freq;
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    
    // Apply rotation to pair
    float x0 = (float)input[idx];
    float x1 = (float)input[idx + 1];
    
    output[idx] = (T)(x0 * cos_angle - x1 * sin_angle);
    output[idx + 1] = (T)(x0 * sin_angle + x1 * cos_angle);
}

// Optimized cached version
template<typename T>
__global__ void rope_forward_cached_kernel(
    const T* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int* __restrict__ positions,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim,
    int max_cached_len,
    int is_2d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * num_heads * head_dim;
    
    if (idx >= total_elements) return;
    
    int b = idx / (seq_len * num_heads * head_dim);
    int s = (idx / (num_heads * head_dim)) % seq_len;
    int h = (idx / head_dim) % num_heads;
    int d = idx % head_dim;
    
    if (d >= rotary_dim) {
        output[idx] = input[idx];
        return;
    }
    
    if (d % 2 == 1) return;
    
    int pos_idx;
    int d_rot = d;
    int cache_dim = is_2d ? (rotary_dim / 4) : (rotary_dim / 2);  // Adjust for 2D
    
    if (is_2d) {
        if (d < rotary_dim / 2) {
            pos_idx = positions[b * seq_len * 2 + s * 2];
            d_rot = d;
        } else {
            pos_idx = positions[b * seq_len * 2 + s * 2 + 1];
            d_rot = d - rotary_dim / 2;
        }
    } else {
        pos_idx = positions[b * seq_len + s];
    }
    
    if (pos_idx >= max_cached_len) {
        // Fallback to computing on the fly
        float freq = 1.0f / powf(10000.0f, 2.0f * (d_rot / 2) / (float)rotary_dim);
        float angle = pos_idx * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        float x0 = (float)input[idx];
        float x1 = (float)input[idx + 1];
        
        output[idx] = (T)(x0 * cos_val - x1 * sin_val);
        output[idx + 1] = (T)(x0 * sin_val + x1 * cos_val);
        return;
    }
    
    int d_pair = d_rot / 2;
    int cache_idx = pos_idx * cache_dim + d_pair;
    
    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];
    
    float x0 = (float)input[idx];
    float x1 = (float)input[idx + 1];
    
    output[idx] = (T)(x0 * cos_val - x1 * sin_val);
    output[idx + 1] = (T)(x0 * sin_val + x1 * cos_val);
}

// Cache precomputation
__global__ void precompute_rope_cache_kernel(
    float* cos_cache,
    float* sin_cache,
    int max_seq_len,
    int rotary_dim,
    float theta_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_seq_len * (rotary_dim / 2);
    
    if (idx >= total) return;
    
    int pos = idx / (rotary_dim / 2);
    int d_pair = idx % (rotary_dim / 2);
    
    float freq = 1.0f / powf(theta_base, 2.0f * d_pair / (float)rotary_dim);
    float angle = pos * freq;
    
    cos_cache[idx] = cosf(angle);
    sin_cache[idx] = sinf(angle);
}

extern "C" {
    cudaError_t rope_forward_f32(
        const float* input,
        const int* positions,
        float* output,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim,
        int rotary_dim,
        float theta_base,
        int is_2d
    ) {
        int total = batch_size * seq_len * num_heads * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        rope_forward_kernel<<<blocks, threads>>>(
            input, positions, output,
            batch_size, seq_len, num_heads, head_dim,
            rotary_dim, theta_base, is_2d
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t rope_forward_cached_f32(
        const float* input,
        const float* cos_cache,
        const float* sin_cache,
        const int* positions,
        float* output,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim,
        int rotary_dim,
        int max_cached_len,
        int is_2d
    ) {
        int total = batch_size * seq_len * num_heads * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        rope_forward_cached_kernel<<<blocks, threads>>>(
            input, cos_cache, sin_cache, positions, output,
            batch_size, seq_len, num_heads, head_dim,
            rotary_dim, max_cached_len, is_2d
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t precompute_rope_cache_f32(
        float* cos_cache,
        float* sin_cache,
        int max_seq_len,
        int rotary_dim,
        float theta_base
    ) {
        int total = max_seq_len * (rotary_dim / 2);
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        precompute_rope_cache_kernel<<<blocks, threads>>>(
            cos_cache, sin_cache, max_seq_len, rotary_dim, theta_base
        );
        
        return cudaGetLastError();
    }
}