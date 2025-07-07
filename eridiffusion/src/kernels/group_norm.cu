#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cuda/std/type_traits>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 256

// Helper for type dispatch
template<typename T>
struct AccType {
    using type = float;
};

// Warp-level reduction with proper sync
template<typename T>
__device__ T warp_reduce_sum(T val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Block-level reduction
template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Final warp reduction
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        val = shared[threadIdx.x];
    } else {
        val = 0;
    }
    
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// Fixed GroupNorm forward kernel
template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int batch_size,
    int num_channels,
    int num_groups,
    int spatial_size,
    float eps
) {
    using AccT = typename AccType<T>::type;
    
    // Grid: (batch_size, num_groups)
    // Block: (min(spatial_size, MAX_THREADS_PER_BLOCK))
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || group_idx >= num_groups) return;
    
    int channels_per_group = num_channels / num_groups;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Shared memory for reductions
    extern __shared__ char shared_mem[];
    AccT* shared_sum = (AccT*)shared_mem;
    AccT* shared_var = (AccT*)(shared_mem + block_size * sizeof(AccT));
    
    // Calculate mean - grid-stride loop over spatial*channel dims
    AccT sum = 0;
    for (int i = tid; i < channels_per_group * spatial_size; i += block_size) {
        int c = i / spatial_size;
        int s = i % spatial_size;
        int channel_idx = group_idx * channels_per_group + c;
        int idx = ((batch_idx * num_channels + channel_idx) * spatial_size) + s;
        sum += (AccT)input[idx];
    }
    
    sum = block_reduce_sum(sum, shared_sum);
    
    __shared__ AccT group_mean;
    if (tid == 0) {
        group_mean = sum / (channels_per_group * spatial_size);
        mean_out[batch_idx * num_groups + group_idx] = group_mean;
    }
    __syncthreads();
    
    // Calculate variance
    AccT var_sum = 0;
    for (int i = tid; i < channels_per_group * spatial_size; i += block_size) {
        int c = i / spatial_size;
        int s = i % spatial_size;
        int channel_idx = group_idx * channels_per_group + c;
        int idx = ((batch_idx * num_channels + channel_idx) * spatial_size) + s;
        AccT diff = (AccT)input[idx] - group_mean;
        var_sum += diff * diff;
    }
    
    var_sum = block_reduce_sum(var_sum, shared_var);
    
    __shared__ AccT group_rstd;
    if (tid == 0) {
        AccT var = var_sum / (channels_per_group * spatial_size);
        group_rstd = rsqrtf(var + eps);
        rstd_out[batch_idx * num_groups + group_idx] = group_rstd;
    }
    __syncthreads();
    
    // Normalize and apply affine transform
    for (int i = tid; i < channels_per_group * spatial_size; i += block_size) {
        int c = i / spatial_size;
        int s = i % spatial_size;
        int channel_idx = group_idx * channels_per_group + c;
        int idx = ((batch_idx * num_channels + channel_idx) * spatial_size) + s;
        
        AccT normalized = ((AccT)input[idx] - group_mean) * group_rstd;
        
        if (weight != nullptr) {
            normalized *= (AccT)weight[channel_idx];
        }
        if (bias != nullptr) {
            normalized += (AccT)bias[channel_idx];
        }
        
        output[idx] = (T)normalized;
    }
}

// C interface with proper error handling
extern "C" {
    cudaError_t group_norm_forward_f32(
        const float* input,
        const float* weight,
        const float* bias,
        float* output,
        float* mean,
        float* rstd,
        int batch_size,
        int num_channels,
        int num_groups,
        int spatial_size,
        float eps
    ) {
        dim3 grid(batch_size, num_groups);
        int threads = min(spatial_size, MAX_THREADS_PER_BLOCK);
        
        // Shared memory size
        size_t shared_size = 2 * threads * sizeof(float);
        
        group_norm_forward_kernel<<<grid, threads, shared_size>>>(
            input, weight, bias, output, mean, rstd,
            batch_size, num_channels, num_groups, spatial_size, eps
        );
        
        return cudaGetLastError();
    }
}