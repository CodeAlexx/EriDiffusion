#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {

__global__ void rms_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    int total_sequences = batch_size * seq_len;
    
    if (idx >= total_sequences) return;
    
    const float* input_row = input + idx * hidden_size;
    float* output_row = output + idx * hidden_size;
    
    // Compute sum of squares using shared memory reduction
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = input_row[i];
        local_sum += val * val;
    }
    
    // Reduce within block
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Compute RMS
    float rms = sqrtf(shared_mem[0] / hidden_size + eps);
    
    // Apply normalization and weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        output_row[i] = (input_row[i] / rms) * weight[i];
    }
}

__global__ void rms_norm_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    float eps,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    int total_sequences = batch_size * seq_len;
    
    if (idx >= total_sequences) return;
    
    const __half* input_row = input + idx * hidden_size;
    __half* output_row = output + idx * hidden_size;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(input_row[i]);
        local_sum += val * val;
    }
    
    // Reduce within block
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Compute RMS
    float rms = sqrtf(shared_mem[0] / hidden_size + eps);
    
    // Apply normalization and weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = __half2float(input_row[i]) / rms;
        output_row[i] = __float2half(normalized * __half2float(weight[i]));
    }
}

}