// Fused Adam optimizer CUDA kernel
// This performs the entire Adam update in a single kernel launch

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Helper for float4 vectorized operations
__device__ __forceinline__ float4 make_float4(float a, float b, float c, float d) {
    return {a, b, c, d};
}

// Standard Adam update kernel
__global__ void adam_update_kernel(
    float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const int t,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Load values
    float g = grad[idx];
    float m_val = m[idx];
    float v_val = v[idx];
    
    // Update biased first moment estimate
    m_val = beta1 * m_val + (1.0f - beta1) * g;
    
    // Update biased second raw moment estimate
    v_val = beta2 * v_val + (1.0f - beta2) * g * g;
    
    // Compute bias-corrected first moment estimate
    float m_hat = m_val / (1.0f - powf(beta1, t));
    
    // Compute bias-corrected second raw moment estimate
    float v_hat = v_val / (1.0f - powf(beta2, t));
    
    // Update parameter
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    
    // Store updated moments
    m[idx] = m_val;
    v[idx] = v_val;
}

// Vectorized Adam update kernel (processes 4 elements at once)
__global__ void adam_update_vectorized_kernel(
    float4* __restrict__ param,
    float4* __restrict__ m,
    float4* __restrict__ v,
    const float4* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const int t,
    const int size_vec4
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size_vec4) return;
    
    // Load float4 values
    float4 g = grad[idx];
    float4 m_val = m[idx];
    float4 v_val = v[idx];
    float4 p_val = param[idx];
    
    // Precompute bias correction factors
    float bias_correction1 = 1.0f - powf(beta1, t);
    float bias_correction2 = 1.0f - powf(beta2, t);
    
    // Process 4 elements
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float gi = ((float*)&g)[i];
        float mi = ((float*)&m_val)[i];
        float vi = ((float*)&v_val)[i];
        
        // Update moments
        mi = beta1 * mi + (1.0f - beta1) * gi;
        vi = beta2 * vi + (1.0f - beta2) * gi * gi;
        
        // Bias correction
        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;
        
        // Update parameter
        ((float*)&p_val)[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        ((float*)&m_val)[i] = mi;
        ((float*)&v_val)[i] = vi;
    }
    
    // Store updated values
    param[idx] = p_val;
    m[idx] = m_val;
    v[idx] = v_val;
}

// Adam update with weight decay (AdamW)
__global__ void adamw_update_kernel(
    float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const int t,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Load values
    float p_val = param[idx];
    float g = grad[idx];
    float m_val = m[idx];
    float v_val = v[idx];
    
    // Apply weight decay directly to parameter
    p_val *= (1.0f - lr * weight_decay);
    
    // Update moments
    m_val = beta1 * m_val + (1.0f - beta1) * g;
    v_val = beta2 * v_val + (1.0f - beta2) * g * g;
    
    // Bias correction
    float m_hat = m_val / (1.0f - powf(beta1, t));
    float v_hat = v_val / (1.0f - powf(beta2, t));
    
    // Update parameter
    p_val -= lr * m_hat / (sqrtf(v_hat) + eps);
    
    // Store updated values
    param[idx] = p_val;
    m[idx] = m_val;
    v[idx] = v_val;
}

// Mixed precision Adam (FP16 parameters, FP32 master weights and moments)
__global__ void adam_mixed_precision_kernel(
    half* __restrict__ param_fp16,          // FP16 parameters
    float* __restrict__ master_param,       // FP32 master copy
    float* __restrict__ m,                  // FP32 moments
    float* __restrict__ v,                  // FP32 moments
    const half* __restrict__ grad_fp16,     // FP16 gradients
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const int t,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Convert gradient to FP32
    float g = __half2float(grad_fp16[idx]);
    
    // Load FP32 values
    float p_val = master_param[idx];
    float m_val = m[idx];
    float v_val = v[idx];
    
    // Adam update in FP32
    m_val = beta1 * m_val + (1.0f - beta1) * g;
    v_val = beta2 * v_val + (1.0f - beta2) * g * g;
    
    float m_hat = m_val / (1.0f - powf(beta1, t));
    float v_hat = v_val / (1.0f - powf(beta2, t));
    
    p_val -= lr * m_hat / (sqrtf(v_hat) + eps);
    
    // Store FP32 master weight
    master_param[idx] = p_val;
    
    // Convert and store FP16 parameter
    param_fp16[idx] = __float2half(p_val);
    
    // Store FP32 moments
    m[idx] = m_val;
    v[idx] = v_val;
}

// Fused LoRA Adam update - updates both down and up projections in one kernel
__global__ void lora_adam_fused_kernel(
    float* __restrict__ down_param,
    float* __restrict__ down_m,
    float* __restrict__ down_v,
    const float* __restrict__ down_grad,
    float* __restrict__ up_param,
    float* __restrict__ up_m,
    float* __restrict__ up_v,
    const float* __restrict__ up_grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const int t,
    const int down_size,
    const int up_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = down_size + up_size;
    
    if (idx >= total_size) return;
    
    // Precompute bias corrections
    float bias_correction1 = 1.0f - powf(beta1, t);
    float bias_correction2 = 1.0f - powf(beta2, t);
    
    if (idx < down_size) {
        // Update down projection
        float g = down_grad[idx];
        float m_val = down_m[idx];
        float v_val = down_v[idx];
        
        m_val = beta1 * m_val + (1.0f - beta1) * g;
        v_val = beta2 * v_val + (1.0f - beta2) * g * g;
        
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;
        
        down_param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
        down_m[idx] = m_val;
        down_v[idx] = v_val;
    } else {
        // Update up projection
        int up_idx = idx - down_size;
        float g = up_grad[up_idx];
        float m_val = up_m[up_idx];
        float v_val = up_v[up_idx];
        
        m_val = beta1 * m_val + (1.0f - beta1) * g;
        v_val = beta2 * v_val + (1.0f - beta2) * g * g;
        
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;
        
        up_param[up_idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
        up_m[up_idx] = m_val;
        up_v[up_idx] = v_val;
    }
}

// C++ wrapper functions
extern "C" {

void launch_adam_update(
    float* param,
    float* m,
    float* v,
    const float* grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    adam_update_kernel<<<blocks, threads, 0, stream>>>(
        param, m, v, grad, lr, beta1, beta2, eps, t, size
    );
}

void launch_adam_update_vectorized(
    float* param,
    float* m,
    float* v,
    const float* grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t,
    int size,
    cudaStream_t stream
) {
    // Ensure size is divisible by 4
    if (size % 4 != 0) {
        // Fall back to regular kernel
        launch_adam_update(param, m, v, grad, lr, beta1, beta2, eps, t, size, stream);
        return;
    }
    
    const int size_vec4 = size / 4;
    const int threads = 256;
    const int blocks = (size_vec4 + threads - 1) / threads;
    
    adam_update_vectorized_kernel<<<blocks, threads, 0, stream>>>(
        (float4*)param, (float4*)m, (float4*)v, (const float4*)grad,
        lr, beta1, beta2, eps, t, size_vec4
    );
}

void launch_adamw_update(
    float* param,
    float* m,
    float* v,
    const float* grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    adamw_update_kernel<<<blocks, threads, 0, stream>>>(
        param, m, v, grad, lr, beta1, beta2, eps, weight_decay, t, size
    );
}

void launch_adam_mixed_precision(
    half* param_fp16,
    float* master_param,
    float* m,
    float* v,
    const half* grad_fp16,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    adam_mixed_precision_kernel<<<blocks, threads, 0, stream>>>(
        param_fp16, master_param, m, v, grad_fp16,
        lr, beta1, beta2, eps, t, size
    );
}

void launch_lora_adam_fused(
    float* down_param,
    float* down_m,
    float* down_v,
    const float* down_grad,
    float* up_param,
    float* up_m,
    float* up_v,
    const float* up_grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t,
    int down_size,
    int up_size,
    cudaStream_t stream
) {
    const int total_size = down_size + up_size;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    
    lora_adam_fused_kernel<<<blocks, threads, 0, stream>>>(
        down_param, down_m, down_v, down_grad,
        up_param, up_m, up_v, up_grad,
        lr, beta1, beta2, eps, t, down_size, up_size
    );
}

} // extern "C"