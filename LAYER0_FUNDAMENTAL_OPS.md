# Layer 0: Fundamental Operations for Diffusion Models

Based on analysis of Flux, SD3.5, SDXL and other modern diffusion models, here are the minimal fundamental operations that Layer 0 needs to support:

## Core Tensor Operations

### 1. **Basic Arithmetic** (Most Critical)
- `add`, `sub`, `mul`, `div` - Element-wise operations
- `affine(scale, bias)` - Combined multiply + add (very common in normalization)
- `neg`, `abs`, `sign`
- `clamp(min, max)` - Used for stability

### 2. **Matrix Operations**
- `matmul` - Matrix multiplication (backbone of linear layers)
- `transpose` - Axis swapping
- `permute` - Arbitrary axis reordering
- `reshape/view` - Tensor reshaping without copying

### 3. **Reduction Operations**
- `sum`, `mean` - Along specified dimensions
- `var`, `std` - For normalization layers
- `max`, `min` - For pooling and clamping
- `argmax`, `argmin` - For sampling

### 4. **Shape Manipulation**
- `unsqueeze/squeeze` - Add/remove dimensions
- `expand/broadcast` - Broadcasting for element-wise ops
- `cat/concat` - Tensor concatenation
- `split/chunk` - Tensor splitting
- `flatten` - Reshape to 2D
- `unflatten` - Restore shape

## Activation Functions

### 5. **Essential Activations**
- `gelu` - Used in transformers (Flux, SD3.5)
- `silu/swish` - Used in ResNet blocks
- `relu` - Basic activation
- `sigmoid` - For gating
- `tanh` - For normalization

### 6. **Advanced Activations**
- `gelu_new/gelu_pytorch` - Exact GELU
- `quick_gelu` - Approximation for speed
- `mish` - Sometimes used in newer models

## Normalization Layers

### 7. **Critical Normalizations**
- `layer_norm` - Essential for transformers
- `rms_norm` - Used in modern models (Flux, SD3.5)
- `group_norm` - Used in CNNs/VAEs
- `batch_norm` - Less common but needed

### 8. **Normalization Helpers**
- `rsqrt` - Reciprocal square root (for fast norm)
- `pow` - Power operation
- `sqrt` - Square root
- `eps` handling - Small constant for numerical stability

## Attention Mechanisms

### 9. **Attention Operations**
- `scaled_dot_product_attention` - Core attention computation
- `softmax` - Along specified dimension
- `dropout` - Training regularization
- `causal_mask` - For autoregressive models

### 10. **Attention Optimizations**
- Flash Attention support (fused kernels)
- Memory-efficient attention
- Chunked attention for long sequences

## Convolution Operations (VAE)

### 11. **Conv Operations**
- `conv2d` - 2D convolution
- `conv_transpose2d` - Upsampling convolution
- `pad` - Padding operations (reflection, replication, constant)
- `avg_pool2d`, `max_pool2d` - Pooling layers

## Memory Patterns

### 12. **Memory Management**
- **Allocation patterns**: 
  - Large persistent weights (GB scale)
  - Temporary activations (dynamic size)
  - Gradient storage (training only)
- **Transfer operations**:
  - `to_device(cuda/cpu)` - Device movement
  - `to_dtype(fp32/fp16/bf16)` - Type conversion
  - `contiguous()` - Memory layout optimization
- **Memory pooling** - Reuse allocations
- **Gradient checkpointing** - Trade compute for memory

## Special Functions

### 13. **Embedding Operations**
- `embedding` - Token to vector lookup
- `positional_encoding` - Sinusoidal or learned
- `timestep_embedding` - For diffusion timesteps

### 14. **Loss Functions**
- `mse_loss` - Mean squared error
- `l1_loss` - Mean absolute error  
- `cross_entropy` - For classification
- `kl_divergence` - For VAEs

### 15. **Sampling Operations**
- `randn` - Gaussian noise generation
- `rand` - Uniform random
- `multinomial` - Categorical sampling
- `topk`, `topk_sampling` - For text generation

## Model-Specific Operations

### 16. **Flux Specific**
- Patchification: `reshape` + `permute` patterns
- Modulation: `linear` + `silu` + `chunk`
- Shifted window attention patterns

### 17. **SD3.5/MMDiT Specific**
- Adaptive layer norm (modulation)
- Cross-attention with pooled embeddings
- Joint attention over image + text

### 18. **VAE Specific**
- Reflection padding
- Group normalization with 32 groups
- Swish/SiLU activation
- Nearest neighbor upsampling

## Inference vs Training

### 19. **Inference Only**
- Forward pass operations
- No gradient computation
- Can use optimized/fused kernels
- Lower memory requirements

### 20. **Training Additional**
- `backward()` - Gradient computation
- `grad` access and manipulation
- `optimizer.step()` patterns
- `loss.backward()` 
- Gradient accumulation
- Mixed precision training hooks

## Minimal Implementation Priority

### Phase 1: Inference Capability
1. Basic tensor ops (add, mul, matmul, reshape)
2. Essential activations (gelu, silu)
3. Layer/RMS norm
4. Basic attention
5. Conv2d for VAE
6. Memory management

### Phase 2: Training Capability  
1. Gradient computation
2. Optimizer support
3. Loss functions
4. Dropout
5. Mixed precision

### Phase 3: Optimizations
1. Fused kernels
2. Flash attention
3. Memory pooling
4. Quantization support

## Device & Data Type Support

### Essential:
- CUDA/GPU support with proper memory management
- CPU fallback for testing
- FP32, FP16, BF16 support
- Proper type promotion rules

### Nice to have:
- INT8 quantization
- Multi-GPU support
- AMD ROCm support

## Key Insights from Analysis

1. **Candle limitations**: Missing CUDA kernels for RMS norm, group norm cause major issues
2. **Memory patterns**: Models allocate large persistent weights + dynamic activations
3. **Common patterns**: Most ops are variations of matmul + normalization + activation
4. **Precision matters**: BF16 preferred for training stability
5. **Fused operations**: Huge performance gains from fusing norm + activation

This is the minimal set needed to run modern diffusion models. A proper Layer 0 implementation would provide these as optimized primitives that higher layers can compose into full models.