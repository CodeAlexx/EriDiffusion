# Flux Tensor Naming Solution

## Problem
When loading Flux models, we encountered a tensor naming mismatch:
- **Model file contains**: `double_blocks.0.img_mlp.0.weight` and `double_blocks.0.img_mlp.2.weight`
- **candle-transformers expects**: `double_blocks.0.img_mlp.w1.weight` and `double_blocks.0.img_mlp.w2.weight`

## Root Cause
Our custom Flux implementation (`models/flux_lora/`) expected different tensor names than candle-transformers' official Flux implementation. The MLP layers in candle-transformers use VarBuilder paths "0" and "2" which get concatenated to form the full tensor paths.

## Solution
Use candle-transformers' Flux model directly instead of our custom implementation:

```rust
use candle_transformers::models::flux;

// Load using candle-transformers Flux
let flux_config = flux::model::Config::dev();
let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
let base_model = flux::model::Flux::new(&flux_config, vb)?;
```

## Key Insights
1. **Tensor naming conventions vary**: Different implementations may use different naming schemes (numbered vs named layers)
2. **candle-transformers compatibility**: When possible, use candle-transformers' implementations for better compatibility with pre-trained models
3. **Debugging approach**: Create simple utilities to inspect tensor names in safetensors files (see `check_flux_tensors.rs`)

## Memory Requirements
With the tensor naming fixed, we confirmed:
- Flux model alone: ~12GB VRAM (from 23.8GB file)
- With preprocessing complete: Model fits in 24GB without quantization
- Full training would still require quantization or additional optimizations

## Next Steps
1. Implement LoRA wrapping around candle-transformers' Flux model
2. Add int8 quantization support if needed for training
3. Update our custom Flux implementation to match candle-transformers' tensor naming