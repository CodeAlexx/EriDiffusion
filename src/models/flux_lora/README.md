# Flux LoRA Implementation Notes

## The Challenge

We need to reconcile two different approaches:
1. **AI-Toolkit**: Uses separate `to_q`, `to_k`, `to_v` layers for attention
2. **Candle**: Uses a single `qkv` layer that outputs all three

## Solution

### Option 1: Match Candle Exactly (Recommended)
Use Candle's structure and create LoRA adapters for their `qkv` layer:

```rust
// In attention forward pass
let qkv = xs.apply(&self.qkv)?;  // Base layer: [B, L, D] -> [B, L, 3*D]
let qkv = qkv + self.qkv_lora.forward(xs)?;  // Add LoRA adaptation
```

### Option 2: Split QKV (More Complex)
Modify the attention to use separate layers like AI-Toolkit:

```rust
pub struct SplitAttention {
    to_q: Linear,
    to_k: Linear, 
    to_v: Linear,
    to_out: Linear,
    // LoRA for each
    to_q_lora: Option<LoRAModule>,
    to_k_lora: Option<LoRAModule>,
    to_v_lora: Option<LoRAModule>,
}
```

### Option 3: Convert Weights
Load pre-trained AI-Toolkit LoRA and combine the weights:

```rust
// Pseudo-code for conversion
let q_lora_a = load("to_q.lora_A");
let k_lora_a = load("to_k.lora_A");
let v_lora_a = load("to_v.lora_A");

// Since they share input dimension, lora_A weights are the same
// Just use one of them for qkv.lora_A

// For lora_B, concatenate along output dimension
let qkv_lora_b = Tensor::cat(&[q_lora_b, k_lora_b, v_lora_b], 0)?;
```

## Current Implementation Status

We've created the building blocks:
1. `attention_candle.rs` - Attention matching Candle's structure
2. `double_block_candle.rs` - Double block with proper naming
3. `model_candle.rs` - Full model structure

The key is to use Candle's exact tensor naming when loading/saving weights.

## Tensor Naming Reference

See `FLUX_TENSOR_MAPPING.md` for complete tensor name mapping.