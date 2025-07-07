# Flux LoRA Training & Inference Strategy

## The Clean Solution

### 1. Training (Our Code)
- Use AI-Toolkit format with separate `to_q`, `to_k`, `to_v`
- Save with safetensors crate using AI-Toolkit naming:
  - `transformer.double_blocks.{i}.img_attn.to_q.lora_A`
  - `transformer.double_blocks.{i}.img_attn.to_k.lora_B`
  - etc.

### 2. Inference (Candle)
- Let Candle handle loading the base Flux model (with `qkv`)
- They can adapt/convert LoRA weights as needed
- We don't need to worry about their internal structure

### 3. Key Insight
The base Flux model uses `qkv` (combined), but LoRA adapters use `to_q/to_k/to_v` (separate). This is actually common:
- Base model: Optimized for inference (combined QKV)
- LoRA adapters: Optimized for training flexibility (separate Q,K,V)

## Implementation

### For Saving LoRA (save_lora.rs)
```rust
// Save in AI-Toolkit format
let key = "transformer.double_blocks.0.img_attn.to_q.lora_A";
tensors.insert(key, lora_weights);
```

### For Training
```rust
// Train with separate Q,K,V from the start
struct FluxAttentionForTraining {
    to_q: Linear,
    to_k: Linear, 
    to_v: Linear,
    // LoRA for each
}
```

### For Users
1. Train with our code → Get AI-Toolkit compatible LoRA
2. Use with SimpleTuner/AI-Toolkit → Works perfectly
3. Use with Candle for inference → They handle conversion

## Benefits
- No complex conversion code
- Compatible with existing ecosystem
- Clean separation of concerns
- Matches what users expect