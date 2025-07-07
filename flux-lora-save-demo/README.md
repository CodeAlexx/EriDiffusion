# Flux LoRA Training Documentation

## Overview

This implementation provides Flux LoRA training that saves weights in AI-Toolkit compatible format. The weights can be used with SimpleTuner, AI-Toolkit, and other standard Flux LoRA inference tools.

## Key Design Decision

We save LoRA weights using the AI-Toolkit naming convention:
- Separate `to_q`, `to_k`, `to_v` layers (not combined `qkv`)
- Uses `transformer.` prefix
- Compatible with the ecosystem standard

## File Structure

```
flux_lora_demo.safetensors
├── transformer.double_blocks.0.img_attn.to_q.lora_A
├── transformer.double_blocks.0.img_attn.to_q.lora_B
├── transformer.double_blocks.0.img_attn.to_k.lora_A
├── transformer.double_blocks.0.img_attn.to_k.lora_B
├── transformer.double_blocks.0.img_attn.to_v.lora_A
├── transformer.double_blocks.0.img_attn.to_v.lora_B
├── transformer.double_blocks.0.img_mlp.0.lora_A
├── transformer.double_blocks.0.img_mlp.0.lora_B
└── ... (continues for all blocks)
```

## Usage

### Training
```rust
use eridiffusion::trainers::flux_lora_simple::SimpleFluxLoRATrainer;

let trainer = SimpleFluxLoRATrainer::new(
    device,
    dtype,
    lora_config,
    output_dir,
);

trainer.train_and_save()?;
```

### Configuration
```rust
let lora_config = LoRAConfig {
    rank: 32,
    alpha: 32.0,
    target_modules: vec![
        "to_q".to_string(),
        "to_k".to_string(),
        "to_v".to_string(),
        "proj_mlp".to_string(),
    ],
};
```

## Inference Options

### Option 1: Use with AI-Toolkit/SimpleTuner
The saved LoRA files work directly with these tools - no conversion needed.

### Option 2: Use with Candle
Since Candle's Flux uses combined `qkv` layers, you have two options:

1. **Modify Candle** to use separate Q,K,V layers (see `candle-flux-separate-qkv.patch`)
2. **Convert the weights** from separate to combined format

### Option 3: Use with ComfyUI/A1111
Most UI tools support AI-Toolkit format LoRAs directly.

## Technical Details

### Why Separate Q,K,V?
- **Training flexibility**: Easier to apply different learning rates or ranks to Q,K,V
- **Ecosystem standard**: AI-Toolkit and SimpleTuner use this format
- **Conversion simplicity**: Easy to combine into QKV if needed

### Base Model Compatibility
The base Flux models use combined `qkv` tensors:
```
double_blocks.0.img_attn.qkv.weight  # Shape: [9216, 3072]
```

But LoRA adapters use separate tensors:
```
transformer.double_blocks.0.img_attn.to_q.lora_A  # Shape: [32, 3072]
transformer.double_blocks.0.img_attn.to_k.lora_A  # Shape: [32, 3072]
transformer.double_blocks.0.img_attn.to_v.lora_A  # Shape: [32, 3072]
```

This is a common pattern where the base model is optimized for inference (combined) while adapters are optimized for training flexibility (separate).

## Examples

### Save LoRA Weights
```rust
let mut lora_weights = HashMap::new();

// Add your trained LoRA weights
lora_weights.insert(
    "double_blocks.0.img_attn.to_q".to_string(),
    (lora_a_tensor, lora_b_tensor)
);

// Save in AI-Toolkit format
save_flux_lora(lora_weights, &output_path, &lora_config)?;
```

### Load for Inference (Python)
```python
# Using AI-Toolkit
from safetensors import safe_open

with safe_open("flux_lora.safetensors", framework="pt") as f:
    # Weights are ready to use with AI-Toolkit format models
    to_q_lora_a = f.get_tensor("transformer.double_blocks.0.img_attn.to_q.lora_A")
```

## Troubleshooting

### "Cannot find tensor qkv.weight"
This means the inference code expects combined QKV. Either:
- Use a tool that supports AI-Toolkit format
- Convert the weights to combined format
- Modify the inference code to use separate Q,K,V

### Shape Mismatch Errors
Ensure:
- LoRA rank matches between training and inference
- Using correct dtype (F16/F32/BF16)
- Tensor naming matches exactly (including "transformer." prefix)