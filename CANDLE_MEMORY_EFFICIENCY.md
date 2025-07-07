# How Candle Manages Memory Without OOM

## Key Differences from Our Implementation

### 1. **Sequential Model Loading**
Candle loads models one at a time and releases them when not needed:

```rust
// 1. First load and use text encoders
let text_embeddings = text_embeddings(...)?;  // Load CLIP
// CLIP is released after embeddings are generated

// 2. Then load VAE only when needed
let vae = sd_config.build_vae(vae_weights, &device, dtype)?;

// 3. Finally load UNet
let unet = sd_config.build_unet(unet_weights, &device, ...)?;
```

### 2. **Text Encoding is Done Once**
They encode text BEFORE loading the main models:
- Load CLIP → Generate embeddings → Unload CLIP
- Text embeddings are cached as tensors
- No need to keep text encoders in memory during sampling

### 3. **VAE is Used Efficiently**
- VAE encode is done once at the beginning (for img2img)
- VAE decode is only called at the very end
- During sampling loop, only UNet is active

### 4. **Our Implementation Issues**

Our trainer keeps everything in memory:
```rust
// We have all of these loaded at once:
- MMDiT model (8GB)
- LoKr adapters
- VAE (1GB) 
- CLIP-L (1GB)
- CLIP-G (2GB)
- T5-XXL (4GB)
- Optimizer states
- Gradients
```

### 5. **Why Candle Examples Don't OOM**

1. **No Training State**: They're only doing inference, no optimizer/gradients
2. **Sequential Loading**: Models are loaded/unloaded as needed
3. **Pre-computed Embeddings**: Text encoding happens once, early
4. **Single Model Active**: Only UNet is active during the main loop

### 6. **How to Fix Our Implementation**

To match Candle's efficiency:

```rust
// Option 1: Pre-encode all prompts before training starts
let all_embeddings = encode_all_prompts_once()?;
// Unload text encoders
drop(text_encoders);

// Option 2: Temporarily unload training models during sampling
fn generate_samples() {
    // Save optimizer state
    save_optimizer_state()?;
    
    // Clear training models from GPU
    drop(mmdit);
    drop(optimizer);
    
    // Load sampling models
    let vae = load_vae()?;
    let text_encoders = load_text_encoders()?;
    
    // Do sampling
    
    // Reload training models
    reload_training_state()?;
}

// Option 3: Use CPU for text encoding
let text_embeddings = encode_on_cpu(prompts)?;
text_embeddings.to_device(&gpu)?;
```

### 7. **The Core Insight**

Candle's examples work because they:
- Never have more than 2 models loaded simultaneously
- Use pre-computed embeddings whenever possible
- Load models just-in-time and release immediately after use
- Don't maintain training state (optimizer, gradients)

Our trainer tries to keep everything loaded for convenience, which causes OOM.