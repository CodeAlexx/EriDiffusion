# Flux Tensor Name Mapping (from Candle)

## Key Insights from Candle's Implementation

### 1. Model Structure
```rust
// Top-level components
img_in.weight                    // Linear layer for patchified input
txt_in.weight                    // Linear layer for text embeddings
time_in.in_layer.weight          // Time embedding first layer
time_in.out_layer.weight         // Time embedding second layer
vector_in.in_layer.weight        // Vector (pooled text) embedding
guidance_in.in_layer.weight      // Guidance embedding (if guidance_embed=true)
final_layer.linear.weight        // Final output projection
```

### 2. Double Blocks Structure
For each block `i` (0 to 18 for Flux-dev):
```rust
// Image stream
double_blocks.{i}.img_mod.lin.weight       // Modulation linear layer
double_blocks.{i}.img_norm1.weight         // Not in file - created as ones!
double_blocks.{i}.img_attn.qkv.weight      // Attention QKV projection
double_blocks.{i}.img_attn.norm.query_norm.weight  // Q normalization
double_blocks.{i}.img_attn.norm.key_norm.weight    // K normalization
double_blocks.{i}.img_attn.proj.weight     // Attention output projection
double_blocks.{i}.img_norm2.weight         // Not in file - created as ones!
double_blocks.{i}.img_mlp.0.weight         // MLP first layer
double_blocks.{i}.img_mlp.2.weight         // MLP second layer

// Text stream (same structure)
double_blocks.{i}.txt_mod.lin.weight
double_blocks.{i}.txt_norm1.weight         // Not in file - created as ones!
double_blocks.{i}.txt_attn.qkv.weight
double_blocks.{i}.txt_attn.norm.query_norm.weight
double_blocks.{i}.txt_attn.norm.key_norm.weight
double_blocks.{i}.txt_attn.proj.weight
double_blocks.{i}.txt_norm2.weight         // Not in file - created as ones!
double_blocks.{i}.txt_mlp.0.weight
double_blocks.{i}.txt_mlp.2.weight
```

### 3. Single Blocks Structure
For each block `i` (0 to 37 for Flux-dev):
```rust
single_blocks.{i}.modulation.lin.weight    // Modulation layer
single_blocks.{i}.pre_norm.weight          // Not in file - created as ones!
single_blocks.{i}.linear1.weight           // Combined QKV + MLP projection
single_blocks.{i}.linear2.weight           // Output projection
single_blocks.{i}.norm.query_norm.weight   // Q normalization
single_blocks.{i}.norm.key_norm.weight     // K normalization
```

### 4. LoRA Adapter Naming (AI-Toolkit style)
When saving LoRA adapters, AI-Toolkit uses:
```rust
// For attention layers
transformer.double_blocks.{i}.img_attn.to_q.lora_A
transformer.double_blocks.{i}.img_attn.to_q.lora_B
// BUT Candle uses qkv, so we need to adapt!

// For MLP layers
transformer.double_blocks.{i}.img_mlp.0.lora_A
transformer.double_blocks.{i}.img_mlp.0.lora_B
```

## Key Differences from Our Implementation

1. **Attention**: Candle uses single `qkv` layer, not separate `to_q`, `to_k`, `to_v`
2. **Normalization**: LayerNorm weights are NOT loaded from file - they're created as ones
3. **MLP naming**: Uses `0` and `2` for layer indices, not descriptive names
4. **Modulation**: Single `lin` layer that outputs 6*dim (for shift, scale, gate × 2)

## How to Match Candle Exactly

1. Use `vb.pp("name")` for nested paths
2. Create norm layers as ones: `Tensor::ones(dim, dtype, device)`
3. Use exact layer names: `qkv`, `proj`, `0`, `2`, etc.
4. Don't expect norm weights in the safetensors file

## Loading Process
```rust
// Candle loads like this:
let vb = VarBuilder::from_mmaped_safetensors(&[path], dtype, device)?;
let img_attn = SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("img_attn"))?;

// Which internally does:
let qkv = linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;  // Loads "img_attn.qkv.weight"
let proj = linear(dim, dim, vb.pp("proj"))?;                // Loads "img_attn.proj.weight"
```