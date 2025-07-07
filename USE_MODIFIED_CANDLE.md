# Using Modified Candle with Separate Q,K,V

## Steps to Use Your Local Candle

### 1. Apply the Modification

```bash
cd /home/alex/diffusers-rs/candle-official
git checkout -b flux-separate-qkv
# Apply the patch manually or edit the file directly
```

Edit `/home/alex/diffusers-rs/candle-official/candle-transformers/src/models/flux/model.rs`:

```rust
// Around line 295, change SelfAttention struct:
pub struct SelfAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    norm: QkNorm,
    proj: Linear,  // or to_out for AI-Toolkit compatibility
    num_heads: usize,
}

// Update the new() method:
fn new(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder) -> Result<Self> {
    let head_dim = dim / num_heads;
    let to_q = candle_nn::linear_b(dim, dim, qkv_bias, vb.pp("to_q"))?;
    let to_k = candle_nn::linear_b(dim, dim, qkv_bias, vb.pp("to_k"))?;
    let to_v = candle_nn::linear_b(dim, dim, qkv_bias, vb.pp("to_v"))?;
    let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
    let proj = candle_nn::linear(dim, dim, vb.pp("to_out.0"))?; // AI-Toolkit naming
    Ok(Self { to_q, to_k, to_v, norm, proj, num_heads })
}

// Update the qkv() method:
fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, l, d) = xs.dims3()?;
    let head_dim = d / self.num_heads;
    
    // Apply separate projections
    let q = xs.apply(&self.to_q)?
        .reshape((b, l, self.num_heads, head_dim))?
        .transpose(1, 2)?;
    let k = xs.apply(&self.to_k)?
        .reshape((b, l, self.num_heads, head_dim))?
        .transpose(1, 2)?;
    let v = xs.apply(&self.to_v)?
        .reshape((b, l, self.num_heads, head_dim))?
        .transpose(1, 2)?;
    
    // Apply normalization
    let q = q.apply(&self.norm.query_norm)?;
    let k = k.apply(&self.norm.key_norm)?;
    Ok((q, k, v))
}
```

### 2. Update Your Cargo.toml

Add this to your `Cargo.toml`:

```toml
[patch.crates-io]
candle-core = { path = "/home/alex/diffusers-rs/candle-official/candle-core" }
candle-nn = { path = "/home/alex/diffusers-rs/candle-official/candle-nn" }
candle-transformers = { path = "/home/alex/diffusers-rs/candle-official/candle-transformers" }
```

### 3. Rebuild Your Project

```bash
cargo clean
cargo build
```

## Benefits

1. **Direct Compatibility**: Candle will now load AI-Toolkit format LoRAs directly
2. **No Conversion Needed**: Your LoRA weights work without modification
3. **Same Performance**: No overhead from conversion

## Alternative: Compatibility Layer

If you don't want to modify Candle, create a compatibility layer:

```rust
// Load AI-Toolkit format and convert to Candle format
pub fn load_flux_with_aikotoolkit_lora(
    base_model_path: &str,
    lora_path: &str,
) -> Result<FluxModel> {
    // This would handle the naming conversion
    // to_q/to_k/to_v -> qkv
    // to_out.0 -> proj
}
```

## Which Approach?

- **Modify Candle**: Best for long-term compatibility
- **Compatibility Layer**: Good if you need to support both formats
- **Just Training**: If you only train, save in AI-Toolkit format and let users handle conversion