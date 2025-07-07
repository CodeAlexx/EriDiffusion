# How to Modify Candle's Flux Implementation

## Option 1: Fork Candle (Recommended)

1. Fork the candle repository:
```bash
git clone https://github.com/huggingface/candle.git
cd candle
```

2. Modify the Flux implementation to use separate Q,K,V:

In `candle-transformers/src/models/flux/model.rs`:

```rust
// Change from:
pub struct SelfAttention {
    qkv: Linear,
    norm: QkNorm,
    proj: Linear,
    num_heads: usize,
}

// To:
pub struct SelfAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    norm: QkNorm,
    proj: Linear,
    num_heads: usize,
}

impl SelfAttention {
    fn new(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        // Change from combined qkv to separate layers
        let to_q = candle_nn::linear_b(dim, dim, qkv_bias, vb.pp("to_q"))?;
        let to_k = candle_nn::linear_b(dim, dim, qkv_bias, vb.pp("to_k"))?;
        let to_v = candle_nn::linear_b(dim, dim, qkv_bias, vb.pp("to_v"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let proj = candle_nn::linear(dim, dim, vb.pp("to_out.0"))?; // Note: to_out.0
        Ok(Self { to_q, to_k, to_v, norm, proj, num_heads })
    }
}
```

3. Update your Cargo.toml to use your fork:
```toml
[dependencies]
candle-core = { git = "https://github.com/YOUR_USERNAME/candle.git", branch = "flux-separate-qkv" }
candle-nn = { git = "https://github.com/YOUR_USERNAME/candle.git", branch = "flux-separate-qkv" }
candle-transformers = { git = "https://github.com/YOUR_USERNAME/candle.git", branch = "flux-separate-qkv" }
```

## Option 2: Local Path Override

1. Clone candle locally:
```bash
cd /home/alex/diffusers-rs
git clone https://github.com/huggingface/candle.git candle-local
```

2. Add to your Cargo.toml:
```toml
[patch.crates-io]
candle-core = { path = "../candle-local/candle-core" }
candle-nn = { path = "../candle-local/candle-nn" }
candle-transformers = { path = "../candle-local/candle-transformers" }
```

## Option 3: Create Wrapper (Without Modifying Candle)

Create a conversion layer that handles the QKV split:

```rust
pub struct FluxModelWrapper {
    base_model: candle_transformers::models::flux::Flux,
}

impl FluxModelWrapper {
    pub fn load_with_separated_qkv(path: &str) -> Result<Self> {
        // Load weights with custom naming
        let mut tensors = SafeTensors::load(path)?;
        
        // Convert separated Q,K,V to combined QKV
        for block in 0..19 {
            let q = tensors.remove(&format!("double_blocks.{}.img_attn.to_q.weight", block))?;
            let k = tensors.remove(&format!("double_blocks.{}.img_attn.to_k.weight", block))?;
            let v = tensors.remove(&format!("double_blocks.{}.img_attn.to_v.weight", block))?;
            
            // Concatenate along output dimension
            let qkv = Tensor::cat(&[q, k, v], 0)?;
            tensors.insert(format!("double_blocks.{}.img_attn.qkv.weight", block), qkv);
        }
        
        // Load with modified tensors
        // ...
    }
}
```

## Which Option to Choose?

1. **Fork Candle** - Best if you want a permanent solution and plan to contribute back
2. **Local Path** - Good for development and testing
3. **Wrapper** - Simplest if you just need compatibility without changing Candle

## Benefits of Modifying Candle

- Direct compatibility with AI-Toolkit LoRA format
- No conversion needed at inference time
- Can contribute improvements back to the community
- Full control over the implementation