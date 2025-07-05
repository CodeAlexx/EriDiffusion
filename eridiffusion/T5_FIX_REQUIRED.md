# T5 Token Length Fix Required for SD 3.5

## Issue
The T5-XXL encoder in `sd3_candle.rs` is currently limited to 77 tokens (same as CLIP), but SD 3.5 requires 154 tokens for T5.

## Required Changes in `sd3_candle.rs`

### 1. Update `StableDiffusion3TripleClipWithTokenizer::new_split()` (line 182):
```rust
// OLD:
let max_position_embeddings = 77usize;

// NEW:
let clip_max_position_embeddings = 77usize;
let t5_max_position_embeddings = 154usize;  // T5 needs 154 tokens
```

### 2. Update CLIP encoder creation (lines 183-223):
```rust
let clip_l = ClipWithTokenizer::new(
    vb_clip_l,
    stable_diffusion::clip::Config::sdxl(),
    "openai/clip-vit-large-patch14",
    clip_max_position_embeddings,  // Use 77 for CLIP
)?;

let clip_g = ClipWithTokenizer::new(
    vb_clip_g,
    stable_diffusion::clip::Config::sdxl2(),
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    clip_max_position_embeddings,  // Use 77 for CLIP
)?;
```

### 3. Update T5 encoder creation (line 200):
```rust
// OLD:
let t5 = T5WithTokenizer::new(vb_t5, max_position_embeddings)?;

// NEW:
let t5 = T5WithTokenizer::new(vb_t5, t5_max_position_embeddings)?;  // Use 154 for T5
```

### 4. Same changes in `StableDiffusion3TripleClipWithTokenizer::new()` (lines 210-234):
```rust
pub fn new(vb: VarBuilder) -> AnyhowResult<Self> {
    let clip_max_position_embeddings = 77usize;
    let t5_max_position_embeddings = 154usize;  // T5 needs 154 tokens
    
    // ... CLIP encoders use clip_max_position_embeddings ...
    
    let t5 = T5WithTokenizer::new(vb.pp("t5xxl.transformer"), t5_max_position_embeddings)?;
    // ...
}
```

### 5. Update T5WithTokenizer to respect the limit (line 151):
```rust
fn encode_text_to_embedding(&mut self, prompt: &str, device: &Device) -> AnyhowResult<Tensor> {
    let mut tokens = self.tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    
    // Ensure we don't exceed T5's max length
    if tokens.len() > self.max_position_embeddings {
        tokens.truncate(self.max_position_embeddings);
    }
    
    // Pad to max length
    tokens.resize(self.max_position_embeddings, 0);
    
    // ... rest of encoding
}
```

## Why This Matters

1. **T5-XXL** is designed to handle longer sequences than CLIP
2. **SD 3.5** specifically uses 154 tokens for T5 to capture more detailed prompt information
3. Truncating to 77 tokens loses important prompt details
4. The config file correctly specifies `t5_max_length: 154`

## Testing

After making these changes, the SD 3.5 training should:
1. Properly encode up to 154 tokens with T5
2. Maintain 77 tokens for CLIP-L and CLIP-G
3. Generate better quality outputs due to fuller prompt understanding