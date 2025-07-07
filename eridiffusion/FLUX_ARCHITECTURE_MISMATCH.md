# Flux Architecture Mismatch Analysis

## Problem Summary
Our custom Flux model (`flux_custom`) has a different architecture than the checkpoint we're trying to load (`flux1-dev.safetensors`).

## Key Differences

### 1. Layer Normalization
- **Checkpoint**: Does not have separate layer norm weights (`img_norm1`, `txt_norm1`, etc.)
- **Our Model**: Expects layer norms from candle-transformers structure
- **Workaround**: Created dummy identity layer norms

### 2. Single Block Architecture
- **Checkpoint**: Uses unified architecture
  - `linear1`: Outputs both QKV (3 × 3072 = 9216) + MLP hidden (12288) = 21504 total dimensions
  - `linear2`: Takes concatenated attention + MLP outputs
  - No separate attention weights
- **Our Model**: Separate attention and MLP modules
  - Expects `attn.to_q/k/v.weight` separately
  - Expects `mlp.linear1/2.weight` separately
- **Workaround**: Created dummy attention weights and tried to extract MLP weights

### 3. Memory Issues
Creating all these dummy tensors causes OOM on 24GB GPU during model initialization.

## Recommendation
Use the candle-transformers Flux model instead of our custom implementation:
1. It's designed to work with this checkpoint format
2. It handles the unified single block architecture correctly
3. It should load without needing dummy tensors

## Implementation Status
- ✅ Fixed double block tensor mapping
- ✅ Fixed time/vector embedding naming
- ✅ Created layer norm identity transforms
- ❌ Single block architecture mismatch (fundamental issue)
- ❌ OOM due to dummy tensor creation

## Next Steps
1. Switch to `candle_transformers::models::flux::Flux`
2. Implement LoRA injection for the candle model
3. Test loading and training