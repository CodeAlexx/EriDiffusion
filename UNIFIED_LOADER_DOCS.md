# Unified Loader Documentation

## Overview

The Unified Loader system provides a flexible, extensible solution for loading various diffusion model checkpoints in EriDiffusion. It handles architectural differences, tensor name mismatches, and missing tensors across different model families.

## Architecture

The loader system consists of three main components:

### 1. UnifiedLoader (`loaders/unified_loader.rs`)
High-level loader with architecture detection and adapter system.

### 2. TensorRemapper (`loaders/tensor_remapper.rs`)
Flexible tensor mapping with fallbacks and synthesis capabilities.

### 3. Model-Specific Adapters
- `FluxAdapter` - Handles Flux model variants
- `SD15Adapter` - For SD 1.5 and SD 2.x models
- `SDXLAdapter` - For SDXL models
- `SD3Adapter` - For SD3 and SD3.5 models

## Usage

### Basic Usage

```rust
use eridiffusion::loaders::{UnifiedLoader, load_flux_weights};
use candle_core::{Device, DType};
use std::path::Path;

// Method 1: Using the helper function (Flux-specific)
let checkpoint_path = Path::new("/path/to/flux1-dev.safetensors");
let device = Device::cuda_if_available(0)?;
let dtype = DType::BF16;

let vb = load_flux_weights(checkpoint_path, device, dtype, 3072)?;

// Method 2: Using UnifiedLoader directly
let loader = UnifiedLoader::new(device, dtype);
let vb = loader.load_into_var_builder(checkpoint_path, "flux")?;

// Method 3: Using TensorRemapper for more control
use eridiffusion::loaders::create_flux_remapper;

let remapper = create_flux_remapper(checkpoint_path, device, dtype)?;
let tensor = remapper.load_with_fallbacks("double_blocks.0.img_attn.to_q.weight")?;
```

### Universal Usage (All Models)

```rust
use eridiffusion::loaders::create_model_remapper;

// Automatically detects model type
let remapper = create_model_remapper(checkpoint_path, device, dtype)?;

// Check detected model type
match remapper.model_type() {
    ModelType::FluxDev => println!("Loading Flux Dev model"),
    ModelType::SD35 => println!("Loading SD 3.5 model"),
    ModelType::SDXL => println!("Loading SDXL model"),
    _ => println!("Loading other model type"),
}

// Load tensors with automatic adaptation
let tensor = remapper.load_tensor("time_embedding.weight")?;
```

## Implementation Details

### Architecture Detection

The system automatically detects model architecture by examining tensor names:

```rust
pub fn detect(tensors: &HashMap<String, Tensor>) -> ModelType {
    if tensors.contains_key("double_blocks.0.img_attn.qkv.weight") {
        // Flux model detected
        if tensors.contains_key("guidance_in.weight") {
            ModelType::FluxDev
        } else {
            ModelType::FluxSchnell
        }
    } else if tensors.contains_key("x_embedder.weight") {
        // SD3/3.5 detected
        ModelType::SD35
    }
    // ... more detection logic
}
```

### Tensor Name Mapping

The remapper handles common naming differences between checkpoints:

1. **Direct Mapping**: Simple name translations
   ```rust
   "time_in.mlp.0.fc1.weight" → "time_in.in_layer.weight"
   "final_layer.weight" → "final_layer.linear.weight"
   ```

2. **Pattern-Based Mapping**: For systematic differences
   ```rust
   "double_blocks.*.img_mlp.linear1.weight" → "double_blocks.*.img_mlp.0.weight"
   ```

3. **Tensor Splitting**: For combined tensors (e.g., QKV)
   ```rust
   "attn.qkv.weight" → ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"]
   ```

4. **Tensor Synthesis**: For missing tensors
   ```rust
   // Create identity layer norms when missing
   "img_norm1.weight" → Tensor::ones(hidden_size)
   "img_norm1.bias" → Tensor::zeros(hidden_size)
   ```

### Memory Efficiency

- Loads tensors to CPU first to avoid OOM during processing
- Moves to target device only when needed
- Supports lazy loading for large checkpoints

## Extending for New Models

### Adding a New Model Type

1. **Add to ModelType enum**:
   ```rust
   pub enum ModelType {
       // ... existing types
       HiDream,
       Chroma,
       OmniGen2,
       WanVace21,
   }
   ```

2. **Update detection logic**:
   ```rust
   impl ModelType {
       pub fn detect(tensors: &HashMap<String, Tensor>) -> Self {
           // Add detection for new model
           if tensors.contains_key("hidream_specific_tensor") {
               return ModelType::HiDream;
           }
           // ...
       }
   }
   ```

3. **Create model-specific adapter**:
   ```rust
   pub struct HiDreamAdapter;
   
   impl WeightAdapter for HiDreamAdapter {
       fn can_adapt(&self, from_arch: &str, to_arch: &str) -> bool {
           from_arch == "hidream" && to_arch == "hidream"
       }
       
       fn adapt_name(&self, name: &str) -> String {
           // HiDream-specific name mappings
           name.replace("old_name", "new_name")
       }
       
       fn adapt_tensor(&self, name: &str, tensor: Tensor) -> Result<Vec<(String, Tensor)>> {
           // Handle HiDream-specific tensor transformations
           Ok(vec![(self.adapt_name(name), tensor)])
       }
   }
   ```

4. **Add mappings to UniversalRemapper**:
   ```rust
   fn add_hidream_mappings(&mut self) {
       self.remapper.add_mapping(
           "hidream.old_layer".to_string(),
           "hidream.new_layer".to_string(),
       );
       // Add more mappings
   }
   ```

### Example: Adding WAN Vace 2.1 Support

```rust
// 1. Detection
if tensors.contains_key("video_encoder.temporal_blocks.0.weight") &&
   tensors.contains_key("motion_module.weight") {
    return ModelType::WanVace21;
}

// 2. Adapter
pub struct WanVaceAdapter {
    num_frames: usize,
}

impl WeightAdapter for WanVaceAdapter {
    fn adapt_tensor(&self, name: &str, tensor: Tensor) -> Result<Vec<(String, Tensor)>> {
        // Handle temporal dimension reshaping
        if name.contains("temporal_blocks") {
            let reshaped = self.reshape_temporal_tensor(&tensor)?;
            Ok(vec![(name.to_string(), reshaped)])
        } else {
            Ok(vec![(name.to_string(), tensor)])
        }
    }
}

// 3. Mappings
fn add_wanvace_mappings(&mut self) {
    // Video-specific mappings
    self.remapper.add_mapping(
        "temporal_encoder.weight".to_string(),
        "video_encoder.temporal_blocks.0.weight".to_string(),
    );
}
```

## Common Modifications

### Adding Custom Tensor Synthesis

```rust
// In TensorRemapper
fn synthesize_tensor(&self, model_path: &str) -> Result<Option<Tensor>> {
    // Add custom synthesis logic
    if model_path.contains("custom_layer") {
        let tensor = self.create_custom_tensor(model_path)?;
        return Ok(Some(tensor));
    }
    
    // Fall back to existing synthesis
    self.synthesize_default(model_path)
}
```

### Supporting Quantized Models

```rust
// Add quantization support
pub struct QuantizedAdapter {
    base_adapter: Box<dyn WeightAdapter>,
    quantization: QuantizationType,
}

impl WeightAdapter for QuantizedAdapter {
    fn adapt_tensor(&self, name: &str, tensor: Tensor) -> Result<Vec<(String, Tensor)>> {
        let base_result = self.base_adapter.adapt_tensor(name, tensor)?;
        
        // Dequantize if needed
        base_result.into_iter()
            .map(|(name, tensor)| {
                let dequantized = self.dequantize(&tensor)?;
                Ok((name, dequantized))
            })
            .collect()
    }
}
```

### Adding Checkpoint Format Support

```rust
// Support for different file formats
pub trait CheckpointLoader {
    fn load(&self, path: &Path) -> Result<HashMap<String, Tensor>>;
}

pub struct GGUFLoader;
impl CheckpointLoader for GGUFLoader {
    fn load(&self, path: &Path) -> Result<HashMap<String, Tensor>> {
        // Load GGUF format
        gguf::load_tensors(path)
    }
}

pub struct PickleLoader;
impl CheckpointLoader for PickleLoader {
    fn load(&self, path: &Path) -> Result<HashMap<String, Tensor>> {
        // Load pickle format
        pickle::load_tensors(path)
    }
}
```

## Troubleshooting

### Common Issues

1. **"No tensor found for: X"**
   - Check if the tensor name needs mapping
   - Verify the checkpoint contains the expected tensor
   - Add custom synthesis if the tensor is optional

2. **Shape mismatches**
   - Ensure correct model variant is detected
   - Check if tensor needs reshaping in adapter
   - Verify hidden_size and other dimensions

3. **OOM during loading**
   - Load to CPU first, then move to GPU
   - Use chunked loading for very large models
   - Enable memory pooling

### Debugging

```rust
// Enable debug mode
let remapper = TensorRemapper::from_checkpoint(path, device, dtype)?
    .with_debug(true);

// List all available tensors
for (name, tensor) in remapper.tensors() {
    println!("{}: {:?}", name, tensor.shape());
}

// Test specific mapping
match remapper.load_with_fallbacks("problematic_tensor") {
    Ok(t) => println!("Success: {:?}", t.shape()),
    Err(e) => println!("Failed: {}", e),
}
```

## Best Practices

1. **Always load to CPU first** for large models to avoid OOM
2. **Use appropriate dtype** (BF16 for Flux, FP16 for most others)
3. **Cache remapper instances** when loading multiple tensors
4. **Add model detection** before blindly applying mappings
5. **Document tensor mappings** for each model type
6. **Test with minimal examples** before full integration

## Future Enhancements

1. **Lazy loading** - Load tensors on-demand
2. **Streaming support** - For models larger than RAM
3. **Automatic optimization** - Detect and apply best loading strategy
4. **Version detection** - Handle different versions of same model
5. **Conversion utilities** - Tools to convert between checkpoint formats
6. **Validation** - Verify loaded tensors match expected architecture