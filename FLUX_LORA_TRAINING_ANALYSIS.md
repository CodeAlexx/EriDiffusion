# Flux LoRA Training Technical Analysis

## Executive Summary

The Flux LoRA training is blocked because the current implementation cannot expose Flux model parameters for LoRA injection. The `FluxDiffusionModel` wrapper returns empty vectors for both `trainable_parameters()` and `parameters()`, making it impossible to inject LoRA adapters into the model layers.

## Core Technical Blockers

### 1. Model Parameter Access Problem

**Current State:**
```rust
// In flux_model_loader.rs
impl DiffusionModel for FluxDiffusionModel {
    fn trainable_parameters(&self) -> Vec<&Tensor> {
        vec![]  // Returns empty - NO PARAMETERS EXPOSED
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        vec![]  // Returns empty - NO PARAMETERS EXPOSED
    }
}
```

**Why This Blocks LoRA:**
- LoRA needs to inject low-rank adapters into specific model layers (attention Q/K/V projections)
- Without access to the model's internal parameters, we cannot:
  - Identify target modules for LoRA injection
  - Create LoRA layers that wrap the original weights
  - Track gradients for training

### 2. Candle Flux Model Structure

The candle-transformers Flux model (`flux::model::Flux`) is likely an opaque struct that doesn't expose its internal layers:
- No public access to attention modules
- No way to traverse the model graph
- No module registration system like PyTorch's `named_modules()`

### 3. Missing Infrastructure

**What SimpleTuner Does (Python):**
```python
# SimpleTuner can access model internals
target_modules = ["to_k", "to_q", "to_v", "add_k_proj", "add_q_proj", "add_v_proj", "to_out.0", "to_add_out"]
model.add_adapter(lora_config)  # Diffusers provides this method
```

**What We Need in Rust:**
1. Access to Flux model's internal structure
2. A way to wrap specific layers with LoRA adapters
3. Parameter tracking for optimization

## Detailed Technical Requirements

### 1. Model Introspection
We need to access Flux's internal layers:
- Double transformer blocks
- Single transformer blocks  
- Attention modules (Q/K/V projections)
- Feed-forward networks

### 2. LoRA Injection Points
Based on SimpleTuner's implementation, Flux LoRA targets:
- **Image attention**: `to_k`, `to_q`, `to_v`, `to_out.0`
- **Text cross-attention**: `add_k_proj`, `add_q_proj`, `add_v_proj`, `to_add_out`
- **Optional FFN**: `ff.net.0.proj`, `ff.net.2`, `ff_context.net.0.proj`, `ff_context.net.2`

### 3. Parameter Management
Need to implement:
- Parameter collection from wrapped layers
- Gradient tracking for LoRA weights only
- Optimizer integration with VarMap

## Potential Solutions

### Solution 1: Fork and Modify candle-transformers (Recommended)
1. Fork the candle-transformers repository
2. Modify `flux::model::Flux` to expose internal modules
3. Add methods to access and wrap attention layers
4. Implement a parameter registry

**Pros:**
- Full control over model structure
- Can add LoRA-specific APIs
- Maintains compatibility with existing code

**Cons:**
- Requires maintaining a fork
- Need to keep up with upstream changes

### Solution 2: Re-implement Flux Model
1. Create our own Flux implementation with exposed internals
2. Design with LoRA in mind from the start
3. Use candle primitives directly

**Pros:**
- Complete control over architecture
- Can optimize for training use case
- No external dependencies

**Cons:**
- Significant development effort
- Risk of implementation differences
- Need to validate against reference

### Solution 3: Dynamic Interception (Hacky)
1. Use unsafe Rust to access private fields
2. Runtime reflection/introspection
3. Monkey-patch the forward pass

**Pros:**
- Works with existing candle-transformers
- No fork needed

**Cons:**
- Fragile and unsafe
- May break with updates
- Poor performance

## Implementation Plan

### Phase 1: Model Analysis (2-3 days)
1. Examine candle-transformers Flux source code
2. Identify exact structure and layer names
3. Document parameter shapes and counts
4. Create a mapping to SimpleTuner's targets

### Phase 2: Fork and Modify (3-4 days)
1. Fork candle-transformers
2. Add parameter access methods to Flux model
3. Implement module registry
4. Test parameter extraction

### Phase 3: LoRA Integration (4-5 days)
1. Update `FluxDiffusionModel` wrapper to expose parameters
2. Implement LoRA layer wrapping mechanism
3. Create target module matching logic
4. Integrate with existing LoRA implementation

### Phase 4: Training Pipeline (3-4 days)
1. Update optimizer to use LoRA parameters
2. Implement proper gradient flow
3. Add checkpointing for LoRA weights
4. Test end-to-end training

### Phase 5: Validation (2-3 days)
1. Compare with SimpleTuner results
2. Verify LoRA weight shapes
3. Test inference with trained LoRA
4. Benchmark performance

## Code Examples

### What We Need to Implement:

```rust
// In a modified Flux model
impl FluxModel {
    pub fn named_modules(&self) -> HashMap<String, &dyn Module> {
        let mut modules = HashMap::new();
        
        // Expose double blocks
        for (i, block) in self.double_blocks.iter().enumerate() {
            modules.insert(format!("double_blocks.{}.to_q", i), &block.to_q);
            modules.insert(format!("double_blocks.{}.to_k", i), &block.to_k);
            modules.insert(format!("double_blocks.{}.to_v", i), &block.to_v);
            // ... etc
        }
        
        // Expose single blocks
        for (i, block) in self.single_blocks.iter().enumerate() {
            modules.insert(format!("single_blocks.{}.to_q", i), &block.to_q);
            // ... etc
        }
        
        modules
    }
    
    pub fn inject_lora(&mut self, config: &LoRAConfig) -> Result<()> {
        // Wrap target modules with LoRA layers
        for (name, module) in self.named_modules() {
            if config.target_modules.contains(&name) {
                // Replace with LoRA-wrapped version
                let lora_module = LoRAWrapper::new(module, config)?;
                self.replace_module(name, lora_module)?;
            }
        }
        Ok(())
    }
}
```

### Updated FluxDiffusionModel:

```rust
impl DiffusionModel for FluxDiffusionModel {
    fn trainable_parameters(&self) -> Vec<&Tensor> {
        // Collect parameters from LoRA layers only
        self.model.lora_parameters()
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        // All parameters including frozen base model
        self.model.all_parameters()
    }
}
```

## Immediate Next Steps

1. **Investigate candle-transformers source**: Understand exact Flux implementation
2. **Prototype parameter extraction**: Try to access model internals
3. **Design LoRA wrapper**: Create a clean abstraction for layer wrapping
4. **Create proof of concept**: Minimal example with one LoRA layer

## Risk Assessment

- **High Risk**: Candle's architecture may fundamentally prevent parameter access
- **Medium Risk**: Performance overhead from wrapper layers
- **Low Risk**: Compatibility with existing training code

## Conclusion

The main blocker is the lack of parameter access in the current Flux model wrapper. The recommended approach is to fork candle-transformers and modify it to expose the internal model structure needed for LoRA injection. This will require significant engineering effort but is the most robust solution.