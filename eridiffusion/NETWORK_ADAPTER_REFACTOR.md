# Network Adapter Trait Refactoring Documentation

## Overview
This document describes the refactoring of the `NetworkAdapter` trait to resolve reference lifetime issues when accessing parameters stored behind locks.

## Problem Statement
The original trait signature was:
```rust
fn parameters(&self) -> HashMap<String, &Tensor>;
```

This caused issues because:
1. Network adapters store their state behind `RwLock` for thread safety
2. Returning references to tensors inside a lock guard would create lifetime conflicts
3. The references couldn't outlive the lock guard, making the API unusable

## Solution
Changed the trait signature to return owned tensors:
```rust
fn parameters(&self) -> HashMap<String, Tensor>;
```

This allows implementations to clone tensors when needed, avoiding lifetime issues.

## Changes Made

### 1. Core Trait Update
**File**: `crates/core/src/network.rs`
```rust
// Before
fn parameters(&self) -> HashMap<String, &Tensor>;

// After
fn parameters(&self) -> HashMap<String, Tensor>;
```

### 2. Network Adapter Implementations

#### LoRA Adapter
**File**: `crates/networks/src/lora.rs`
```rust
fn parameters(&self) -> HashMap<String, Tensor> {
    let state = self.state.read();
    let mut params = HashMap::new();
    
    for (name, layer) in &state.layers {
        // Clone LoRA A and B weights
        params.insert(
            format!("{}.lora_a.weight", name), 
            layer.lora_a.weight().clone()
        );
        params.insert(
            format!("{}.lora_b.weight", name), 
            layer.lora_b.weight().clone()
        );
        
        // Clone biases if present
        if let Some(bias) = layer.lora_a.bias() {
            params.insert(format!("{}.lora_a.bias", name), bias.clone());
        }
        if let Some(bias) = layer.lora_b.bias() {
            params.insert(format!("{}.lora_b.bias", name), bias.clone());
        }
    }
    
    params
}
```

#### DoRA Adapter
**File**: `crates/networks/src/dora.rs`
```rust
fn parameters(&self) -> HashMap<String, Tensor> {
    let state = self.state.read();
    let mut params = HashMap::new();
    
    for (name, layer) in &state.layers {
        // Clone all DoRA parameters
        params.insert(format!("{}.lora_a.weight", name), layer.lora_layer.lora_a.weight().clone());
        params.insert(format!("{}.lora_b.weight", name), layer.lora_layer.lora_b.weight().clone());
        params.insert(format!("{}.magnitude", name), layer.magnitude.clone());
        
        // Clone biases if present
        if let Some(bias) = layer.lora_layer.lora_a.bias() {
            params.insert(format!("{}.lora_a.bias", name), bias.clone());
        }
        if let Some(bias) = layer.lora_layer.lora_b.bias() {
            params.insert(format!("{}.lora_b.bias", name), bias.clone());
        }
    }
    
    params
}
```

#### LoKr Adapter
**File**: `crates/networks/src/lokr.rs`
```rust
fn parameters(&self) -> HashMap<String, Tensor> {
    let state = self.state.read();
    let mut params = HashMap::new();
    
    for (name, layer) in &state.layers {
        // Clone W1 and W2 tensors
        params.insert(format!("{}.w1", name), layer.w1.clone());
        params.insert(format!("{}.w2", name), layer.w2.clone());
        
        // Clone mid layer if CP decomposition
        if let Some(ref mid) = layer.w1_b {
            params.insert(format!("{}.w1_b", name), mid.clone());
        }
        if let Some(ref mid) = layer.w2_b {
            params.insert(format!("{}.w2_b", name), mid.clone());
        }
    }
    
    params
}
```

#### LoCoN Adapter
**File**: `crates/networks/src/locon.rs`
```rust
fn parameters(&self) -> HashMap<String, Tensor> {
    let state = self.state.read();
    let mut params = HashMap::new();
    
    for (name, layer) in &state.layers {
        // Clone convolutional LoRA weights
        params.insert(format!("{}.lora_a.weight", name), layer.lora_a.weight().clone());
        params.insert(format!("{}.lora_b.weight", name), layer.lora_b.weight().clone());
        
        // Clone mid layer if using CP decomposition
        if let Some(ref mid) = layer.lora_mid {
            params.insert(format!("{}.lora_mid.weight", name), mid.weight().clone());
        }
    }
    
    params
}
```

#### GLoRA Adapter
**File**: `crates/networks/src/glora.rs`
```rust
fn parameters(&self) -> HashMap<String, Tensor> {
    let state = self.state.read();
    let mut params = HashMap::new();
    
    for (name, layer) in &state.layers {
        // Clone all GLoRA parameters
        params.insert(format!("{}.lora_a.weight", name), layer.lora_a.weight().clone());
        params.insert(format!("{}.lora_b.weight", name), layer.lora_b.weight().clone());
        
        // Clone norm, gate, and gating weights
        params.insert(format!("{}.norm_weight", name), layer.norm_weight.clone());
        params.insert(format!("{}.norm_bias", name), layer.norm_bias.clone());
        params.insert(format!("{}.gate_weight", name), layer.gate_weight.clone());
        params.insert(format!("{}.gate_bias", name), layer.gate_bias.clone());
        params.insert(format!("{}.gating_weight", name), layer.gating_weight.clone());
    }
    
    params
}
```

#### Stacked Adapter
**File**: `crates/networks/src/utils.rs`
```rust
fn parameters(&self) -> HashMap<String, Tensor> {
    let mut all_params = HashMap::new();
    
    // Collect parameters from all adapters
    for (i, adapter) in self.adapters.iter().enumerate() {
        let params = adapter.parameters();
        for (key, tensor) in params {
            // Prefix with adapter index to avoid conflicts
            all_params.insert(format!("adapter_{}.{}", i, key), tensor);
        }
    }
    
    all_params
}
```

### 3. Additional Fixes

#### Error Enum Updates
- Changed `Error::NetworkAdapter` to `Error::Network` to match the actual enum definition

#### Conv2d Configuration
- Added missing `cudnn_fwd_algo: None` field to all `Conv2dConfig` initializations

#### Tensor Operations
- Fixed scalar addition operations by creating scalar tensors
- Fixed tensor indexing in kronecker product implementation
- Added proper error handling with `Ok()` wrappers

#### Async/Lifetime Issues
- Moved lock acquisitions to proper scopes before `await` points
- Fixed borrowing issues by cloning or restructuring code

## Impact

### Performance
- Cloning tensors adds overhead, but it's negligible compared to actual computation
- Most use cases (saving/loading checkpoints) already require owned data

### API Consistency
- All network adapters now have consistent parameter access patterns
- No more lifetime issues when working with parameters
- Easier to compose and stack adapters

### Future Improvements
1. Consider adding a `parameters_ref(&self) -> impl Iterator<Item = (&str, &Tensor)>` method for read-only access
2. Add parameter caching to avoid repeated clones
3. Implement Copy-on-Write (CoW) tensors to reduce cloning overhead

## Testing
The refactored code compiles successfully with no errors. All network adapters can now properly expose their parameters for:
- Checkpoint saving/loading
- Weight visualization
- Gradient updates during training
- Parameter statistics and analysis