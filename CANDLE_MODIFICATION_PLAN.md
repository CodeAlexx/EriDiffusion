# Candle Framework Modification Plan

## Executive Summary
Transform Candle from a rigid, memory-inefficient framework into a flexible, production-ready ML framework that works WITH us, not against us. This plan outlines fundamental changes to make Candle useable for real-world diffusion model training on 24GB GPUs.

## Core Philosophy
1. **Memory First**: Every operation should consider memory efficiency
2. **Flexibility Over Purity**: Practical solutions over theoretical elegance
3. **Developer Experience**: Clear errors, helpful suggestions, intuitive APIs
4. **Zero-Copy Operations**: Minimize memory movement and duplication
5. **Progressive Enhancement**: Start with core changes, build up

## Phase 1: Fundamental Memory Management (Week 1-2)

### 1.1 Replace VarMap with Zero-Copy Storage
```rust
// Current: HashMap<String, Var> duplicates everything
// New: Direct tensor views with lazy loading
pub struct TensorStore {
    // Memory-mapped safetensors file
    mmap: Mmap,
    // Tensor metadata without data duplication
    metadata: HashMap<String, TensorInfo>,
    // LRU cache for frequently accessed tensors
    cache: LruCache<String, Arc<Tensor>>,
}
```

### 1.2 Implement Memory Pool at Storage Level
```rust
// Integrate memory pooling directly into candle_core::Storage
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    // New: Pooled storage that reuses allocations
    CpuPooled(PooledStorage<CpuBackend>),
    CudaPooled(PooledStorage<CudaBackend>),
}
```

### 1.3 Add Memory Pressure Callbacks
```rust
// Allow registration of memory pressure handlers
pub trait MemoryPressureHandler {
    fn on_low_memory(&mut self, available: usize, required: usize) -> MemoryAction;
}

pub enum MemoryAction {
    FreeCache,
    OffloadToHost,
    Defragment,
    Fail,
}
```

## Phase 2: Flexible Tensor Operations (Week 3-4)

### 2.1 Smart Tensor Name Resolution
```rust
pub trait TensorResolver {
    // Try multiple strategies to find tensor
    fn resolve(&self, requested: &str) -> Option<String>;
    
    // Built-in strategies:
    // - Exact match
    // - Common variations (. vs /, underscores, etc.)
    // - Fuzzy matching with edit distance
    // - User-defined mappings
}

impl VarBuilder {
    pub fn with_resolver(mut self, resolver: impl TensorResolver) -> Self {
        self.resolver = Some(Box::new(resolver));
        self
    }
}
```

### 2.2 Partial Tensor Loading with Transformation
```rust
pub trait TensorTransform {
    fn can_transform(&self, from: &Shape, to: &Shape) -> bool;
    fn transform(&self, tensor: &Tensor, target_shape: &Shape) -> Result<Tensor>;
}

// Examples:
// - Reshape compatible shapes
// - Pad/crop mismatched dimensions
// - Split/merge for architectural differences
// - Type casting with validation
```

### 2.3 Lazy Tensor Initialization
```rust
pub struct LazyTensor {
    init_fn: Box<dyn FnOnce() -> Result<Tensor>>,
    cached: OnceCell<Tensor>,
}

// Only materialize when actually used
impl Module for LazyLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = self.weight.get_or_init()?;
        // ...
    }
}
```

## Phase 3: Better Error Handling (Week 5)

### 3.1 Context-Aware Errors
```rust
#[derive(Debug)]
pub struct CangleError {
    kind: ErrorKind,
    context: ErrorContext,
    suggestions: Vec<String>,
    stack: Vec<String>,
}

pub struct ErrorContext {
    operation: String,
    tensor_name: Option<String>,
    shapes: Option<(Shape, Shape)>,
    device: Device,
    available_memory: Option<usize>,
}

// Example output:
// Error: Shape mismatch in matmul operation
// Expected: [1024, 768] × [768, 512]
// Got: [1024, 768] × [512, 512]
// 
// Suggestions:
// - Check if weight tensor needs transposition
// - Verify model architecture matches checkpoint
// - Common fix: tensor.transpose(0, 1)?
```

### 3.2 Operation Tracing
```rust
// Optional operation tracking for debugging
pub struct OpTrace {
    enabled: bool,
    ops: Vec<OpInfo>,
}

// Automatically track tensor operations when enabled
// Useful for finding where things go wrong
```

## Phase 4: CUDA Kernel Management (Week 6)

### 4.1 Automatic Kernel Compilation
```rust
pub struct KernelManager {
    kernels: HashMap<String, CudaModule>,
    compile_cache: PathBuf,
}

impl KernelManager {
    // Auto-compile and cache kernels
    pub fn get_or_compile(&mut self, name: &str, source: &str) -> Result<&CudaModule>;
    
    // Version checking and recompilation
    pub fn validate_cuda_version(&self) -> Result<()>;
}
```

### 4.2 Kernel Fusion Opportunities
```rust
// Detect and fuse common operation patterns
pub trait KernelFusion {
    fn can_fuse(&self, ops: &[Op]) -> bool;
    fn fuse(&self, ops: &[Op]) -> Result<FusedOp>;
}

// Examples:
// - LayerNorm + Linear
// - Multiple elementwise ops
// - Attention QKV projection
```

## Phase 5: Device Management (Week 7)

### 5.1 Unified Device Abstraction
```rust
pub trait DeviceOps {
    fn allocate(&mut self, size: usize) -> Result<*mut u8>;
    fn deallocate(&mut self, ptr: *mut u8);
    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()>;
    fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
}

// Consistent interface across all backends
```

### 5.2 Multi-Device Coordination
```rust
pub struct DeviceManager {
    devices: Vec<Box<dyn DeviceOps>>,
    memory_tracker: MemoryTracker,
    stream_pool: StreamPool,
}

// Automatic device selection based on memory availability
// Coordinated transfers between devices
```

## Phase 6: Module System Enhancement (Week 8)

### 6.1 Stateful Modules with Memory Awareness
```rust
pub trait ModuleExt: Module {
    fn memory_usage(&self) -> MemoryStats;
    fn can_offload(&self) -> bool;
    fn offload(&mut self) -> Result<()>;
    fn reload(&mut self) -> Result<()>;
}
```

### 6.2 Dynamic Module Composition
```rust
// Build modules programmatically
pub struct ModuleBuilder {
    layers: Vec<Box<dyn Module>>,
}

impl ModuleBuilder {
    pub fn add_if<F>(&mut self, condition: bool, f: F) -> &mut Self 
    where F: FnOnce() -> Box<dyn Module>;
    
    pub fn replace_layer(&mut self, index: usize, layer: Box<dyn Module>);
}
```

## Phase 7: Integration Improvements (Week 9)

### 7.1 Native SafeTensors Integration
```rust
// Direct integration without intermediate conversions
impl SafeTensors {
    pub fn as_var_builder(&self, device: &Device) -> Result<VarBuilder>;
    pub fn lazy_load(&self) -> Result<LazyVarBuilder>;
}
```

### 7.2 Checkpoint Compatibility Layer
```rust
pub trait CheckpointAdapter {
    fn adapt_state_dict(&self, state_dict: StateDict) -> Result<StateDict>;
    fn get_architecture(&self) -> ModelArchitecture;
}

// Pre-built adapters for common conversions:
// - PyTorch -> Candle
// - HuggingFace -> Candle  
// - ONNX -> Candle
```

## Phase 8: Developer Experience (Week 10)

### 8.1 Builder Pattern APIs
```rust
// Intuitive model construction
let model = FluxModel::builder()
    .hidden_size(3072)
    .num_layers(24)
    .attention_heads(24)
    .with_lora(LoraConfig::default())
    .memory_efficient(true)
    .build()?;
```

### 8.2 Debugging Tools
```rust
// Built-in debugging utilities
pub mod debug {
    pub fn check_tensor_stats(t: &Tensor) -> TensorStats;
    pub fn visualize_computation_graph(model: &dyn Module);
    pub fn profile_memory_usage<F>(f: F) -> MemoryProfile;
    pub fn trace_allocations<F>(f: F) -> Vec<AllocationEvent>;
}
```

## Implementation Strategy

### Priority 1: Core Memory Changes (Must Have)
1. Zero-copy VarMap replacement
2. Memory pooling integration  
3. Smart tensor name resolution
4. Better error messages

### Priority 2: Usability (Should Have)
1. Partial tensor loading
2. Lazy initialization
3. Builder APIs
4. Native safetensors support

### Priority 3: Performance (Nice to Have)
1. Kernel fusion
2. Multi-device coordination
3. Advanced memory strategies

## Testing Strategy
1. Maintain backward compatibility where possible
2. Extensive unit tests for each component
3. Integration tests with real models (Flux, SD3.5, etc.)
4. Memory leak detection
5. Performance benchmarks

## Migration Path
1. New features opt-in via feature flags
2. Deprecation warnings for changed APIs
3. Migration guide with examples
4. Automated migration tool for common patterns

## Success Metrics
1. Load Flux model in <5GB RAM (currently ~12GB)
2. Zero memory duplication in VarMap
3. 50% reduction in OOM errors
4. 90% of tensor loading "just works"
5. Clear, actionable error messages
6. 2x faster model initialization

## Risks and Mitigations
1. **Risk**: Breaking existing code
   - **Mitigation**: Feature flags, compatibility layer
   
2. **Risk**: Performance regression
   - **Mitigation**: Comprehensive benchmarks, profiling
   
3. **Risk**: Increased complexity
   - **Mitigation**: Clean abstractions, good documentation

## Timeline
- Week 1-2: Core memory management
- Week 3-4: Flexible tensor operations  
- Week 5: Error handling
- Week 6: CUDA kernel management
- Week 7: Device management
- Week 8: Module system
- Week 9: Integration improvements
- Week 10: Developer experience
- Week 11-12: Testing and refinement

## Conclusion
These modifications will transform Candle from a framework we fight against into one that actively helps us succeed. The focus on memory efficiency, flexibility, and developer experience will make it suitable for production ML workloads on consumer hardware.

The key insight is that we need to modify Candle at its core - not work around its limitations with external systems. By integrating our proven solutions (memory pooling, smart loading, etc.) directly into Candle, we create a framework that understands the realities of training large models on limited hardware.