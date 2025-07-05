# AI-Toolkit-RS Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Interface                         │
│                    (REST API + WebSocket)                    │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────┐
│                      Inference Engine                        │
│              (Schedulers, Optimizations, Pipeline)           │
└─────────────┬───────────────────────────────┬───────────────┘
              │                               │
┌─────────────┴───────────────┐ ┌────────────┴────────────────┐
│      Training System         │ │        Model System          │
│  (Trainer, Loss, Optimizer)  │ │   (All Diffusion Models)     │
└─────────────┬───────────────┘ └────────────┬────────────────┘
              │                               │
┌─────────────┴───────────────────────────────┴───────────────┐
│                     Network Adapters                         │
│           (LoRA, ControlNet, IP-Adapter, etc.)              │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                        Core Layer                            │
│     (Memory, Tensor Ops, Device, Validation, Errors)        │
└─────────────────────────────────────────────────────────────┘
```

## Crate Dependencies

```
web ──────────────┐
                  ↓
inference ────────┤
                  ↓
training ─────────┤
         ↘        ↓
          data ───┤
                  ↓
models ───────────┤
      ↘           ↓
       networks ──┤
                  ↓
extensions ───────┤
                  ↓
              core (foundation)
```

## Core Design Principles

### 1. **Zero-Copy Operations**
- Tensor views instead of clones where possible
- Shared memory pool for allocations
- Efficient data transfer between devices

### 2. **Type Safety**
- Strong typing prevents runtime errors
- Compile-time validation of tensor shapes
- Builder patterns for complex configurations

### 3. **Modularity**
- Each crate has a single responsibility
- Clean interfaces between modules
- Easy to extend without modifying core

### 4. **Performance First**
- Fused operations to reduce kernel launches
- Static dispatch for hot paths
- Memory pooling to reduce allocations
- Parallel data loading

## Key Components

### Core Crate
**Purpose**: Foundation for all other crates

**Key Types**:
```rust
pub struct Device;           // CPU/CUDA abstraction
pub struct MemoryPool;       // Allocation management
pub trait TensorOps;         // Extended tensor operations
pub struct ValidationError;  // Input validation
```

**Design Decisions**:
- Error types use `thiserror` for rich context
- Memory pool is thread-safe with Arc<Mutex<>>
- Validation is mandatory, not optional
- CUDA operations have CPU fallback

### Models Crate
**Purpose**: Diffusion model implementations

**Key Traits**:
```rust
pub trait DiffusionModel {
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput>;
    fn architecture(&self) -> ModelArchitecture;
}
```

**Model Hierarchy**:
- Base trait defines common interface
- Each model implements architecture-specific logic
- Auto-detection examines weight shapes
- State dict management is standardized

### Networks Crate
**Purpose**: Parameter-efficient adapters

**Key Design**:
- Adapters compose with base models
- Weight merging for deployment
- Dynamic module targeting
- Minimal memory overhead

### Training Crate
**Purpose**: Complete training loop

**Pipeline Architecture**:
```rust
pub trait TrainingPipeline {
    fn prepare_batch(&self, batch: &DataLoaderBatch) -> Result<PreparedBatch>;
    fn encode_prompts(&self, prompts: &[String]) -> Result<PromptEmbeds>;
    fn compute_loss(&self, ...) -> Result<Tensor>;
}
```

**Key Features**:
- Model-specific pipelines
- Gradient accumulation
- Mixed precision
- Distributed training

### Data Pipeline
**Design Goals**:
- Minimize CPU-GPU transfer
- Cache computed latents
- Dynamic batching
- Aspect ratio preservation

**Implementation**:
- Parallel data loading
- Pre-computed VAE encoding
- Smart bucketing algorithm
- Memory-mapped files

### Inference Engine
**Optimization Strategy**:
- Scheduler-specific optimizations
- Attention caching
- Operation fusion
- Dynamic shapes

## Memory Management

### Memory Pool Design
```rust
pub struct MemoryPool {
    allocations: HashMap<AllocationKey, MemoryBlock>,
    free_blocks: BTreeMap<usize, Vec<MemoryBlock>>,
    device_memory: HashMap<Device, DeviceMemory>,
}
```

**Allocation Strategy**:
1. Check free blocks for exact/larger size
2. Reuse if within threshold (80%)
3. Allocate new if needed
4. Defragment periodically

### Tensor Operations
**Fused Operations**:
- AddMulScalar: `x * scale + bias`
- Clamp: `min(max(x, low), high)`
- GELU: Fused activation
- LeakyReLU: Fused with slope

## Error Handling

### Error Hierarchy
```
Error
├── Model (model-specific errors)
├── Training (training errors)
├── Inference (inference errors)
├── Network (network/adapter errors)
├── Data (data loading errors)
├── Memory (allocation errors)
├── Device (CUDA/device errors)
├── Validation (input validation)
├── Plugin (extension errors)
└── Runtime (general runtime errors)
```

### Error Context
Every error includes:
- Source location
- Operation context
- Recovery hints
- Chain of causes

## Performance Optimizations

### 1. **Static Dispatch**
Replace virtual calls with enums:
```rust
pub enum ModelDispatch {
    SD15(SD15Model),
    SDXL(SDXLModel),
    // ...
}
```

### 2. **Tensor Fusion**
Combine multiple operations:
```rust
// Instead of:
let x = tensor.add(bias)?;
let x = x.mul(scale)?;
let x = x.clamp(0.0, 1.0)?;

// Use:
let x = tensor.apply_fused(&[
    FusedOp::AddScalar(bias),
    FusedOp::MulScalar(scale),
    FusedOp::Clamp(0.0, 1.0),
])?;
```

### 3. **Gradient Accumulation**
Pre-allocated buffers:
```rust
pub struct GradientAccumulator {
    accumulated_grads: Vec<Tensor>,
    accumulation_steps: usize,
    current_step: usize,
}
```

## Safety Guarantees

### Input Validation
- Tensor shape validation
- Finite value checks
- Range validation
- Path sanitization

### Memory Safety
- Bounds checking on views
- No buffer overflows
- RAII for resources
- Safe FFI wrappers

### Thread Safety
- Send + Sync where needed
- No data races
- Atomic operations
- Mutex protection

## Extension Points

### 1. **Custom Models**
Implement `DiffusionModel` trait

### 2. **Custom Networks**
Implement `NetworkAdapter` trait

### 3. **Custom Schedulers**
Implement `Scheduler` trait

### 4. **Plugins**
Use plugin API with sandboxing

## Future Architecture Considerations

### 1. **Sharding**
- Model parallel training
- Pipeline parallelism
- Tensor sharding

### 2. **Quantization**
- 8-bit/4-bit weights
- Dynamic quantization
- Quantization-aware training

### 3. **Compilation**
- Graph optimization
- Kernel fusion
- AOT compilation

### 4. **Distributed**
- Multi-node training
- Gradient compression
- Communication optimization