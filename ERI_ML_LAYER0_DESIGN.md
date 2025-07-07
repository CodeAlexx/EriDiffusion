# Eri ML - Layer 0 Design

## Philosophy
Build ONLY what we need, when we need it. Start minimal, add as we hit real limitations.

## Layer 0: Core Tensor & Memory (Week 1)

### 0.1 Minimal Tensor Structure
```rust
// Start simple - just what we need for Flux/SD3.5
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    strides: Vec<usize>,
    device: Device,
    dtype: DType,
}

pub enum Storage {
    // Start with owned storage, add views later
    Cpu(Vec<u8>),
    Cuda(CudaBuffer),
}

pub enum DType {
    F32,    // Start with F32
    F16,    // For inference 
    BF16,   // For training
    I64,    // For indices
}
```

### 0.2 Essential Operations (What Flux/SD3.5 Actually Use)
```rust
// Week 1: Absolute minimums
impl Tensor {
    // Creation
    pub fn zeros(shape: &[usize], dtype: DType, device: &Device) -> Result<Self>;
    pub fn ones(shape: &[usize], dtype: DType, device: &Device) -> Result<Self>;
    pub fn randn(shape: &[usize], dtype: DType, device: &Device) -> Result<Self>;
    
    // Shape ops
    pub fn reshape(&self, shape: &[usize]) -> Result<Self>;
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self>;
    pub fn permute(&self, dims: &[usize]) -> Result<Self>;
    
    // Math ops (what models actually use)
    pub fn matmul(&self, other: &Self) -> Result<Self>;
    pub fn add(&self, other: &Self) -> Result<Self>;
    pub fn mul(&self, other: &Self) -> Result<Self>;
    
    // Reductions
    pub fn mean(&self, dims: &[usize]) -> Result<Self>;
    pub fn sum(&self, dims: &[usize]) -> Result<Self>;
}
```

### 0.3 Memory Management (Our Advantage)
```rust
// Integrate memory pooling from day 1
pub struct MemoryPool {
    pools: HashMap<(usize, Device), Vec<Allocation>>,
}

impl MemoryPool {
    // Get tensor from pool or allocate
    pub fn get_tensor(&mut self, shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor>;
    
    // Return tensor to pool
    pub fn return_tensor(&mut self, tensor: Tensor);
}

// Global pool - this is what makes us better than Candle
lazy_static! {
    static ref MEMORY_POOL: Mutex<MemoryPool> = Mutex::new(MemoryPool::new());
}
```

## Layer 1: Model Operations (Week 2)

### 1.1 Normalization (Critical for Flux/SD3.5)
```rust
// RMS Norm - Flux/SD3.5 use this extensively
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // Use our custom CUDA kernel from day 1
    match x.device() {
        Device::Cuda(_) => cuda_kernels::rms_norm(x, weight, eps),
        Device::Cpu => cpu_ops::rms_norm(x, weight, eps),
    }
}

// Layer Norm - transformer standard
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor>;

// Group Norm - VAE critical
pub fn group_norm(x: &Tensor, groups: usize, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor>;
```

### 1.2 Activations (What models actually use)
```rust
pub fn gelu(x: &Tensor) -> Result<Tensor>;
pub fn silu(x: &Tensor) -> Result<Tensor>;  // Also called swish
pub fn sigmoid(x: &Tensor) -> Result<Tensor>;
pub fn tanh(x: &Tensor) -> Result<Tensor>;
```

### 1.3 Attention Building Blocks
```rust
// Just the primitives - build attention on top
pub fn softmax(x: &Tensor, dim: i64) -> Result<Tensor>;
pub fn scaled_dot_product_attention(
    q: &Tensor, 
    k: &Tensor, 
    v: &Tensor,
    scale: Option<f32>
) -> Result<Tensor>;
```

## Layer 2: Weight Loading (Week 3)

### 2.1 Direct SafeTensors Integration
```rust
// No VarMap! Direct memory-mapped loading
pub struct WeightLoader {
    mmap: Mmap,
    metadata: HashMap<String, TensorInfo>,
}

impl WeightLoader {
    pub fn new(path: &Path) -> Result<Self>;
    
    // Zero-copy tensor creation
    pub fn get_tensor(&self, name: &str, device: &Device) -> Result<Tensor>;
    
    // With name resolution
    pub fn get_tensor_fuzzy(&self, name: &str, device: &Device) -> Result<Tensor> {
        // Try exact match
        // Try common variations (. -> /, etc)
        // Return best match
    }
}
```

### 2.2 Lazy Loading Pattern
```rust
// For huge models - only load what's needed
pub struct LazyTensor {
    loader: Arc<WeightLoader>,
    name: String,
    loaded: OnceCell<Tensor>,
}

impl LazyTensor {
    pub fn realize(&self, device: &Device) -> Result<&Tensor>;
}
```

## Layer 3: Module System (Week 4)

### 3.1 Simple Module Trait
```rust
pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

// Linear layer - most common
pub struct Linear {
    weight: LazyTensor,  // Lazy by default!
    bias: Option<LazyTensor>,
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.realize(x.device())?;
        let y = x.matmul(w)?;
        if let Some(b) = &self.bias {
            y.add(b.realize(x.device())?)
        } else {
            Ok(y)
        }
    }
}
```

### 3.2 Module Builders
```rust
// Clean API from the start
impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> LinearBuilder {
        LinearBuilder::new(in_features, out_features)
    }
}

pub struct LinearBuilder {
    // Builder pattern for all modules
}
```

## What We DON'T Build (Yet)

1. **Autograd** - Start with inference only
2. **Complex optimizers** - Just SGD/Adam when needed
3. **Distributed** - Single GPU first
4. **Quantization** - Add when we need it
5. **Graph optimization** - Manual fusion first

## Migration Strategy

### Week 1: Core Tensor Ops
- Copy minimal tensor code from Candle
- Add our memory pool
- Test with simple matmuls

### Week 2: Model Operations  
- Port our CUDA kernels (RMS norm, etc)
- Implement activations
- Test with single transformer block

### Week 3: Weight Loading
- Implement zero-copy SafeTensors
- Add fuzzy name matching
- Test loading Flux weights

### Week 4: Run Flux
- Build minimal modules needed
- Forward pass only
- Verify correctness vs Candle

### Week 5+: Add as Needed
- Training support when needed
- More ops as we hit them
- Optimization when bottlenecked

## Key Advantages Over Candle

1. **Memory pooling from day 1** - No more OOMs
2. **Zero-copy weight loading** - No VarMap duplication  
3. **Fuzzy tensor matching** - Loading "just works"
4. **Our CUDA kernels** - RMS norm that actually works
5. **Lazy everything** - Only materialize when needed

## Success Criteria

1. Load Flux weights in <2GB RAM (vs 12GB with Candle)
2. Forward pass matches Candle output
3. No CUDA kernel errors
4. Clear error messages
5. Extensible design for future needs

## Next Steps

1. Create `eri-ml` crate structure
2. Copy minimal tensor code from Candle
3. Integrate our memory pool
4. Port our CUDA kernels
5. Test with Flux

The key insight: We don't need all of Candle. We need about 20% of it, done RIGHT for our use case.