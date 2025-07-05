//! Benchmarks for inference operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use eridiffusion_core::{Device, tensor::TensorOps};
use eridiffusion_models::optimizations::*;
use candle_core::{Tensor, DType};

fn benchmark_attention_implementations(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");
    let device = Device::Cpu;
    
    let configs = vec![
        ("small", 1, 4, 64, 64),   // batch, heads, seq_len, head_dim
        ("medium", 2, 8, 128, 64),
        ("large", 4, 16, 256, 64),
    ];
    
    for (name, batch, heads, seq_len, head_dim) in configs {
        let q = Tensor::randn(0f32, 1f32, &[batch, heads, seq_len, head_dim], &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, &[batch, heads, seq_len, head_dim], &device).unwrap();
        let v = Tensor::randn(0f32, 1f32, &[batch, heads, seq_len, head_dim], &device).unwrap();
        
        // Benchmark standard attention
        group.bench_function(
            BenchmarkId::new("standard", name),
            |b| {
                b.iter(|| {
                    black_box(
                        q.scaled_dot_product_attention(&k, &v, None).unwrap()
                    );
                });
            }
        );
        
        // Benchmark flash attention
        group.bench_function(
            BenchmarkId::new("flash", name),
            |b| {
                b.iter(|| {
                    black_box(
                        AttentionOptimizer::flash_attention(
                            &q, &k, &v, (head_dim as f32).sqrt(), false
                        ).unwrap()
                    );
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_kv_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache");
    let device = Device::Cpu;
    
    let configs = vec![
        ("small", 512),
        ("medium", 1024),
        ("large", 2048),
    ];
    
    for (name, max_len) in configs {
        let mut cache = KVCache::new(max_len);
        
        // New tokens to add
        let new_len = 32;
        let keys = Tensor::randn(0f32, 1f32, &[1, 8, new_len, 64], &device).unwrap();
        let values = Tensor::randn(0f32, 1f32, &[1, 8, new_len, 64], &device).unwrap();
        
        group.bench_function(
            BenchmarkId::new("update", name),
            |b| {
                b.iter(|| {
                    black_box(
                        cache.update("layer0", &keys, &values).unwrap()
                    );
                    cache.reset(); // Reset for next iteration
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_ops");
    let device = Device::Cpu;
    
    let sizes = vec![
        ("small", 256),
        ("medium", 1024),
        ("large", 4096),
    ];
    
    for (name, size) in sizes {
        let tensor = Tensor::randn(0f32, 1f32, &[size, size], &device).unwrap();
        
        // Benchmark normalization
        group.bench_function(
            BenchmarkId::new("normalize", name),
            |b| {
                b.iter(|| {
                    black_box(tensor.normalize(1).unwrap());
                });
            }
        );
        
        // Benchmark cosine similarity
        let other = Tensor::randn(0f32, 1f32, &[size, size], &device).unwrap();
        group.bench_function(
            BenchmarkId::new("cosine_sim", name),
            |b| {
                b.iter(|| {
                    black_box(tensor.cosine_similarity(&other).unwrap());
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_fused_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_ops");
    let device = Device::Cpu;
    
    let sizes = vec![
        ("small", vec![64, 64]),
        ("medium", vec![256, 256]),
        ("large", vec![1024, 1024]),
    ];
    
    for (name, shape) in sizes {
        let tensor = Tensor::randn(0f32, 1f32, &shape, &device).unwrap();
        
        // Benchmark fused multiply-add
        group.bench_function(
            BenchmarkId::new("mul_add", name),
            |b| {
                b.iter(|| {
                    black_box(
                        FusedKernel::apply(&tensor, FusedOp::AddMulScalar(2.0, 1.0)).unwrap()
                    );
                });
            }
        );
        
        // Benchmark GELU
        group.bench_function(
            BenchmarkId::new("gelu", name),
            |b| {
                b.iter(|| {
                    black_box(
                        FusedKernel::apply(&tensor, FusedOp::Gelu).unwrap()
                    );
                });
            }
        );
        
        // Benchmark sequence of operations
        let ops = vec![
            FusedOp::AddMulScalar(0.5, 0.0),
            FusedOp::Gelu,
            FusedOp::Clamp(0.0, 1.0),
        ];
        
        group.bench_function(
            BenchmarkId::new("sequence", name),
            |b| {
                b.iter(|| {
                    black_box(
                        BatchedOps::apply_sequence(&tensor, &ops).unwrap()
                    );
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_memory_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool");
    let device = Device::Cpu;
    
    let allocation_sizes = vec![
        ("small", 1024),
        ("medium", 1024 * 1024),
        ("large", 10 * 1024 * 1024),
    ];
    
    for (name, size) in allocation_sizes {
        group.bench_function(
            BenchmarkId::new("allocate_deallocate", name),
            |b| {
                use eridiffusion_core::memory::{MemoryPool, MemoryPoolConfig};
                
                let config = MemoryPoolConfig {
                    initial_size: size * 2,
                    max_size: size * 10,
                    enable_defrag: false,
                    reuse_threshold: 0.8,
                };
                
                let pool = MemoryPool::new(config).unwrap();
                
                b.iter(|| {
                    let alloc = pool.allocate(size, candle_core::DType::F32).unwrap();
                    black_box(&alloc);
                    pool.deallocate(alloc).unwrap();
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_attention_implementations,
    benchmark_kv_cache,
    benchmark_tensor_operations,
    benchmark_fused_operations,
    benchmark_memory_pool
);
criterion_main!(benches);