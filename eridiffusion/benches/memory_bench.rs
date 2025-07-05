use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use eridiffusion_core::{Device, memory::*};
use candle_core::{Shape, DType as CandleDType};

fn benchmark_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    // Test different tensor sizes
    let sizes = vec![
        (32, 32),      // 1K elements
        (256, 256),    // 64K elements
        (1024, 1024),  // 1M elements
        (4096, 4096),  // 16M elements
    ];
    
    let device = Device::Cpu;
    let pool = MemoryPool::new(device, MemoryPoolConfig::default());
    
    for (height, width) in sizes {
        let size = height * width;
        group.bench_with_input(
            BenchmarkId::new("pool_allocation", size),
            &(height, width),
            |b, &(h, w)| {
                b.iter(|| {
                    let shape = Shape::from_dims(&[h, w]);
                    let tensor = pool.allocate_tensor(&shape, CandleDType::F32).unwrap();
                    pool.release_tensor(&tensor).unwrap();
                    black_box(tensor);
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_memory_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_reuse");
    
    let device = Device::Cpu;
    let pool = MemoryPool::new(device, MemoryPoolConfig::default());
    let shape = Shape::from_dims(&[1024, 1024]);
    
    // Pre-allocate and release to populate the pool
    for _ in 0..10 {
        let tensor = pool.allocate_tensor(&shape, CandleDType::F32).unwrap();
        pool.release_tensor(&tensor).unwrap();
    }
    
    group.bench_function("reuse_from_pool", |b| {
        b.iter(|| {
            let tensor = pool.allocate_tensor(&shape, CandleDType::F32).unwrap();
            pool.release_tensor(&tensor).unwrap();
            black_box(tensor);
        });
    });
    
    group.bench_function("new_allocation", |b| {
        b.iter(|| {
            let tensor = candle_core::Tensor::zeros(
                &shape,
                CandleDType::F32,
                &candle_core::Device::Cpu,
            ).unwrap();
            black_box(tensor);
        });
    });
    
    group.finish();
}

fn benchmark_tensor_views(c: &mut Criterion) {
    use eridiffusion_core::tensor::TensorView;
    
    let mut group = c.benchmark_group("tensor_views");
    
    // Create a large tensor
    let data: Vec<f32> = (0..1_048_576).map(|i| i as f32).collect();
    let tensor = candle_core::Tensor::from_vec(
        data,
        &[1024, 1024],
        &candle_core::Device::Cpu,
    ).unwrap();
    
    let view = TensorView::new(tensor.clone());
    
    group.bench_function("view_slice", |b| {
        b.iter(|| {
            let sliced = view.slice(&[0..512, 0..512]).unwrap();
            black_box(sliced);
        });
    });
    
    group.bench_function("tensor_slice", |b| {
        b.iter(|| {
            let sliced = tensor.narrow(0, 0, 512).unwrap()
                .narrow(1, 0, 512).unwrap();
            black_box(sliced);
        });
    });
    
    group.bench_function("view_reshape", |b| {
        b.iter(|| {
            let reshaped = view.reshape(&[2048, 512]).unwrap();
            black_box(reshaped);
        });
    });
    
    group.finish();
}

fn benchmark_memory_manager(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_manager");
    
    let manager = memory_pools();
    let device = Device::Cpu;
    
    group.bench_function("get_pool", |b| {
        b.iter(|| {
            let pool = manager.get_pool(&device);
            black_box(pool);
        });
    });
    
    // Benchmark concurrent access
    use std::sync::Arc;
    use std::thread;
    
    let manager = Arc::new(MemoryPoolManager::new(MemoryPoolConfig::default()));
    
    group.bench_function("concurrent_allocation", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let manager = manager.clone();
                    thread::spawn(move || {
                        let pool = manager.get_pool(&Device::Cpu);
                        let shape = Shape::from_dims(&[256, 256]);
                        for _ in 0..10 {
                            let tensor = pool.allocate_tensor(&shape, CandleDType::F32).unwrap();
                            pool.release_tensor(&tensor).unwrap();
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_memory_allocation,
    benchmark_memory_reuse,
    benchmark_tensor_views,
    benchmark_memory_manager
);
criterion_main!(benches);