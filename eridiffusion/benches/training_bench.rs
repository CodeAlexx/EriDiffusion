//! Benchmarks for training operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use eridiffusion_core::{Device, DType as CoreDType};
use eridiffusion_training::{loss::*, optimizer::*, gradient_accumulator::*};
use candle_core::{Tensor, DType};

fn benchmark_loss_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_functions");
    let device = Device::Cpu;
    
    // Different tensor sizes to benchmark
    let sizes = vec![(32, 32), (128, 128), (512, 512)];
    
    for (h, w) in sizes {
        let pred = Tensor::randn(0f32, 1f32, &[1, 3, h, w], &device).unwrap();
        let target = Tensor::randn(0f32, 1f32, &[1, 3, h, w], &device).unwrap();
        
        let config = LossConfig {
            weight: 1.0,
            reduction: ReductionType::Mean,
        };
        
        // Benchmark MSE Loss
        group.bench_with_input(
            BenchmarkId::new("MSE", format!("{}x{}", h, w)),
            &(&pred, &target),
            |b, (pred, target)| {
                let loss = MSELoss::new(config).unwrap();
                b.iter(|| {
                    black_box(loss.compute(pred, target).unwrap())
                });
            }
        );
        
        // Benchmark MAE Loss
        group.bench_with_input(
            BenchmarkId::new("MAE", format!("{}x{}", h, w)),
            &(&pred, &target),
            |b, (pred, target)| {
                let loss = MAELoss::new(config).unwrap();
                b.iter(|| {
                    black_box(loss.compute(pred, target).unwrap())
                });
            }
        );
        
        // Benchmark Huber Loss
        group.bench_with_input(
            BenchmarkId::new("Huber", format!("{}x{}", h, w)),
            &(&pred, &target),
            |b, (pred, target)| {
                let loss = HuberLoss::new(config, 1.0).unwrap();
                b.iter(|| {
                    black_box(loss.compute(pred, target).unwrap())
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_gradient_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_accumulation");
    let device = Device::Cpu;
    
    let param_sizes = vec![
        ("small", vec![512, 512]),
        ("medium", vec![2048, 2048]),
        ("large", vec![4096, 4096]),
    ];
    
    for (name, shape) in param_sizes {
        let param = Tensor::randn(0f32, 1f32, &shape, &device).unwrap();
        let params = vec![&param];
        
        group.bench_function(
            BenchmarkId::new("accumulate", name),
            |b| {
                let mut accumulator = GradientAccumulator::new(4);
                accumulator.initialize(&params).unwrap();
                
                let grad = Tensor::randn(0f32, 1f32, &shape, &device).unwrap();
                
                b.iter(|| {
                    accumulator.accumulate(&[grad.clone()]).unwrap();
                    if accumulator.is_ready() {
                        black_box(accumulator.get_gradients().unwrap());
                    }
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_gradient_clipping(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_clipping");
    let device = Device::Cpu;
    
    let num_params = vec![10, 50, 100];
    
    for n in num_params {
        let mut gradients: Vec<Tensor> = Vec::new();
        for _ in 0..n {
            gradients.push(Tensor::randn(0f32, 1f32, &[128, 128], &device).unwrap());
        }
        
        group.bench_function(
            BenchmarkId::new("global_norm", n),
            |b| {
                let clipper = GradientClipper::new(1.0, ClipMethod::GlobalNorm);
                
                b.iter(|| {
                    let mut grads_clone = gradients.clone();
                    black_box(clipper.clip(&mut grads_clone).unwrap());
                });
            }
        );
        
        group.bench_function(
            BenchmarkId::new("by_value", n),
            |b| {
                let clipper = GradientClipper::new(1.0, ClipMethod::Value);
                
                b.iter(|| {
                    let mut grads_clone = gradients.clone();
                    black_box(clipper.clip(&mut grads_clone).unwrap());
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_optimizer_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_step");
    let device = Device::Cpu;
    
    // Create parameters and gradients
    let param_shapes = vec![
        ("small", vec![128, 128]),
        ("medium", vec![512, 512]),
        ("large", vec![1024, 1024]),
    ];
    
    for (name, shape) in param_shapes {
        let params: Vec<Tensor> = (0..10)
            .map(|_| Tensor::randn(0f32, 1f32, &shape, &device).unwrap())
            .collect();
        let param_refs: Vec<&Tensor> = params.iter().collect();
        
        let grads: Vec<Tensor> = (0..10)
            .map(|_| Tensor::randn(0f32, 0.01f32, &shape, &device).unwrap())
            .collect();
        
        let config = OptimizerConfig {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        
        group.bench_function(
            BenchmarkId::new("adam", name),
            |b| {
                let mut optimizer = AdamOptimizer::new(config, &param_refs).unwrap();
                
                b.iter(|| {
                    black_box(optimizer.step(&param_refs, &grads, config.learning_rate).unwrap());
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_loss_functions,
    benchmark_gradient_accumulation,
    benchmark_gradient_clipping,
    benchmark_optimizer_step
);
criterion_main!(benches);