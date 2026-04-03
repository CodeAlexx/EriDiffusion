#[cfg(all(feature = "cuda-raw", feature = "alloc-debug"))]
#[test]
fn allocator_debug_poison_runs() {
    use eridiffusion_core::memory_pools;
    use eridiffusion_core::Device;
    use flame_core::DType as FlameDType;
    use flame_core::Shape;

    // Just ensure allocation + release does not panic and triggers debug path
    let device = Device::Cuda(0);
    let pool = memory_pools().get_pool(&device);
    let shape = Shape::from_dims(&[4, 4]);
    let t = pool.allocate_tensor(&shape, FlameDType::F32).expect("alloc");
    pool.release_tensor(&t).expect("release");
}

