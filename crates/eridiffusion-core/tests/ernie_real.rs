use std::path::Path;
use flame_core::{self, autograd::AutogradContext, DType, Shape, Tensor};
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::models::ErnieModel;

#[test]
fn ernie_real_latents() {
    let device = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);

    let mut cfg = TrainConfig::default();
    cfg.training_method = TrainingMethod::Lora;
    cfg.lora_rank = 16;

    let paths = vec![
        std::path::PathBuf::from("/home/alex/models/ERNIE-Image/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"),
        std::path::PathBuf::from("/home/alex/models/ERNIE-Image/transformer/diffusion_pytorch_model-00002-of-00002.safetensors"),
    ];
    let model = ErnieModel::load(&paths, &cfg, device.clone()).unwrap();

    // Load real VAE latent + zeros text
    let cache_path = Path::new("/tmp/ernie_cache/s_0000.safetensors");
    let cached = flame_core::serialization::load_file(cache_path, &device).unwrap();
    let latent = cached.get("latent").unwrap().to_dtype(DType::BF16).unwrap();
    println!("Loaded latent: {:?}", latent.shape().dims());

    let t = 0.5;
    let noise = Tensor::randn(latent.shape().clone(), 0.0, 0.01, device.clone()).unwrap();
    let noisy = latent.mul_scalar(1.0 - t).unwrap().add(&noise.mul_scalar(t).unwrap()).unwrap();
    let txt = Tensor::zeros(Shape::from_dims(&[1, 77, 3072]), device.clone()).unwrap();
    let timestep = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone()).unwrap();

    let noisy = noisy.requires_grad_(true);
    let pred = model.forward(&noisy, &txt, &timestep).unwrap();
    println!("Output: {:?}", pred.shape().dims());

    let target = noise.sub(&latent).unwrap();
    let diff = pred.sub(&target).unwrap();
    let loss = diff.square().unwrap().mean().unwrap();
    let loss_val = loss.to_dtype(DType::F32).unwrap().to_vec().unwrap()[0];
    println!("Loss: {:.6}", loss_val);
    assert!(loss_val.is_finite() && loss_val < 10000.0, "Loss bad: {}", loss_val);

    let grads = loss.backward().unwrap();
    let mut nan = 0;
    for (_, g) in grads.iter() {
        for x in g.to_dtype(DType::F32).unwrap().to_vec().unwrap() {
            if x.is_nan() { nan += 1; }
        }
    }
    println!("Grads: {} total, {} NaN", grads.len(), nan);
    assert_eq!(nan, 0, "NaN grads!");
    println!("PASS: real VAE latent, loss={:.4}", loss_val);
    AutogradContext::clear();
}
