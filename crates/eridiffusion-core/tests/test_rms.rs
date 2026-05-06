use flame_core::{DType, Shape, Tensor};

#[test]
fn verify_rms_norm() {
    let d = flame_core::global_cuda_device();
    flame_core::config::set_default_dtype(DType::BF16);
    
    // Test: values of 100 should normalize to ~1/sqrt(eps) with default epsilon
    let x = Tensor::from_vec(vec![100.0; 4096], Shape::from_dims(&[1, 1, 4096]), d.clone()).unwrap();
    let n = flame_core::norm::rms_norm(&x, &[4096], None::<&Tensor>, 1e-6).unwrap();
    let vals = n.to_vec().unwrap();
    let rms = (vals.iter().map(|x| x*x).sum::<f32>() / vals.len() as f32).sqrt();
    println!("Input=100.0, RMS after norm: {:.6} (should be ~1.0)", rms);
    assert!(rms > 0.5 && rms < 2.0, "RMSNorm failed to normalize");
}
