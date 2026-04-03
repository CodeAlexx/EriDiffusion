#[cfg(test)]
mod tests {
    use super::super::SdxlUnet;
    use flame_core::{Device, Tensor, Shape, DType};
    use eridiffusion_common_weights::SafeLoader;

    #[test]
    fn eps_contract() {
        let dev = Device::cuda(0).unwrap();
        let cfg_yaml = serde_yaml::to_value(serde_yaml::from_str::<serde_yaml::Value>("prediction_type: epsilon\nctx1_dim: 768\nctx2_dim: 1280\nseq: 64\nin_ch: 4\nbase_ch: 320\nch_mult: [1,2,4,4]\nnum_heads: 20\nhead_dim: 64").unwrap()).unwrap();
        // Loader not used beyond guard; create an empty file scenario by mocking? For now, skip guard by using a tiny dummy loader
        // This test focuses on eps shape.
        let b=2usize; let h=32usize; let w=32usize;
        let lat = Tensor::zeros_dtype(Shape::from(vec![b,h,w,4]), DType::BF16, dev.cuda_device().clone()).unwrap();
        let t = Tensor::zeros_dtype(Shape::from(vec![b]), DType::F32, dev.cuda_device().clone()).unwrap();
        let ctx1 = Tensor::zeros_dtype(Shape::from(vec![b,64,768]), DType::BF16, dev.cuda_device().clone()).unwrap();
        let ctx2 = Tensor::zeros_dtype(Shape::from(vec![b,64,1280]), DType::BF16, dev.cuda_device().clone()).unwrap();
        let lengths = Tensor::from_vec(vec![64.0, 64.0], Shape::from(vec![b]), dev.cuda_device().clone()).unwrap().to_dtype(DType::I32).unwrap();
        let unet = SdxlUnet { cfg: crate::config::SdxlConfig::default(), device: dev.clone() };
        let out = unet.eps(&lat, &t, &ctx1, &ctx2, &lengths).unwrap();
        assert_eq!(out.shape().dims(), &[b,h,w,4]);
        assert_eq!(out.dtype(), DType::BF16);
    }
}
