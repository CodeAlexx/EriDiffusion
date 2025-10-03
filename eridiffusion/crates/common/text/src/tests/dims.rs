use anyhow::Result;
use flame_core::{Device, Tensor, Shape, DType};
use eridiffusion_common_text::{HfTokenizer, clip_l::ClipL, openclip_g::OpenClipG};
use eridiffusion_common_weights::write_safetensors;

#[test]
fn forward_shapes_nonzero() -> Result<()> {
    let dev = Device::cuda(0)?;
    let tmp = tempfile::tempdir()?;
    // Create a tiny embedding weight [vocab, dim]
    let vocab = 100usize; let dim = 32usize; let seq=16usize; let b=2usize;
    let emb_data: Vec<f32> = (0..vocab*dim).map(|i| ((i%113) as f32)/113.0 - 0.5).collect();
    let emb = Tensor::from_vec(emb_data, Shape::from(vec![vocab, dim]), dev.cuda_device().clone())?
        .to_dtype(DType::BF16)?;
    let path = tmp.path().join("te.safetensors");
    write_safetensors(&path, &[("emb.weight".into(), emb)])?;

    // Fake ids
    let ids = Tensor::from_vec(vec![1.0,2.0,3.0,4.0, 5.0,6.0,7.0,8.0,
                                    1.0,1.0,1.0,1.0, 2.0,2.0,2.0,2.0],
                               Shape::from(vec![b,seq]), dev.cuda_device().clone())?;

    let te1 = ClipL::from_weights_auto(path.to_str().unwrap(), &dev, seq)?;
    let out1 = te1.forward(&ids)?;
    assert_eq!(out1.shape().dims().to_vec(), vec![b,seq,te1.ctx_dim]);
    assert_eq!(out1.dtype(), DType::BF16);
    let m1 = out1.to_dtype(DType::F32)?.to_vec()?.iter().copied().sum::<f32>().abs();
    assert!(m1 > 0.0);

    let te2 = OpenClipG::from_weights_auto(path.to_str().unwrap(), &dev, seq)?;
    let out2 = te2.forward(&ids)?;
    assert_eq!(out2.shape().dims().to_vec(), vec![b,seq,te2.ctx_dim]);
    assert_eq!(out2.dtype(), DType::BF16);
    let m2 = out2.to_dtype(DType::F32)?.to_vec()?.iter().copied().sum::<f32>().abs();
    assert!(m2 > 0.0);
    Ok(())
}

