use anyhow::Result;
use flame_core::{Device, DType, Shape, Tensor};
use image::{ImageBuffer, Rgb};
use std::sync::OnceLock;

pub mod euler_a;
pub mod vae;

pub use euler_a::{euler_a_step, Karras};
pub use vae::{vae_decode_to_png, vae_encode, VaeKind, VaeSpec, VaePolicy};

fn trace_verbose() -> bool {
    static ONCE: OnceLock<bool> = OnceLock::new();
    *ONCE.get_or_init(|| std::env::var("VAE_TRACE_VERBOSE").ok().as_deref() == Some("1"))
}

pub(crate) fn log_stats_gpu_only(tag: &str, tensor: &Tensor) {
    if !trace_verbose() {
        return;
    }
    eprintln!(
        "[vae.stats] {tag} dtype={:?} storage={:?} shape={:?}",
        tensor.dtype(),
        tensor.storage_dtype(),
        tensor.shape().dims()
    );
}

/// Create BF16 NHWC tensor with random normal values using CPU RNG then upload
pub fn randn_nhwc_bf16(dims: [usize; 4], seed: u64, device: &Device) -> Result<Tensor> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let (b, h, w, c) = (dims[0], dims[1], dims[2], dims[3]);
    let n = b * h * w * c;
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    let cpu: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let t = Tensor::from_vec(cpu, Shape::from_dims(&[b, h, w, c]), device.cuda_device().clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}

/// Build a [n] tensor of the same scalar timestep value
pub fn timestep_batch(device: &Device, t: i32, n: usize) -> Result<Tensor> {
    let v = vec![t as f32; n];
    Ok(Tensor::from_vec(v, Shape::from_dims(&[n]), device.cuda_device().clone())?)
}

/// Split a [2, ...] tensor into two [1, ...] tensors along batch dim
pub fn split_rows2(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = t.shape().dims().to_vec();
    anyhow::ensure!(dims[0] == 2, "expected batch=2, got {:?}", dims[0]);
    let per = t.shape().elem_count() / 2;
    let a = t
        .slice_1d(0, per)?
        .reshape(&std::iter::once(1).chain(dims.iter().skip(1).cloned()).collect::<Vec<_>>())?;
    let b = t
        .slice_1d(per, per * 2)?
        .reshape(&std::iter::once(1).chain(dims.iter().skip(1).cloned()).collect::<Vec<_>>())?;
    Ok((a, b))
}

/// Save BF16 NHWC tensor to an 8-bit RGB PNG using [-1,1] -> [0,255]
pub fn save_png_nhwc_bf16(img: &Tensor, out: &str) -> Result<()> {
    let dims = img.shape().dims().to_vec();
    anyhow::ensure!(dims.len() == 4 && dims[0] == 1 && dims[3] == 3, "expect [1,H,W,3], got {:?}", dims);
    let (h, w) = (dims[1], dims[2]);
    let v = img.to_dtype(DType::F32)?.to_vec()?;
    let mut buf = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 3;
            let r = ((v[base + 0] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((v[base + 1] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let b = ((v[base + 2] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    buf.save(out)?;
    Ok(())
}
