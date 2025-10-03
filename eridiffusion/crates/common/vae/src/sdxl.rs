use anyhow::{ensure, Context, Result};
use flame_core::{CudaDevice, Tensor};

use crate::scale::{apply_decode_scale, apply_encode_scale, read_vae_scaling};
use crate::spec::{VaePolicy, VaeSpec};

/// Encode NHWC BF16 images -> NHWC BF16 latents using the native SDXL VAE
pub fn encode(spec: &VaeSpec, images: &Tensor, policy: VaePolicy) -> Result<Tensor> {
    native::encode(spec, images, policy)
}

/// Decode NHWC BF16 latents -> NHWC BF16 images using the native SDXL VAE
pub fn decode(spec: &VaeSpec, latents: &Tensor, policy: VaePolicy) -> Result<Tensor> {
    native::decode(spec, latents, policy)
}

mod native {
    use super::*;
    use eridiffusion_common_weights::strict_loader::{tensor_from_bytes, StrictMmapLoader};
    use flame_core::{group_norm, image_ops_nhwc, DType};
    #[cfg(feature = "bf16_u16")]
    use cudarc::driver::{sys, LaunchAsync, LaunchConfig};
    #[cfg(feature = "bf16_u16")]
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, OnceLock};

    const GROUPS: usize = 32;
    const EPS: f32 = 1e-6;

    const POST_DECODE_MODULE: &str = "sdxl_vae_finish";
    const POST_DECODE_FUNC: &str = "sdxl_vae_finish_bf16";

    #[cfg(feature = "bf16_u16")]
    const POST_DECODE_KERNEL: &str = r#"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" __global__ void sdxl_vae_finish_bf16(__nv_bfloat16* data, int numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    float v = __bfloat162float(data[idx]);
    v = (v + 1.0f) * 0.5f;
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    data[idx] = __float2bfloat16_rn(v);
}
"#;

    #[cfg(feature = "bf16_u16")]
    fn ensure_post_decode_kernel(device: &Arc<CudaDevice>) -> Result<()> {
        if device
            .get_func(POST_DECODE_MODULE, POST_DECODE_FUNC)
            .is_some()
        {
            return Ok(());
        }

        let mut opts = CompileOptions::default();
        if let Ok(include_dir) = std::env::var("CUDA_INCLUDE_DIR") {
            opts.include_paths.push(include_dir);
        } else if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
            opts.include_paths.push(format!("{cuda_home}/include"));
        }

        let ptx = compile_ptx_with_opts(POST_DECODE_KERNEL, opts)
            .context("failed to compile VAE post-decode PTX")?;
        device
            .load_ptx(ptx, POST_DECODE_MODULE, &[POST_DECODE_FUNC])
            .context("failed to load VAE post-decode PTX")?;
        Ok(())
    }

    #[cfg(feature = "bf16_u16")]
    fn post_decode_affine_clamp(mut tensor: Tensor) -> Result<Tensor> {
        ensure!(
            tensor.dtype() == DType::BF16,
            "VAE post-decode kernel expects BF16 tensor, got {:?}",
            tensor.dtype()
        );
        let device = tensor.device().clone();
        ensure_post_decode_kernel(&device)?;
        let numel = tensor.shape().elem_count();
        if numel == 0 {
            return Ok(tensor);
        }

        let dev_ptr = tensor
            .as_mut_device_ptr_bf16("sdxl_vae.post_decode")
            .map_err(anyhow::Error::from)? as usize as sys::CUdeviceptr;
        let threads = 256u32;
        let blocks = ((numel + threads as usize - 1) / threads as usize) as u32;
        if blocks == 0 {
            return Ok(tensor);
        }

        let func = device
            .get_func(POST_DECODE_MODULE, POST_DECODE_FUNC)
            .context("VAE post-decode kernel missing after load")?;
        unsafe {
            func.launch(
                LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                },
                (dev_ptr, numel as i32),
            )
            .context("failed to launch VAE post-decode kernel")?;
        }
        Ok(tensor)
    }

    #[cfg(not(feature = "bf16_u16"))]
    fn post_decode_affine_clamp(tensor: Tensor) -> Result<Tensor> {
        let f32 = tensor.to_dtype(DType::F32)?;
        let adjusted = f32.add_scalar(1.0)?.mul_scalar(0.5)?.clamp(0.0, 1.0)?;
        Ok(adjusted.to_dtype(DType::BF16)?)
    }

    #[derive(Hash, PartialEq, Eq)]
    struct CacheKey {
        path: PathBuf,
        device_ordinal: usize,
        latent_channels: usize,
        latent_div: usize,
        scale_bits: u64,
    }

    static CACHE: OnceLock<Mutex<HashMap<CacheKey, Arc<SdxlNativeVae>>>> = OnceLock::new();

    pub fn encode(spec: &VaeSpec, images: &Tensor, policy: VaePolicy) -> Result<Tensor> {
        ensure!(matches!(policy, VaePolicy::GpuFirst), "SDXL VAE requires CUDA device");
        let device = images.device().clone();
        let vae = get_cached(spec, device)?;
        vae.encode(spec, images)
    }

    pub fn decode(spec: &VaeSpec, latents: &Tensor, policy: VaePolicy) -> Result<Tensor> {
        ensure!(matches!(policy, VaePolicy::GpuFirst), "SDXL VAE requires CUDA device");
        let device = latents.device().clone();
        let vae = get_cached(spec, device)?;
        vae.decode(spec, latents)
    }

    fn get_cached(spec: &VaeSpec, device: Arc<CudaDevice>) -> Result<Arc<SdxlNativeVae>> {
        let path = PathBuf::from(&spec.path);
        ensure!(path.exists(), "SDXL VAE weights not found at {}", path.display());
        let key = CacheKey {
            path,
            device_ordinal: device.ordinal(),
            latent_channels: spec.latent_channels,
            latent_div: spec.latent_div,
            scale_bits: spec.latent_scale.to_bits() as u64,
        };
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(found) = cache.lock().unwrap().get(&key).cloned() {
            return Ok(found);
        }
        let vae = Arc::new(SdxlNativeVae::load(spec, device)?);
        cache.lock().unwrap().insert(key, vae.clone());
        Ok(vae)
    }

    struct Conv2dLayer {
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
    }

    impl Conv2dLayer {
        fn from_loader(
            loader: &mut StrictMmapLoader,
            device: &Arc<CudaDevice>,
            weight_key: &str,
            bias_key: Option<&str>,
            stride: usize,
            padding: usize,
        ) -> Result<Self> {
            let weight = load_param(loader, device, weight_key)?;
            let bias = match bias_key {
                Some(key) => Some(load_param(loader, device, key)?),
                None => None,
            };
            Ok(Self {
                weight,
                bias,
                stride,
                padding,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            Ok(x.conv2d(&self.weight, self.bias.as_ref(), self.stride, self.padding)?)
        }
    }

    struct GroupNormLayer {
        weight: Tensor,
        bias: Tensor,
        groups: usize,
        eps: f32,
    }

    impl GroupNormLayer {
        fn from_loader(
            loader: &mut StrictMmapLoader,
            device: &Arc<CudaDevice>,
            weight_key: &str,
            bias_key: &str,
            groups: usize,
            eps: f32,
        ) -> Result<Self> {
            Ok(Self {
                weight: load_param(loader, device, weight_key)?,
                bias: load_param(loader, device, bias_key)?,
                groups,
                eps,
            })
        }

        fn apply(&self, x: &Tensor) -> Result<Tensor> {
            Ok(group_norm(x, self.groups, Some(&self.weight), Some(&self.bias), self.eps)?)
        }
    }

    struct ResNetBlock {
        norm1: GroupNormLayer,
        conv1: Conv2dLayer,
        norm2: GroupNormLayer,
        conv2: Conv2dLayer,
        shortcut: Option<Conv2dLayer>,
    }

    impl ResNetBlock {
        fn new(
            loader: &mut StrictMmapLoader,
            device: &Arc<CudaDevice>,
            prefix: &str,
            in_channels: usize,
            out_channels: usize,
        ) -> Result<Self> {
            let norm1 = GroupNormLayer::from_loader(
                loader,
                device,
                &format!("{}.norm1.weight", prefix),
                &format!("{}.norm1.bias", prefix),
                GROUPS,
                EPS,
            )?;
            let conv1 = Conv2dLayer::from_loader(
                loader,
                device,
                &format!("{}.conv1.weight", prefix),
                Some(&format!("{}.conv1.bias", prefix)),
                1,
                1,
            )?;
            let norm2 = GroupNormLayer::from_loader(
                loader,
                device,
                &format!("{}.norm2.weight", prefix),
                &format!("{}.norm2.bias", prefix),
                GROUPS,
                EPS,
            )?;
            let conv2 = Conv2dLayer::from_loader(
                loader,
                device,
                &format!("{}.conv2.weight", prefix),
                Some(&format!("{}.conv2.bias", prefix)),
                1,
                1,
            )?;
            let shortcut = if in_channels != out_channels {
                Some(Conv2dLayer::from_loader(
                    loader,
                    device,
                    &format!("{}.nin_shortcut.weight", prefix),
                    Some(&format!("{}.nin_shortcut.bias", prefix)),
                    1,
                    0,
                )?)
            } else {
                None
            };
            Ok(Self {
                norm1,
                conv1,
                norm2,
                conv2,
                shortcut,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let residual = if let Some(shortcut) = &self.shortcut {
                shortcut.forward(x)?
            } else {
                x.clone()
            };
            let h = self.norm1.apply(x)?.silu()?;
            let h = self.conv1.forward(&h)?;
            let h = self.norm2.apply(&h)?.silu()?;
            let h = self.conv2.forward(&h)?;
            Ok(h.add(&residual)?)
        }
    }

    struct DownBlock {
        resnets: Vec<ResNetBlock>,
        downsample: Option<Conv2dLayer>,
    }

    impl DownBlock {
        fn new(
            loader: &mut StrictMmapLoader,
            device: &Arc<CudaDevice>,
            block_idx: usize,
            in_channels: usize,
            out_channels: usize,
            add_downsample: bool,
        ) -> Result<Self> {
            let mut resnets = Vec::with_capacity(2);
            resnets.push(ResNetBlock::new(
                loader,
                device,
                &format!("encoder.down.{}.block.0", block_idx),
                in_channels,
                out_channels,
            )?);
            resnets.push(ResNetBlock::new(
                loader,
                device,
                &format!("encoder.down.{}.block.1", block_idx),
                out_channels,
                out_channels,
            )?);
            let downsample = if add_downsample {
                Some(Conv2dLayer::from_loader(
                    loader,
                    device,
                    &format!("encoder.down.{}.downsample.conv.weight", block_idx),
                    Some(&format!("encoder.down.{}.downsample.conv.bias", block_idx)),
                    2,
                    1,
                )?)
            } else {
                None
            };
            Ok(Self { resnets, downsample })
        }

        fn forward(&self, mut x: Tensor) -> Result<Tensor> {
            for resnet in &self.resnets {
                x = resnet.forward(&x)?;
            }
            if let Some(down) = &self.downsample {
                x = down.forward(&x)?;
            }
            Ok(x)
        }
    }

    struct UpBlock {
        resnets: Vec<ResNetBlock>,
        upsample: Option<Conv2dLayer>,
    }

    impl UpBlock {
        fn new(
            loader: &mut StrictMmapLoader,
            device: &Arc<CudaDevice>,
            block_idx: usize,
            in_channels: usize,
            out_channels: usize,
            add_upsample: bool,
        ) -> Result<Self> {
            let mut resnets = Vec::with_capacity(3);
            // Up blocks have three ResNet layers in SDXL VAE
            for layer in 0..3 {
                let prefix = format!("decoder.up.{}.block.{}", block_idx, layer);
                let input_ch = if layer == 0 { in_channels } else { out_channels };
                resnets.push(ResNetBlock::new(loader, device, &prefix, input_ch, out_channels)?);
            }
            let upsample = if add_upsample {
                let weight_key = format!("decoder.up.{}.upsample.conv.weight", block_idx);
                match loader.info(&weight_key) {
                    Ok(_) => Some(Conv2dLayer::from_loader(
                        loader,
                        device,
                        &weight_key,
                        Some(&format!("decoder.up.{}.upsample.conv.bias", block_idx)),
                        1,
                        1,
                    )?),
                    Err(_) => None,
                }
            } else {
                None
            };
            Ok(Self { resnets, upsample })
        }

        fn forward(&self, mut x: Tensor) -> Result<Tensor> {
            for resnet in &self.resnets {
                x = resnet.forward(&x)?;
            }
            if let Some(conv) = &self.upsample {
                x = upsample_and_conv(&x, conv)?;
            }
            Ok(x)
        }
    }

    struct AttentionBlock {
        norm: GroupNormLayer,
        q: Conv2dLayer,
        k: Conv2dLayer,
        v: Conv2dLayer,
        proj_out: Conv2dLayer,
    }

    impl AttentionBlock {
        fn new(loader: &mut StrictMmapLoader, device: &Arc<CudaDevice>, prefix: &str) -> Result<Self> {
            Ok(Self {
                norm: GroupNormLayer::from_loader(
                    loader,
                    device,
                    &format!("{}.norm.weight", prefix),
                    &format!("{}.norm.bias", prefix),
                    GROUPS,
                    EPS,
                )?,
                q: Conv2dLayer::from_loader(
                    loader,
                    device,
                    &format!("{}.q.weight", prefix),
                    Some(&format!("{}.q.bias", prefix)),
                    1,
                    0,
                )?,
                k: Conv2dLayer::from_loader(
                    loader,
                    device,
                    &format!("{}.k.weight", prefix),
                    Some(&format!("{}.k.bias", prefix)),
                    1,
                    0,
                )?,
                v: Conv2dLayer::from_loader(
                    loader,
                    device,
                    &format!("{}.v.weight", prefix),
                    Some(&format!("{}.v.bias", prefix)),
                    1,
                    0,
                )?,
                proj_out: Conv2dLayer::from_loader(
                    loader,
                    device,
                    &format!("{}.proj_out.weight", prefix),
                    Some(&format!("{}.proj_out.bias", prefix)),
                    1,
                    0,
                )?,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let residual = x.clone();
            let x_norm = self.norm.apply(x)?;
            let q = self.q.forward(&x_norm)?;
            let k = self.k.forward(&x_norm)?;
            let v = self.v.forward(&x_norm)?;

            let [b, c, h, w] = q.dims4();
            let hw = h * w;

            let q = q.reshape(&[b, c, hw])?.transpose_dims(1, 2)?; // [B, HW, C]
            let k = k.reshape(&[b, c, hw])?;
            let k_t = k.transpose_dims(1, 2)?; // [B, C, HW]
            let v = v.reshape(&[b, c, hw])?.transpose_dims(1, 2)?; // [B, HW, C]

            let scale = (c as f32).sqrt();
            let scores = q.bmm(&k_t)?.mul_scalar(1.0 / scale)?;
            let attn = scores.softmax(-1)?;
            let out = attn.bmm(&v)?; // [B, HW, C]
            let out = out.transpose_dims(1, 2)?.reshape(&[b, c, h, w])?;
            let proj = self.proj_out.forward(&out)?;
            Ok(proj.add(&residual)?)
        }
    }

    struct MidBlock {
        resnet1: ResNetBlock,
        attn: AttentionBlock,
        resnet2: ResNetBlock,
    }

    impl MidBlock {
        fn new(loader: &mut StrictMmapLoader, device: &Arc<CudaDevice>, prefix: &str, channels: usize) -> Result<Self> {
            Ok(Self {
                resnet1: ResNetBlock::new(loader, device, &format!("{}.block_1", prefix), channels, channels)?,
                attn: AttentionBlock::new(loader, device, &format!("{}.attn_1", prefix))?,
                resnet2: ResNetBlock::new(loader, device, &format!("{}.block_2", prefix), channels, channels)?,
            })
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let h = self.resnet1.forward(x)?;
            let h = self.attn.forward(&h)?;
            self.resnet2.forward(&h)
        }
    }

    struct SdxlNativeVae {
        latent_channels: usize,
        latent_scale: f32,
        encoder_conv_in: Conv2dLayer,
        encoder_down: Vec<DownBlock>,
        encoder_mid: MidBlock,
        encoder_norm_out: GroupNormLayer,
        encoder_conv_out: Conv2dLayer,
        quant_conv: Conv2dLayer,
        post_quant_conv: Conv2dLayer,
        decoder_conv_in: Conv2dLayer,
        decoder_mid: MidBlock,
        decoder_up: Vec<UpBlock>,
        decoder_norm_out: GroupNormLayer,
        decoder_conv_out: Conv2dLayer,
    }

    impl SdxlNativeVae {
        fn load(spec: &VaeSpec, device: Arc<CudaDevice>) -> Result<Self> {
            let mut loader = StrictMmapLoader::open(Path::new(&spec.path))
                .with_context(|| format!("open SDXL VAE weights at {}", spec.path))?;
            let base_channels = {
                let info = loader.info("encoder.down.0.block.0.conv1.weight")?;
                ensure!(info.shape.len() == 4, "unexpected conv1 weight shape: {:?}", info.shape);
                info.shape[0]
            };
            ensure!(base_channels == 128, "unexpected base channels {}", base_channels);

            let encoder_conv_in = Conv2dLayer::from_loader(
                &mut loader,
                &device,
                "encoder.conv_in.weight",
                Some("encoder.conv_in.bias"),
                1,
                1,
            )?;

            let channel_mult = [1, 2, 4, 4];
            let mut encoder_down = Vec::with_capacity(channel_mult.len());
            let mut prev = base_channels;
            for (idx, mult) in channel_mult.iter().enumerate() {
                let channels = base_channels * mult;
                encoder_down.push(DownBlock::new(
                    &mut loader,
                    &device,
                    idx,
                    prev,
                    channels,
                    idx < channel_mult.len() - 1,
                )?);
                prev = channels;
            }

            let encoder_mid = MidBlock::new(&mut loader, &device, "encoder.mid", 512)?;

            let encoder_norm_out = GroupNormLayer::from_loader(
                &mut loader,
                &device,
                "encoder.norm_out.weight",
                "encoder.norm_out.bias",
                GROUPS,
                EPS,
            )?;
            let encoder_conv_out = Conv2dLayer::from_loader(
                &mut loader,
                &device,
                "encoder.conv_out.weight",
                Some("encoder.conv_out.bias"),
                1,
                1,
            )?;
            let quant_conv = Conv2dLayer::from_loader(
                &mut loader,
                &device,
                "quant_conv.weight",
                Some("quant_conv.bias"),
                1,
                0,
            )?;
            let post_quant_conv = Conv2dLayer::from_loader(
                &mut loader,
                &device,
                "post_quant_conv.weight",
                Some("post_quant_conv.bias"),
                1,
                0,
            )?;

            let decoder_conv_in = Conv2dLayer::from_loader(
                &mut loader,
                &device,
                "decoder.conv_in.weight",
                Some("decoder.conv_in.bias"),
                1,
                1,
            )?;
            let decoder_mid = MidBlock::new(&mut loader, &device, "decoder.mid", 512)?;

            let decoder_in_channels = [512, 512, 512, 256];
            let decoder_out_channels = [512, 512, 256, 128];
            let mut decoder_up = Vec::with_capacity(4);
            for (idx, (&in_ch, &out_ch)) in decoder_in_channels
                .iter()
                .zip(decoder_out_channels.iter())
                .enumerate()
            {
                decoder_up.push(UpBlock::new(
                    &mut loader,
                    &device,
                    idx,
                    in_ch,
                    out_ch,
                    idx < decoder_in_channels.len() - 1,
                )?);
            }

            let decoder_norm_out = GroupNormLayer::from_loader(
                &mut loader,
                &device,
                "decoder.norm_out.weight",
                "decoder.norm_out.bias",
                GROUPS,
                EPS,
            )?;
            let decoder_conv_out = Conv2dLayer::from_loader(
                &mut loader,
                &device,
                "decoder.conv_out.weight",
                Some("decoder.conv_out.bias"),
                1,
                1,
            )?;

            Ok(Self {
                latent_channels: spec.latent_channels,
                latent_scale: read_vae_scaling(spec),
                encoder_conv_in,
                encoder_down,
                encoder_mid,
                encoder_norm_out,
                encoder_conv_out,
                quant_conv,
                post_quant_conv,
                decoder_conv_in,
                decoder_mid,
                decoder_up,
                decoder_norm_out,
                decoder_conv_out,
            })
        }

        fn encode(&self, spec: &VaeSpec, images: &Tensor) -> Result<Tensor> {
            let dims = images.shape().dims().to_vec();
            ensure!(dims.len() == 4 && dims[3] == 3, "images must be NHWC with 3 channels; got {:?}", dims);
            let mut x = images.to_dtype(DType::F32)?.permute(&[0, 3, 1, 2])?; // NCHW
            x = x.mul_scalar(2.0)?.add_scalar(-1.0)?;
            x = self.encoder_conv_in.forward(&x)?;
            for block in &self.encoder_down {
                x = block.forward(x)?;
            }
            x = self.encoder_mid.forward(&x)?;
            x = self.encoder_norm_out.apply(&x)?.silu()?;
            x = self.encoder_conv_out.forward(&x)?;
            x = self.quant_conv.forward(&x)?;

            let mut chunks = x.chunk(2, 1)?;
            let mean = chunks.remove(0);
            let scaled = apply_encode_scale(&mean, self.latent_scale)?;
            let latents = scaled.permute(&[0, 2, 3, 1])?.to_dtype(DType::BF16)?;
            ensure!(
                latents.shape().dims()[3] == spec.latent_channels,
                "encoded latents mismatch: expected {} channels got {}",
                spec.latent_channels,
                latents.shape().dims()[3]
            );
            Ok(latents)
        }

        fn decode(&self, _spec: &VaeSpec, latents: &Tensor) -> Result<Tensor> {
            let dims = latents.shape().dims().to_vec();
            ensure!(
                dims.len() == 4 && dims[3] == self.latent_channels,
                "latents must be NHWC with {} channels; got {:?}",
                self.latent_channels,
                dims
            );
            let mut z = latents.to_dtype(DType::F32)?.permute(&[0, 3, 1, 2])?;
            z = apply_decode_scale(&z, self.latent_scale)?;
            z = self.post_quant_conv.forward(&z)?;
            z = self.decoder_conv_in.forward(&z)?;
            z = self.decoder_mid.forward(&z)?;
            for block in &self.decoder_up {
                z = block.forward(z)?;
            }
            z = self.decoder_norm_out.apply(&z)?.silu()?;
            z = self.decoder_conv_out.forward(&z)?;
            let nhwc = z.permute(&[0, 2, 3, 1])?;
            let bf16 = nhwc.to_dtype(DType::BF16)?;
            post_decode_affine_clamp(bf16)
        }
    }

    fn upsample_and_conv(x: &Tensor, conv: &Conv2dLayer) -> Result<Tensor> {
        let dims = x.dims4();
        let nhwc = x.permute(&[0, 2, 3, 1])?;
        let up = image_ops_nhwc::resize_bilinear_nhwc(&nhwc, dims[2] * 2, dims[3] * 2, false)?;
        let nchw = up.permute(&[0, 3, 1, 2])?;
        conv.forward(&nchw)
    }

    fn load_param(loader: &mut StrictMmapLoader, device: &Arc<CudaDevice>, key: &str) -> Result<Tensor> {
        let info = loader.info(key)?;
        let bytes = loader.bytes(key)?;
        let tensor = tensor_from_bytes(
            flame_core::Device::from(device.clone()),
            &info,
            bytes,
        )?;
        loader.mark_used(key);
        let tensor = if tensor.dtype() != DType::F32 {
            tensor.to_dtype(DType::F32)?
        } else {
            tensor
        };
        Ok(tensor)
    }
}
