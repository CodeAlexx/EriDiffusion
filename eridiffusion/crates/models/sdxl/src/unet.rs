use anyhow::{Result, bail};
use flame_core::{Tensor, Shape, DType, Device};
use crate::config::SdxlConfig;
use eridiffusion_common_text::masks as text_masks;
use eridiffusion_common_weights as cw;
use crate::stransformer::{SpatialTransformer, STBlock};
use crate::xattn::CrossAttn;
use crate::nhwc_utils::{khw_kicoc_to_oihw, nhwc_to_nchw, nchw_to_nhwc};
use crate::resblk::ResBlk;

pub struct SdxlUnet {
    pub cfg: SdxlConfig,
    pub device: Device,
    reg: cw::ParamRegistry,
    adapters: Option<adapters::adapter::AdapterSet>,
}

impl SdxlUnet {
    pub fn from_loader(cfg_yaml: &serde_yaml::Value, ld: &mut cw::SafeLoader, _reg_unused: &mut cw::ParamRegistry, device: &Device) -> Result<Self> {
        super::weight_load::guard_unet_loader(ld)?;
        // Parse config: allow minimal subset; default prediction_type=epsilon
        let mut cfg: SdxlConfig = serde_yaml::from_value(cfg_yaml.clone()).unwrap_or_default();
        if cfg.prediction_type.to_lowercase() != "epsilon" {
            bail!("UNet prediction_type={} not supported; sampler assumes epsilon", cfg.prediction_type);
        }
        // Map weights into ParamRegistry under canonical names (strict)
        let mut registry = cw::ParamRegistry::new();
        super::weight_load::map_and_register(ld, &mut registry, device)?;
        Ok(Self { cfg, device: device.clone(), reg: registry, adapters: None })
    }
    pub fn with_adapters(mut self, adapters: adapters::adapter::AdapterSet) -> Self { self.adapters = Some(adapters); self }
    pub fn build_base_shapes(&self) -> adapters::loader::BaseShapes {
        let mut bs = adapters::loader::BaseShapes::default();
        // Iterate registry keys and record shapes + conv flag
        for (name, t) in self.reg.iter() {
            let dims = t.shape().dims().to_vec();
            let is_conv = dims.len()==4;
            bs.by_target.insert(name.clone(), (dims.iter().map(|&d| d as i64).collect(), is_conv));
        }
        bs
    }

    /// Epsilon forward: multi-stage UNet (encoder with ResBlk + optional ST, skips, simple upsample decoder), head conv.
    /// lengths: I32 [B] used to build padding mask [B,1,1,seq]
    pub fn eps(&self, lat: &Tensor, t: &Tensor, ctx1: &Tensor, ctx2: &Tensor, lengths: &Tensor) -> Result<Tensor> {
        // Validate NHWC and dims
        let ld = lat.shape().dims().to_vec();
        anyhow::ensure!(ld.len()==4 && ld[3]==self.cfg.in_ch, "latents must be NHWC with {} channels; got {:?}", self.cfg.in_ch, ld);
        let b = ld[0];
        let s1 = ctx1.shape().dims().to_vec();
        let s2 = ctx2.shape().dims().to_vec();
        anyhow::ensure!(s1.len()==3 && s1[0]==b && s1[2]==self.cfg.ctx1_dim, "ctx1 shape mismatch: {:?}", s1);
        anyhow::ensure!(s2.len()==3 && s2[0]==b && s2[2]==self.cfg.ctx2_dim, "ctx2 shape mismatch: {:?}", s2);
        // Build mask [B,1,1,seq] from lengths (I32)
        let seq = s1[1] as usize;
        let mask_bool = text_masks::build_padding_mask_bool(lengths, seq)?;
        let mask_f32 = text_masks::sanitize_mask(&mask_bool.to_dtype(DType::F32)?)?;
        // Time embedding (sin-cos -> fc1->silu->fc2) if mapped
        let mut temb: Option<Tensor> = None;
        if self.has("unet.t_embed.fc1.weight") {
            let w1 = self.get("unet.t_embed.fc1.weight")?.to_dtype(DType::F32)?;
            let b1 = self.get("unet.t_embed.fc1.bias")?.to_dtype(DType::F32)?;
            let w2 = self.get("unet.t_embed.fc2.weight")?.to_dtype(DType::F32)?;
            let b2 = self.get("unet.t_embed.fc2.bias")?.to_dtype(DType::F32)?;
            let in_dim = w1.shape().dims()[0];
            let bsz = t.shape().dims()[0];
            // Build sinusoidal embedding of size in_dim
            let half = in_dim / 2;
            let idx = Tensor::arange(0.0, half as f32, 1.0, t.device().cuda_device().clone())?; // [half]
            let freq = idx.mul_scalar(2.0 / in_dim as f32)?.mul_scalar((10000f32).ln())?.exp()?; // [half]
            let tt = t.reshape(&[bsz,1])?.to_dtype(DType::F32)?; // [B,1]
            let ang = tt.matmul(&freq.reshape(&[half,1])?.transpose()?)?; // [B,half]
            let sin = ang.sin()?; let cos = ang.cos()?;
            let in_vec = Tensor::cat(&[&sin,&cos], 1)?; // [B,in_dim]
            let h = in_vec.matmul(&w1)?.add(&b1)?.silu()?;
            let e = h.matmul(&w2)?.add(&b2)?; // [B,embed]
            temb = Some(e);
        }

        // Stem conv 3x3
        let stem_w = self.get("unet.enc.s0.stem.weight")?;
        let stem_b = self.get("unet.enc.s0.stem.bias")?;
        let w = khw_kicoc_to_oihw(stem_w)?;
        let x_nchw = nhwc_to_nchw(lat)?;
        let y = x_nchw.conv2d(&w, Some(stem_b), 1, 1)?;
        let mut x = nchw_to_nhwc(&y)?;

        // Encoder: iterate stages s=0.. until no more blocks detected
        let mut skips: Vec<Tensor> = Vec::new();
        let mut s = 0usize;
        loop {
            let base = format!("unet.enc.s{}", s);
            let mut did_any = false;
            // ResBlk b0
            let b0 = format!("{}.b0", base);
            if self.has(&format!("{}.in_conv.weight", b0)) || self.has(&format!("{}.out_conv.weight", b0)) {
                let rb = self.build_resblk(&b0)?;
                x = rb.forward(&x, temb.as_ref())?;
                did_any = true;
            }
            // SpatialTransformer if present
            if self.has(&format!("{}.st.proj_in.weight", base)) {
                let st = self.build_st(&format!("{}.st", base))?;
                x = st.forward(&x, ctx1, ctx2, &mask_f32)?;
                did_any = true;
            }
            // Save skip before downsample
            if did_any { skips.push(x.to_dtype(DType::F32)?) };
            // Downsample op if present
            if self.has(&format!("{}.down_op.weight", base)) {
                let dw = self.get(&format!("{}.down_op.weight", base))?;
                let db = self.get(&format!("{}.down_op.bias", base)).ok().cloned();
                let w = khw_kicoc_to_oihw(dw)?;
                let x2 = nhwc_to_nchw(&x.to_dtype(DType::F32)?)?;
                let y = x2.conv2d(&w, db.as_ref(), 2, 1)?; // stride2, pad1
                x = nchw_to_nhwc(&y)?;
                did_any = true;
            }
            if !did_any { break; }
            s += 1;
            if s > 16 { break; } // safety cap
        }

        // Middle resblocks if present
        for mb in 0..2 {
            let midp = format!("unet.mid.b{}", mb);
            if self.has(&format!("{}.in_conv.weight", midp)) {
                let rb = self.build_resblk(&midp)?;
                x = rb.forward(&x, temb.as_ref())?;
            }
        }

        // Decoder: iterate stages s=0.., apply upsample op, add skip, apply ST and ResBlk
        let mut d = 0usize;
        loop {
            let base = format!("unet.dec.s{}", d);
            let mut did_any = false;
            // upsample op if present
            if self.has(&format!("{}.up_op.weight", base)) {
                // nearest x2 then conv
                let dims_x = x.shape().dims().to_vec();
                let (b,h,w,c) = (dims_x[0], dims_x[1], dims_x[2], dims_x[3]);
                let xu = x.to_dtype(DType::F32)?.reshape(&[b,h,1,w,1,c])?
                     .broadcast_to(&Shape::from_dims(&[b,h,2,w,2,c]))?
                     .reshape(&[b,h*2,w*2,c])?;
                let uw = self.get(&format!("{}.up_op.weight", base))?;
                let ub = self.get(&format!("{}.up_op.bias",   base)).ok().cloned();
                let w = khw_kicoc_to_oihw(uw)?;
                let x2 = nhwc_to_nchw(&xu)?;
                let y = x2.conv2d(&w, ub.as_ref(), 1, 1)?;
                x = nchw_to_nhwc(&y)?;
                did_any = true;
            }
            // add skip if available
            if let Some(skp) = skips.pop() {
                x = x.to_dtype(DType::F32)?.add(&skp)?;
                did_any = true;
            }
            // ST if present
            if self.has(&format!("{}.st.proj_in.weight", base)) {
                let st = self.build_st(&format!("{}.st", base))?;
                x = st.forward(&x, ctx1, ctx2, &mask_f32)?;
                did_any = true;
            }
            // ResBlk b0
            let b0 = format!("{}.b0", base);
            if self.has(&format!("{}.in_conv.weight", b0)) || self.has(&format!("{}.out_conv.weight", b0)) {
                let rb = self.build_resblk(&b0)?;
                x = rb.forward(&x, temb.as_ref())?;
                did_any = true;
            }
            if !did_any { break; }
            d += 1;
            if d > 16 { break; }
        }

        // Head norm (optional)
        if self.has("unet.head.norm.weight") {
            let gamma = self.get("unet.head.norm.weight")?.to_dtype(DType::F32)?;
            let beta  = self.get("unet.head.norm.bias")?.to_dtype(DType::F32)?;
            let g = gamma.reshape(&[1,1,1,gamma.shape().dims()[0]])?;
            let b = beta.reshape(&[1,1,1,beta.shape().dims()[0]])?;
            x = x.to_dtype(DType::F32)?.mul(&g)?.add(&b)?;
        }
        // Head conv 1x1
        let head_w = self.get("unet.head.conv.weight")?;
        let head_b = self.get("unet.head.conv.bias")?;
        let hw = khw_kicoc_to_oihw(head_w)?;
        let x2 = nhwc_to_nchw(&x.to_dtype(DType::F32)?)?;
        let y = x2.conv2d(&hw, Some(head_b), 1, 0)?;
        let out = nchw_to_nhwc(&y)?.to_dtype(DType::BF16)?;
        Ok(out)
    }
}

impl SdxlUnet {
    fn has(&self, name: &str) -> bool { self.reg.get(name).is_some() }
    fn get(&self, name: &str) -> Result<&Tensor> {
        self.reg.get(name).ok_or_else(|| anyhow::anyhow!(format!("missing canonical tensor: {}", name)))
    }
    fn build_st(&self, prefix: &str) -> Result<SpatialTransformer> {
        let pi_w = self.get(&format!("{}.proj_in.weight", prefix))?.to_dtype(DType::F32)?;
        let pi_b = self.get(&format!("{}.proj_in.bias",  prefix))?.to_dtype(DType::F32)?;
        let po_w = self.get(&format!("{}.proj_out.weight", prefix))?.to_dtype(DType::F32)?;
        let po_b = self.get(&format!("{}.proj_out.bias",  prefix))?.to_dtype(DType::F32)?;
        // tb0 only
        let tb = format!("{}.tb0", prefix);
        let qw = self.get(&format!("{}.xattn.to_q.weight", tb))?.to_dtype(DType::F32)?;
        let qb = self.get(&format!("{}.xattn.to_q.bias",   tb))?.to_dtype(DType::F32)?;
        let kw = self.get(&format!("{}.xattn.to_k.weight", tb))?.to_dtype(DType::F32)?;
        let kb = self.get(&format!("{}.xattn.to_k.bias",   tb))?.to_dtype(DType::F32)?;
        let vw = self.get(&format!("{}.xattn.to_v.weight", tb))?.to_dtype(DType::F32)?;
        let vb = self.get(&format!("{}.xattn.to_v.bias",   tb))?.to_dtype(DType::F32)?;
        let ow = self.get(&format!("{}.xattn.to_out.weight", tb))?.to_dtype(DType::F32)?;
        let ob = self.get(&format!("{}.xattn.to_out.bias",   tb))?.to_dtype(DType::F32)?;
        let ff1_w = self.get(&format!("{}.ff.fc1.weight", tb))?.to_dtype(DType::F32)?;
        let ff1_b = self.get(&format!("{}.ff.fc1.bias",   tb))?.to_dtype(DType::F32)?;
        let ff2_w = self.get(&format!("{}.ff.fc2.weight", tb))?.to_dtype(DType::F32)?;
        let ff2_b = self.get(&format!("{}.ff.fc2.bias",   tb))?.to_dtype(DType::F32)?;
        let c = qw.shape().dims()[0];
        let hd = self.cfg.head_dim;
        anyhow::ensure!(c % hd == 0, "head_dim={} must divide C={}", hd, c);
        let nh = c / hd;
        // Validate attn2 K/V expect 2048-dim context
        anyhow::ensure!(kw.shape().dims()[0] == 2048, "attn2.to_k IN dim must be 2048, got {:?}", kw.shape().dims());
        anyhow::ensure!(vw.shape().dims()[0] == 2048, "attn2.to_v IN dim must be 2048, got {:?}", vw.shape().dims());
        // Q/Out are square [C,C]
        anyhow::ensure!(qw.shape().dims()[1] == c, "to_q OUT dim must equal C (got {:?})", qw.shape().dims());
        anyhow::ensure!(ow.shape().dims()[0] == c && ow.shape().dims()[1] == c, "to_out must be [C,C], got {:?}", ow.shape().dims());
        let attn_cat1 = CrossAttn { to_q_w: qw.clone(), to_q_b: qb.clone(), to_k_w: kw.clone(), to_k_b: kb.clone(), to_v_w: vw.clone(), to_v_b: vb.clone(), to_out_w: ow.clone(), to_out_b: ob.clone(), n_heads: nh, head_dim: hd };
        let attn_cat2 = CrossAttn { to_q_w: qw,          to_q_b: qb,          to_k_w: kw,          to_k_b: kb,          to_v_w: vw,          to_v_b: vb,          to_out_w: ow,          to_out_b: ob,          n_heads: nh, head_dim: hd };
        // Only use a single cross-attn over concatenated ctx in minimal path (xattn1)
        let block = STBlock { self_attn: None, xattn1: attn_cat1, xattn2: attn_cat2, ff_fc1_w: ff1_w, ff_fc1_b: ff1_b, ff_fc2_w: ff2_w, ff_fc2_b: ff2_b };
        Ok(SpatialTransformer { proj_in_w: pi_w, proj_in_b: pi_b, blocks: vec![block], proj_out_w: po_w, proj_out_b: po_b })
    }

    fn build_resblk(&self, prefix: &str) -> Result<ResBlk> {
        // Expect keys like: in_gn.weight/bias, in_conv.weight/bias, out_gn.weight/bias, out_conv.weight/bias, optional skip.weight/bias
        let igw = self.get(&format!("{}.in_gn.weight", prefix))?.clone();
        let igb = self.get(&format!("{}.in_gn.bias",  prefix))?.clone();
        let icw = self.get(&format!("{}.in_conv.weight", prefix))?.clone();
        let icb = self.get(&format!("{}.in_conv.bias",  prefix))?.clone();
        let ogw = self.get(&format!("{}.out_gn.weight", prefix))?.clone();
        let ogb = self.get(&format!("{}.out_gn.bias",  prefix))?.clone();
        let ocw = self.get(&format!("{}.out_conv.weight", prefix))?.clone();
        let ocb = self.get(&format!("{}.out_conv.bias",  prefix))?.clone();
        let skw = self.reg.get(&format!("{}.skip.weight", prefix)).cloned();
        let skb = self.reg.get(&format!("{}.skip.bias",   prefix)).cloned();
        let t2w = self.reg.get(&format!("{}.temb2.weight", prefix)).cloned();
        let t2b = self.reg.get(&format!("{}.temb2.bias",   prefix)).cloned();
        Ok(ResBlk { in_gn_w: igw, in_gn_b: igb, in_conv_w: icw, in_conv_b: icb, out_gn_w: ogw, out_gn_b: ogb, out_conv_w: ocw, out_conv_b: ocb, skip_w: skw, skip_b: skb, temb2_w: t2w, temb2_b: t2b, groups: 32 })
    }
}
