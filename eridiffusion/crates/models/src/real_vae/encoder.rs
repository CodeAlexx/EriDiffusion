
//! Encoder: RGB [B,3,H,W] -> latents [B,4,H/8,W/8]

#![allow(dead_code)]
use super::tensor::CpuTensor;
use super::ops::*;

#[derive(Clone, Debug)]
pub struct EncWeights {
    pub c1_w: Vec<f32>, pub c1_b: Vec<f32>, // 3->64, k3
    pub c2_w: Vec<f32>, pub c2_b: Vec<f32>, // 64->128, k3, s2
    pub c3_w: Vec<f32>, pub c3_b: Vec<f32>, // 128->256, k3, s2
    pub c4_w: Vec<f32>, pub c4_b: Vec<f32>, // 256->4, k3, s2 (latent)
}
impl EncWeights {
    pub fn random() -> Self {
        fn rnd(n: usize)->Vec<f32>{ (0..n).map(|_| (rand::random::<f32>()-0.5)*0.02).collect() }
        Self {
            c1_w: rnd(64*3*3*3),  c1_b: vec![0.0;64],
            c2_w: rnd(128*64*3*3),c2_b: vec![0.0;128],
            c3_w: rnd(256*128*3*3),c3_b: vec![0.0;256],
            c4_w: rnd(4*256*3*3), c4_b: vec![0.0;4],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Encoder { pub w: EncWeights, pub scaling: f32 }
impl Encoder {
    pub fn new_random(scaling: f32) -> Self { Self { w: EncWeights::random(), scaling } }

    pub fn forward(&self, x: &CpuTensor) -> CpuTensor {
        // c1: 3->64, k3,s1,p1
        let mut y = conv2d_nchw(x, &self.w.c1_w, Some(&self.w.c1_b), 3, 64, 3,3, 1,1, 1);
        group_norm_inplace(&mut y, 8, 1e-5); silu_inplace(&mut y);
        // c2: 64->128, k3, s2
        y = conv2d_nchw(&y, &self.w.c2_w, Some(&self.w.c2_b), 64,128,3,3, 2,1, 1);
        group_norm_inplace(&mut y, 8, 1e-5); silu_inplace(&mut y);
        // c3: 128->256, k3, s2
        y = conv2d_nchw(&y, &self.w.c3_w, Some(&self.w.c3_b), 128,256,3,3, 2,1, 1);
        group_norm_inplace(&mut y, 8, 1e-5); silu_inplace(&mut y);
        // c4: 256->4, k3, s2 (final down to H/8,W/8)
        let mut lat = conv2d_nchw(&y, &self.w.c4_w, Some(&self.w.c4_b), 256,4,3,3, 2,1, 1);
        // Scale division to match latent convention
        for v in &mut lat.data { *v /= self.scaling.max(1e-8); }
        lat
    }
}
