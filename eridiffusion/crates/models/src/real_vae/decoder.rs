
//! Decoder: latents [B,4,H/8,W/8] -> RGB [B,3,H,W]

#![allow(dead_code)]
use super::tensor::CpuTensor;
use super::ops::*;

#[derive(Clone, Debug)]
pub struct DecWeights {
    pub c1_w: Vec<f32>, pub c1_b: Vec<f32>, // 4->256, k3
    pub c2_w: Vec<f32>, pub c2_b: Vec<f32>, // 256->128, k3
    pub c3_w: Vec<f32>, pub c3_b: Vec<f32>, // 128->64, k3
    pub c4_w: Vec<f32>, pub c4_b: Vec<f32>, // 64->3, k3
}
impl DecWeights {
    pub fn random() -> Self {
        fn rnd(n: usize)->Vec<f32>{ (0..n).map(|_| (rand::random::<f32>()-0.5)*0.02).collect() }
        Self {
            c1_w: rnd(256*4*3*3),  c1_b: vec![0.0;256],
            c2_w: rnd(128*256*3*3),c2_b: vec![0.0;128],
            c3_w: rnd(64*128*3*3), c3_b: vec![0.0;64],
            c4_w: rnd(3*64*3*3),   c4_b: vec![0.0;3],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Decoder { pub w: DecWeights, pub scaling: f32 }
impl Decoder {
    pub fn new_random(scaling: f32) -> Self { Self { w: DecWeights::random(), scaling } }

    pub fn forward(&self, x: &CpuTensor) -> CpuTensor {
        // scale multiply to return to image space
        let mut z = x.clone();
        for v in &mut z.data { *v *= self.scaling; }

        // upsample x2 then conv 4->256
        z = nearest_upsample_x2(&z);
        let mut y = conv2d_nchw(&z, &self.w.c1_w, Some(&self.w.c1_b), 4,256,3,3, 1,1, 1);
        group_norm_inplace(&mut y, 8, 1e-5); silu_inplace(&mut y);

        // upsample x2
        y = nearest_upsample_x2(&y);
        y = conv2d_nchw(&y, &self.w.c2_w, Some(&self.w.c2_b), 256,128,3,3, 1,1, 1);
        group_norm_inplace(&mut y, 8, 1e-5); silu_inplace(&mut y);

        // upsample x2
        y = nearest_upsample_x2(&y);
        y = conv2d_nchw(&y, &self.w.c3_w, Some(&self.w.c3_b), 128,64,3,3, 1,1, 1);
        group_norm_inplace(&mut y, 8, 1e-5); silu_inplace(&mut y);

        // head 64->3
        let y = conv2d_nchw(&y, &self.w.c4_w, Some(&self.w.c4_b), 64,3,3,3, 1,1, 1);
        y
    }
}
