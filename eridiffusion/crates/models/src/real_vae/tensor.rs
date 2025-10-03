
//! Minimal CPU tensor for VAE math (NCHW, f32).
//! Intentionally simple so it compiles everywhere; swap with flame_core later.

#![allow(dead_code)]
#[derive(Clone, Debug)]
pub struct CpuTensor {
    pub n: usize, pub c: usize, pub h: usize, pub w: usize,
    pub data: Vec<f32>,
}
impl CpuTensor {
    pub fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self { n, c, h, w, data: vec![0.0; n*c*h*w] }
    }
    pub fn from_vec(n: usize, c: usize, h: usize, w: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), n*c*h*w);
        Self { n, c, h, w, data }
    }
    #[inline] pub fn idx(&self, n:usize,c:usize,y:usize,x:usize)->usize {
        ((n*self.c + c)*self.h + y)*self.w + x
    }
}
