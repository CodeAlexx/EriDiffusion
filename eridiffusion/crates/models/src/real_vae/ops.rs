
//! Pure Rust ops: conv2d (NCHW), group norm, SiLU, nearest upsample.

#![allow(dead_code)]
use super::tensor::CpuTensor;

pub fn silu_inplace(x: &mut CpuTensor) {
    for v in &mut x.data {
        let s = 1.0 / (1.0 + (-*v).exp());
        *v = *v * s;
    }
}

pub fn group_norm_inplace(x: &mut CpuTensor, groups: usize, eps: f32) {
    assert!(x.c % groups == 0);
    let ch_per = x.c / groups;
    for n in 0..x.n {
        for g in 0..groups {
            let c0 = g * ch_per;
            let mut mean = 0.0;
            let mut var = 0.0;
            let mut cnt = 0usize;
            for c in c0..c0+ch_per {
                for y in 0..x.h { for z in 0..x.w {
                    mean += x.data[x.idx(n,c,y,z)];
                    cnt += 1;
                }}
            }
            mean /= cnt as f32;
            for c in c0..c0+ch_per {
                for y in 0..x.h { for z in 0..x.w {
                    let d = x.data[x.idx(n,c,y,z)] - mean;
                    var += d*d;
                }}
            }
            var = (var / cnt as f32 + eps).sqrt();
            for c in c0..c0+ch_per {
                for y in 0..x.h { for z in 0..x.w {
                    let id = x.idx(n,c,y,z);
                    x.data[id] = (x.data[id] - mean) / var;
                }}
            }
        }
    }
}

pub fn conv2d_nchw(
    x: &CpuTensor,
    w: &Vec<f32>, // [Cout, Cin, Kh, Kw]
    b: Option<&Vec<f32>>,
    cin: usize, cout: usize, kh: usize, kw: usize,
    stride: usize, pad: usize, dilation: usize
) -> CpuTensor {
    let out_h = (x.h + 2*pad - dilation*(kh-1) - 1)/stride + 1;
    let out_w = (x.w + 2*pad - dilation*(kw-1) - 1)/stride + 1;
    let mut y = CpuTensor::zeros(x.n, cout, out_h, out_w);
    for n in 0..x.n {
        for oc in 0..cout {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0f32;
                    for ic in 0..cin {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let iy = oh*stride + kh_i*dilation;
                                let ix = ow*stride + kw_i*dilation;
                                // account for padding
                                let sy = iy as isize - pad as isize;
                                let sx = ix as isize - pad as isize;
                                if sy>=0 && sx>=0 && (sy as usize) < x.h && (sx as usize) < x.w {
                                    let v = x.data[x.idx(n, ic, sy as usize, sx as usize)];
                                    let widx = (((oc*cin + ic)*kh + kh_i)*kw + kw_i) as usize;
                                    acc += v * w[widx];
                                }
                            }
                        }
                    }
                    if let Some(bias) = b { acc += bias[oc]; }
                    y.data[y.idx(n, oc, oh, ow)] = acc;
                }
            }
        }
    }
    y
}

pub fn nearest_upsample_x2(x: &CpuTensor) -> CpuTensor {
    let mut y = CpuTensor::zeros(x.n, x.c, x.h*2, x.w*2);
    for n in 0..x.n {
        for c in 0..x.c {
            for y0 in 0..x.h {
                for x0 in 0..x.w {
                    let v = x.data[x.idx(n,c,y0,x0)];
                    let yy = y0*2; let xx = x0*2;
                    y.data[y.idx(n,c,yy,xx)] = v;
                    y.data[y.idx(n,c,yy,xx+1)] = v;
                    y.data[y.idx(n,c,yy+1,xx)] = v;
                    y.data[y.idx(n,c,yy+1,xx+1)] = v;
                }
            }
        }
    }
    y
}
