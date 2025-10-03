
//! Full VAE that stitches encoder+decoder and exposes encode/decode with a scaling factor.
//! You can load weights via safetensors (strict names) or use random init to run immediately.

#![allow(dead_code)]
use super::tensor::CpuTensor;
use super::encoder::{Encoder, EncWeights};
use super::decoder::{Decoder, DecWeights};
use super::safetensors::{StrictWeights};

#[derive(Clone, Debug)]
pub struct RealVae {
    pub enc: Encoder,
    pub dec: Decoder,
    pub scaling: f32,
}

impl RealVae {
    pub fn new_random(scaling: f32) -> Self {
        Self { enc: Encoder { w: EncWeights::random(), scaling },
               dec: Decoder { w: DecWeights::random(), scaling },
               scaling }
    }
    pub fn encode(&self, x: &CpuTensor) -> CpuTensor { self.enc.forward(x) }
    pub fn decode(&self, z: &CpuTensor) -> CpuTensor { self.dec.forward(z) }
}

/// Illustrative loader (optional): expects specific keys; if any missing, keep random weights.
pub fn try_load_from_safetensors(path: &std::path::Path, scaling: f32) -> RealVae {
    let mut vae = RealVae::new_random(scaling);
    if let Ok(mut sw) = StrictWeights::open(path) {
        macro_rules! load_into {
            ($dst:expr, $key:expr) => {
                if let Ok(bytes) = sw.get_raw($key) {
                    // interpret as f32
                    let n = bytes.len()/4;
                    let mut v = vec![0f32; n];
                    for i in 0..n {
                        let mut b = [0u8;4];
                        b.copy_from_slice(&bytes[i*4..(i+1)*4]);
                        v[i] = f32::from_le_bytes(b);
                    }
                    $dst.clone_from(&v);
                }
            };
        }
        // Encoder weights (example names; adapt to your header names)
        load_into!(vae.enc.w.c1_w, "encoder.c1.weight");
        load_into!(vae.enc.w.c1_b, "encoder.c1.bias");
        load_into!(vae.enc.w.c2_w, "encoder.c2.weight");
        load_into!(vae.enc.w.c2_b, "encoder.c2.bias");
        load_into!(vae.enc.w.c3_w, "encoder.c3.weight");
        load_into!(vae.enc.w.c3_b, "encoder.c3.bias");
        load_into!(vae.enc.w.c4_w, "encoder.c4.weight");
        load_into!(vae.enc.w.c4_b, "encoder.c4.bias");
        // Decoder
        load_into!(vae.dec.w.c1_w, "decoder.c1.weight");
        load_into!(vae.dec.w.c1_b, "decoder.c1.bias");
        load_into!(vae.dec.w.c2_w, "decoder.c2.weight");
        load_into!(vae.dec.w.c2_b, "decoder.c2.bias");
        load_into!(vae.dec.w.c3_w, "decoder.c3.weight");
        load_into!(vae.dec.w.c3_b, "decoder.c3.bias");
        load_into!(vae.dec.w.c4_w, "decoder.c4.weight");
        load_into!(vae.dec.w.c4_b, "decoder.c4.bias");
    }
    vae
}
