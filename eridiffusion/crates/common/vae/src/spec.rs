#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeKind { Sdxl, Flux, Sd35 }

#[derive(Clone, Debug)]
pub struct VaeSpec {
    pub kind: VaeKind,
    pub path: String,
    pub latent_div: usize,
    pub latent_channels: usize,
    pub latent_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaePolicy { GpuFirst, CpuOnly }

