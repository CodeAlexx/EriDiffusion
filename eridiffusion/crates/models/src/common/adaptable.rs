use adapters::adapter::AdapterSet;
use flame_core::Tensor;
use eridiffusion_core::Result;

pub struct AdaptableLinear {
    pub target: String,
    pub base_w: Tensor,   // [IN, OUT]
    pub bias: Option<Tensor>,
    pub adapters: Option<AdapterSet>,
}

impl AdaptableLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(set) = &self.adapters {
            if let Some(a) = set.get(&self.target) {
                let mut y = a.apply_linear(&self.base_w, x)?;
                if let Some(b) = &self.bias { y = y.add(b)?; }
                return Ok(y);
            }
        }
        let mut y = x.matmul(&self.base_w)?;
        if let Some(b) = &self.bias { y = y.add(b)?; }
        Ok(y)
    }
}

pub struct AdaptableConv2d {
    pub target: String,
    pub base_w: Tensor,   // [KH,KW,IC,OC]
    pub bias: Option<Tensor>,
    pub adapters: Option<AdapterSet>,
}

impl AdaptableConv2d {
    pub fn forward_nhwc(&self, x: &Tensor, stride:(usize,usize), pad:(usize,usize)) -> Result<Tensor> {
        if let Some(set) = &self.adapters {
            if let Some(a) = set.get(&self.target) {
                let mut y = a.apply_conv2d(&self.base_w, x, stride, pad)?;
                if let Some(b) = &self.bias { y = y.add(b)?; }
                return Ok(y);
            }
        }
        // No adapter: NHWC->NCHW conv
        let x_nc = x.permute(&[0,3,1,2])?;
        let w = self.base_w.permute(&[3,2,0,1])?;
        let y_nc = x_nc.conv2d(&w, None, stride.0, pad.0)?;
        let y_nc = if let Some(b) = &self.bias { y_nc.add(b)? } else { y_nc };
        Ok(y_nc.permute(&[0,2,3,1])?)
    }
}
