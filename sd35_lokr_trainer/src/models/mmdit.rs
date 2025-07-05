use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

// Simplified MMDiT wrapper - we'll use the candle implementation directly
pub struct MMDiT {
    inner: candle_transformers::models::mmdit::model::MMDiT,
}

impl MMDiT {
    pub fn new_sd3_5_large(use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        // Use the SD3.5 large config from candle
        let config = candle_transformers::models::mmdit::model::Config::sd3_5_large();
        let inner = candle_transformers::models::mmdit::model::MMDiT::new(&config, use_flash_attn, vb)?;
        
        Ok(Self { inner })
    }
    
    pub fn forward(
        &self,
        x: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<Tensor> {
        Ok(self.inner.forward(x, timesteps, context, y, skip_layers)?)
    }
}