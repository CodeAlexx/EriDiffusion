// Integration with your existing VAE code

use crate::cuda_alignment::{CudaAlignmentUtils, AlignedTensorOps, AlignedWeightLoader};

// Update your AutoEncoderKL to use aligned tensors
impl AutoEncoderKL {
    pub fn new(
        config: VAEConfig,
        device: Device,
        weights: std::collections::HashMap<String, Tensor>,
        max_resolution: Option<usize>,
    ) -> Result<Self> {
        let max_res = max_resolution.unwrap_or(1024);
        
        println!("Creating VAE for max resolution: {}x{}", max_res, max_res);
        
        // ALIGNMENT FIX: Align all weights before loading
        let target_dtype = DType::BF16; // or DType::F32
        let aligned_weights = AlignedWeightLoader::load_aligned_weights(&weights, target_dtype)?;
        
        // Validate alignment
        let unaligned = AlignedWeightLoader::validate_alignment(&aligned_weights, target_dtype);
        if !unaligned.is_empty() {
            println!("Warning: Some weights are still unaligned: {:?}", unaligned);
        }
        
        let mut encoder = Encoder::new(&config, device.clone(), max_res)?;
        let mut decoder = Decoder::new(&config, device.clone(), max_res)?;

        // Load aligned weights
        encoder.load_weights(&aligned_weights)?;
        decoder.load_weights(&aligned_weights)?;

        // Handle quant_conv with alignment
        let quant_conv = if config.use_quant_conv {
            let weight = aligned_weights.get("quant_conv.weight")
                .ok_or_else(|| flame_core::Error::InvalidOperation("quant_conv.weight not found".into()))?;
            let weight_shape = weight.shape().dims();
            let mut conv = Conv2d::new(weight_shape[1], weight_shape[0], weight_shape[2], 1, 0, device.cuda_device().clone())?;
            conv.weight = weight.ensure_cuda_aligned()?; // Apply alignment
            if let Some(bias) = aligned_weights.get("quant_conv.bias") {
                conv.bias = Some(bias.ensure_cuda_aligned()?);
            }
            Some(conv)
        } else {
            None
        };

        let post_quant_conv = if config.use_post_quant_conv {
            let weight = aligned_weights.get("post_quant_conv.weight")
                .ok_or_else(|| flame_core::Error::InvalidOperation("post_quant_conv.weight not found".into()))?;
            let weight_shape = weight.shape().dims();
            let mut conv = Conv2d::new(weight_shape[1], weight_shape[0], weight_shape[2], 1, 0, device.cuda_device().clone())?;
            conv.weight = weight.ensure_cuda_aligned()?; // Apply alignment
            if let Some(bias) = aligned_weights.get("post_quant_conv.bias") {
                conv.bias = Some(bias.ensure_cuda_aligned()?);
            }
            Some(conv)
        } else {
            None
        };

        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            config,
            device,
        })
    }

    pub fn encode(&self, x: &Tensor) -> Result<DiagonalGaussianDistribution> {
        // ALIGNMENT FIX: Ensure input tensor is aligned
        let aligned_input = x.ensure_cuda_aligned()?;
        let h = self.encoder.forward(&aligned_input)?;

        let moments = if let Some(qc) = &self.quant_conv {
            qc.forward(&h)?
        } else {
            h
        };

        DiagonalGaussianDistribution::new(moments)
    }
}

// Update your Encoder forward pass
impl Encoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        println!("Encoder forward - input shape: {:?}, dtype: {:?}", x.shape(), x.dtype());
        
        // ALIGNMENT FIX: Ensure input is aligned and in correct dtype
        let x = x.to_aligned_dtype(DType::BF16)?;
        
        let mut h = self.conv_in.forward(&x)?;

        // Ensure intermediate tensors remain aligned
        for block in &self.down_blocks {
            h = block.forward(&h)?;
            h = h.ensure_cuda_aligned()?; // Keep alignment through pipeline
        }

        h = self.mid_block.forward(&h)?;
        h = h.ensure_cuda_aligned()?;

        h = self.norm_out.forward(&h)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }
}

// Update weight loading with alignment
impl ResnetBlock {
    pub fn load_weights(&mut self, weights: &std::collections::HashMap<String, Tensor>, prefix: &str) -> Result<()> {
        // Load conv1 weights with alignment
        if let Some(weight) = weights.get(&format!("{}.conv1.weight", prefix)) {
            self.conv1.weight = weight.to_aligned_dtype(DType::BF16)?; // Already aligned from AlignedWeightLoader
        }
        if let Some(bias) = weights.get(&format!("{}.conv1.bias", prefix)) {
            self.conv1.bias = Some(bias.to_aligned_dtype(DType::BF16)?);
        }
        
        // Similar for other weights...
        if let Some(weight) = weights.get(&format!("{}.conv2.weight", prefix)) {
            self.conv2.weight = weight.to_aligned_dtype(DType::BF16)?;
        }
        if let Some(bias) = weights.get(&format!("{}.conv2.bias", prefix)) {
            self.conv2.bias = Some(bias.to_aligned_dtype(DType::BF16)?);
        }
        
        // Load norm weights
        if let Some(weight) = weights.get(&format!("{}.norm1.weight", prefix)) {
            self.norm1.weight = Some(weight.to_aligned_dtype(DType::BF16)?);
        }
        if let Some(bias) = weights.get(&format!("{}.norm1.bias", prefix)) {
            self.norm1.bias = Some(bias.to_aligned_dtype(DType::BF16)?);
        }
        
        if let Some(weight) = weights.get(&format!("{}.norm2.weight", prefix)) {
            self.norm2.weight = Some(weight.to_aligned_dtype(DType::BF16)?);
        }
        if let Some(bias) = weights.get(&format!("{}.norm2.bias", prefix)) {
            self.norm2.bias = Some(bias.to_aligned_dtype(DType::BF16)?);
        }
        
        // Load conv_shortcut if present
        if let Some(cs) = &mut self.conv_shortcut {
            if let Some(weight) = weights.get(&format!("{}.conv_shortcut.weight", prefix)) {
                cs.weight = weight.to_aligned_dtype(DType::BF16)?;
            }
            if let Some(bias) = weights.get(&format!("{}.conv_shortcut.bias", prefix)) {
                cs.bias = Some(bias.to_aligned_dtype(DType::BF16)?);
            }
        }
        
        Ok(())
    }
}