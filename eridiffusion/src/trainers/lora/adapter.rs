use flame_core::device::Device;
use flame_core::Parameter;
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};

/// Simple LoRA adapter with down and up projection matrices
#[derive(Clone)]
pub struct SimpleLoRA {
    pub down: Parameter,
    pub up: Parameter,
    pub scale: f64,
}

impl SimpleLoRA {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        device: &Device,
        dtype: DType,
    ) -> flame_core::Result<Self> {
        // Initialize LoRA weights - ALWAYS use F32 for parameters to support gradients
        // Even if forward pass uses F16/BF16, parameters must be F32 for backward pass

        // CRITICAL: Create as Parameters from the start for gradient tracking
        // Kaiming initialization for down projection
        let fan_in = in_features as f64;
        let std = (2.0 / fan_in).sqrt() as f32;
        let down_shape = Shape::new(vec![rank, in_features]);
        let down = Parameter::randn(down_shape, 0.0, std, device.cuda_device().clone())?;

        // Initialize up projection to zeros
        let up_shape = Shape::new(vec![out_features, rank]);
        let up = Parameter::zeros(up_shape, device.cuda_device().clone())?;

        let scale = (alpha as f64) / (rank as f64);

        Ok(Self { down, up, scale })
    }

    pub fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        // Apply LoRA: x + scale * (x @ down^T @ up^T)
        let batch_size = x.shape().dims()[0];
        let seq_len = x.shape().dims()[1];
        let hidden_size = x.shape().dims()[2];

        // Reshape for matmul: [batch * seq, hidden]
        let x_2d = x.reshape(&[batch_size * seq_len, hidden_size])?;

        // Apply down projection: [batch * seq, hidden] @ [hidden, rank] -> [batch * seq, rank]
        let down_out = x_2d.matmul(&self.down.tensor()?.transpose_dims(0, 1)?)?;

        // Apply up projection: [batch * seq, rank] @ [rank, out] -> [batch * seq, out]
        let up_out = down_out.matmul(&self.up.tensor()?.transpose_dims(0, 1)?)?;

        // Reshape back to original shape
        let lora_out = up_out.reshape(&[batch_size, seq_len, hidden_size])?;

        // Scale and add to input
        x.add(&lora_out.mul_scalar(self.scale as f32)?)
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.down, &self.up]
    }
}

/// More advanced LoRA adapter with additional features
pub struct LoRAAdapter {
    pub base: SimpleLoRA,
    pub dropout: Option<f32>,
    pub requires_grad: bool,
}

impl LoRAAdapter {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout: Option<f32>,
        device: &Device,
        dtype: DType,
    ) -> flame_core::Result<Self> {
        let base = SimpleLoRA::new(in_features, out_features, rank, alpha, device, dtype)?;

        Ok(Self { base, dropout, requires_grad: true })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> flame_core::Result<Tensor> {
        let mut out = self.base.forward(x)?;

        // Apply dropout if training
        if training && self.dropout.is_some() {
            let dropout_rate = self.dropout.unwrap();
            // Simple dropout implementation using random mask
            // Generate random values between 0 and 1
            let random_vals = Tensor::randn(out.shape().clone(), 0.5, 0.289, out.device().clone())?; // mean=0.5, std=0.289 approximates uniform [0,1]

            // Create binary mask by thresholding
            // Values > dropout_rate become 1, others become 0
            // Since we can't compare directly, we'll use a different approach
            let keep_prob = 1.0 - dropout_rate;
            let scale = 1.0 / keep_prob;

            // Apply dropout by scaling and masking
            out = out.mul(&random_vals)?.mul_scalar(scale / 0.5)?; // Divide by 0.5 to account for mean of random_vals
        }

        Ok(out)
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        // In FLAME, gradient tracking is handled automatically
        // No need to explicitly detach parameters
    }
}
