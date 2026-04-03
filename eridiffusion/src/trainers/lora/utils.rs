use super::adapter::SimpleLoRA;
use flame_core::device::Device;
use flame_core::Parameter;
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};

/// Apply LoRA transformation to a tensor
pub fn apply_lora_to_tensor(
    x: &Tensor,
    lora: &SimpleLoRA,
    scale: Option<f64>,
) -> flame_core::Result<Tensor> {
    let scale = scale.unwrap_or(lora.scale);

    // Get dimensions
    let original_shape = x.shape().dims();
    let batch_seq = original_shape[0] * original_shape[1];
    let hidden = original_shape[2];

    // Reshape to 2D for matmul
    let x_2d = x.reshape(&[batch_seq, hidden])?;

    // Apply LoRA: down projection then up projection
    let down_out = x_2d.matmul(&lora.down.tensor()?.transpose_dims(0, 1)?)?;
    let up_out = down_out.matmul(&lora.up.tensor()?.transpose_dims(0, 1)?)?;

    // Reshape back and scale
    let lora_out = up_out.reshape(original_shape)?;
    let scaled = lora_out.mul_scalar(scale as f32)?;

    // Add to original
    x.add(&scaled)
}

/// Merge LoRA weights into base weights (for inference)
pub fn merge_lora_weights(
    base_weight: &Tensor,
    lora: &SimpleLoRA,
    scale: Option<f64>,
) -> flame_core::Result<Tensor> {
    let scale = scale.unwrap_or(lora.scale);

    // Compute LoRA update: scale * up @ down
    let up_tensor = lora.up.tensor()?;
    let down_tensor = lora.down.tensor()?;
    let lora_update = up_tensor.matmul(&down_tensor)?.mul_scalar(scale as f32)?;

    // Add to base weight
    base_weight.add(&lora_update)
}

/// Initialize LoRA adapters with specific initialization strategy
pub fn init_lora_weights(
    rank: usize,
    in_features: usize,
    out_features: usize,
    alpha: f32,
    init_strategy: &str,
    device: &Device,
) -> flame_core::Result<SimpleLoRA> {
    match init_strategy {
        "kaiming" => {
            // Default Kaiming initialization
            SimpleLoRA::new(in_features, out_features, rank, alpha, device, DType::F32)
        }
        "xavier" => {
            // Xavier/Glorot initialization
            let fan_in = in_features as f64;
            let fan_out = rank as f64;
            let std = (2.0 / (fan_in + fan_out)).sqrt();

            let down_init = Tensor::randn(
                Shape::new(vec![rank, in_features]),
                0.0,
                std as f32,
                device.cuda_device().clone(),
            )?;
            let down = Parameter::new(down_init);

            let up_init =
                Tensor::zeros(Shape::new(vec![out_features, rank]), device.cuda_device().clone())?;
            let up = Parameter::new(up_init);

            let scale = (alpha as f64) / (rank as f64);
            Ok(SimpleLoRA { down, up, scale })
        }
        "normal" => {
            // Simple normal initialization
            let down_init = Tensor::randn(
                Shape::new(vec![rank, in_features]),
                0.0,
                0.02,
                device.cuda_device().clone(),
            )?;
            let down = Parameter::new(down_init);

            let up_init =
                Tensor::zeros(Shape::new(vec![out_features, rank]), device.cuda_device().clone())?;
            let up = Parameter::new(up_init);

            let scale = (alpha as f64) / (rank as f64);
            Ok(SimpleLoRA { down, up, scale })
        }
        _ => {
            // Default to Kaiming
            SimpleLoRA::new(in_features, out_features, rank, alpha, device, DType::F32)
        }
    }
}

/// Calculate effective rank of LoRA adapter (for analysis)
pub fn calculate_effective_rank(lora: &SimpleLoRA) -> flame_core::Result<usize> {
    // Compute singular values of the LoRA product
    let up_tensor = lora.up.tensor()?;
    let down_tensor = lora.down.tensor()?;
    let _product = up_tensor.matmul(&down_tensor)?;

    // For now, return the configured rank
    // TODO: Implement SVD to calculate actual effective rank
    Ok(down_tensor.shape().dims()[0])
}

/// Scale LoRA weights by a factor (useful for merging multiple LoRAs)
pub fn scale_lora_weights(lora: &mut SimpleLoRA, factor: f64) -> flame_core::Result<()> {
    lora.scale *= factor;
    Ok(())
}
