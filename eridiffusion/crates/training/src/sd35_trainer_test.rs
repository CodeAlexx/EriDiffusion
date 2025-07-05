//! Unit tests for SD 3.5 trainer

#[cfg(test)]
mod tests {
    use super::super::*;
    use eridiffusion_core::{Device, ModelInputs};
    use eridiffusion_models::{SD3Model, TextEncoder, VAE};
    use candle_core::{Tensor, DType};
    use std::collections::HashMap;
    
    /// Create mock text encoder for testing
    fn create_mock_text_encoder(hidden_size: usize, device: &Device) -> Result<TextEncoder> {
        // This would be a mock implementation
        Ok(TextEncoder::mock(hidden_size, device)?)
    }
    
    #[test]
    fn test_encode_text_triple() -> Result<()> {
        let device = Device::Cpu;
        
        // Create mock encoders
        let clip_l = create_mock_text_encoder(768, &device)?;
        let clip_g = create_mock_text_encoder(1280, &device)?;
        let t5 = create_mock_text_encoder(4096, &device)?;
        
        // Test encoding
        let prompts = vec!["A beautiful sunset".to_string()];
        let result = SD35Trainer::encode_text_triple(
            &prompts,
            &clip_l,
            &clip_g,
            &t5,
            77, // max_length
        )?;
        
        // Check dimensions
        assert_eq!(result.encoder_hidden_states.rank(), 3); // [batch, seq, hidden]
        assert_eq!(result.pooled_output.rank(), 2); // [batch, hidden]
        
        // Check hidden dimension is correct (2048 + 2048 + 4096 = 8192)
        assert_eq!(result.encoder_hidden_states.dim(2)?, 8192);
        
        Ok(())
    }
    
    #[test]
    fn test_prepare_training_inputs() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        
        // Create mock tensors
        let latents = Tensor::randn(batch_size, 16, 32, 32, DType::F32, &device)?;
        let timesteps = Tensor::rand(batch_size, DType::F32, &device)?;
        let text_embeddings = Tensor::randn(batch_size, 77, 8192, DType::F32, &device)?;
        let pooled_projections = Tensor::randn(batch_size, 2048, DType::F32, &device)?;
        
        // Prepare inputs
        let inputs = SD35Trainer::prepare_training_inputs(
            &latents,
            &timesteps,
            &text_embeddings,
            &pooled_projections,
        )?;
        
        // Verify inputs
        assert_eq!(inputs.latents.dims(), latents.dims());
        assert_eq!(inputs.timestep.dims(), timesteps.dims());
        assert!(inputs.encoder_hidden_states.is_some());
        assert!(inputs.pooled_projections.is_some());
        
        Ok(())
    }
    
    #[test]
    fn test_compute_flow_matching_loss() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        
        // Create mock tensors
        let model_output = Tensor::randn(batch_size, 16, 32, 32, DType::F32, &device)?;
        let target = Tensor::randn(batch_size, 16, 32, 32, DType::F32, &device)?;
        let timesteps = Tensor::rand(batch_size, DType::F32, &device)?;
        
        // Compute loss
        let loss = SD35Trainer::compute_flow_matching_loss(
            &model_output,
            &target,
            &timesteps,
            5.0, // snr_gamma
        )?;
        
        // Check loss is scalar
        assert_eq!(loss.rank(), 0);
        assert!(loss.to_scalar::<f32>()? >= 0.0);
        
        Ok(())
    }
    
    #[test]
    fn test_get_sd35_target_modules() -> Result<()> {
        // Test getting target modules for LoRA
        let modules = SD35Trainer::get_sd35_target_modules(false);
        
        // Check we have modules
        assert!(!modules.is_empty());
        
        // Check for expected patterns
        assert!(modules.iter().any(|m| m.contains("attn")));
        assert!(modules.iter().any(|m| m.contains("mlp")));
        assert!(modules.iter().any(|m| m.contains("joint_blocks")));
        
        // Test with text encoder
        let modules_with_te = SD35Trainer::get_sd35_target_modules(true);
        assert!(modules_with_te.len() > modules.len());
        assert!(modules_with_te.iter().any(|m| m.contains("text_model")));
        
        Ok(())
    }
    
    #[test]
    fn test_sd35_training_config() -> Result<()> {
        // Test config creation and defaults
        let config = SD35TrainingConfig::default();
        
        assert_eq!(config.model_variant, SD35ModelVariant::Large);
        assert_eq!(config.resolution, 1024);
        assert_eq!(config.num_train_timesteps, 1000);
        assert!(config.flow_matching);
        assert_eq!(config.t5_max_length, 256);
        
        Ok(())
    }
    
    #[test]
    fn test_velocity_prediction() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        
        // Create test data
        let original = Tensor::randn(batch_size, 16, 32, 32, DType::F32, &device)?;
        let noise = Tensor::randn_like(&original)?;
        let timesteps = Tensor::new(&[0.3f32, 0.7f32], &device)?;
        
        // Create noisy samples
        let t_expanded = timesteps.reshape((batch_size, 1, 1, 1))?;
        let noisy = &original * (1.0 - &t_expanded)? + &noise * &t_expanded;
        
        // Compute velocity target
        let velocity = (&original - &noise)? / (1.0 - &t_expanded)?;
        
        // Verify shapes
        assert_eq!(velocity.dims(), original.dims());
        
        Ok(())
    }
    
    #[test]
    fn test_snr_weighting() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 4;
        
        // Test SNR computation
        let timesteps = Tensor::new(&[0.1f32, 0.3, 0.5, 0.9], &device)?;
        let snr = &timesteps / (1.0 - &timesteps)?;
        
        // Check SNR increases with timestep
        let snr_vec = snr.to_vec1::<f32>()?;
        for i in 1..snr_vec.len() {
            assert!(snr_vec[i] > snr_vec[i-1]);
        }
        
        // Test weight computation
        let weight = (snr / (1.0 + snr)?).sqrt()?;
        let weight_vec = weight.to_vec1::<f32>()?;
        
        // Weights should be between 0 and 1
        for w in &weight_vec {
            assert!(*w >= 0.0 && *w <= 1.0);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_training_step_integration() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        
        // Create mock model and data
        let latents = Tensor::randn(batch_size, 16, 32, 32, DType::F32, &device)?;
        let text_embeddings = Tensor::randn(batch_size, 77, 8192, DType::F32, &device)?;
        let pooled = Tensor::randn(batch_size, 2048, DType::F32, &device)?;
        
        // Sample timesteps
        let timesteps = Tensor::rand(batch_size, DType::F32, &device)?;
        
        // Create noise
        let noise = Tensor::randn_like(&latents)?;
        
        // Create noisy latents
        let t_expanded = timesteps.reshape((batch_size, 1, 1, 1))?;
        let noisy_latents = &latents * (1.0 - &t_expanded)? + &noise * &t_expanded;
        
        // Prepare inputs
        let inputs = SD35Trainer::prepare_training_inputs(
            &noisy_latents,
            &timesteps,
            &text_embeddings,
            &pooled,
        )?;
        
        // Verify all required fields are present
        assert!(inputs.latents.dims()[0] == batch_size);
        assert!(inputs.encoder_hidden_states.is_some());
        assert!(inputs.pooled_projections.is_some());
        
        Ok(())
    }
}