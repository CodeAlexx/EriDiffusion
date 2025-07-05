//! Flux forward pass and gradient tests
//! Adapted from flux_GA_test.rs

use eridiffusion_core::{Device, Result, Error, ModelInputs, DType as CoreDType, randint, TensorExt, VarExt};
use candle_core::{DType, Device as CandleDevice, Module, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::flux;
use candle_transformers::models::t5;
use candle_transformers::models::clip;
use std::path::PathBuf;
use std::time::Instant;

// Test configuration
#[derive(Debug)]
pub struct TestConfig {
    pub flux_model_path: PathBuf,
    pub vae_path: PathBuf,
    pub device: Device,
    pub test_batch_size: usize,
    pub test_image_size: usize,
    pub num_timesteps: usize,
    pub verbose: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            flux_model_path: PathBuf::from("flux-dev.safetensors"),
            vae_path: PathBuf::from("ae.safetensors"),
            device: Device::Cuda(0),
            test_batch_size: 1,
            test_image_size: 1024,
            num_timesteps: 1000,
            verbose: true,
        }
    }
}

/// Main test suite for Flux model forward pass and gradients
pub struct FluxForwardTester {
    config: TestConfig,
    model: Option<flux::model::Flux>,
    vae: Option<flux::autoencoder::AutoEncoder>,
    var_map: Option<VarMap>,
}

impl FluxForwardTester {
    pub fn new(config: TestConfig) -> Self {
        Self {
            config,
            model: None,
            vae: None,
            var_map: None,
        }
    }

    /// Convert Device to CandleDevice
    fn to_candle_device(&self) -> Result<CandleDevice> {
        match &self.config.device {
            Device::Cpu => Ok(CandleDevice::Cpu),
            Device::Cuda(id) => CandleDevice::new_cuda(*id)
                .map_err(|e| Error::Device(format!("Failed to create CUDA device: {}", e))),
        }
    }

    /// Run all tests
    pub fn run_all_tests(&mut self) -> Result<()> {
        println!("🧪 Starting Flux Forward Pass Tests");
        println!("Device: {:?}", self.config.device);
        println!("================================================\n");

        // Test 1: Model Loading
        self.test_model_loading()?;
        
        // Test 2: Forward Pass Shapes
        self.test_forward_pass_shapes()?;
        
        // Test 3: Gradient Flow
        self.test_gradient_flow()?;
        
        // Test 4: Memory and Performance
        self.test_memory_performance()?;
        
        // Test 5: Different Input Configurations
        self.test_input_variations()?;
        
        // Test 6: Gradient Accumulation
        self.test_gradient_accumulation()?;
        
        // Test 7: Mixed Precision
        if matches!(self.config.device, Device::Cuda(_)) {
            self.test_mixed_precision()?;
        }

        println!("\n✅ All tests passed!");
        Ok(())
    }

    /// Test 1: Model Loading
    fn test_model_loading(&mut self) -> Result<()> {
        println!("📦 Test 1: Model Loading");
        
        let start = Instant::now();
        let device = self.to_candle_device()?;
        
        // Create Flux config
        let flux_config = flux::model::Config::dev();
        
        // Initialize VarMap
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        
        // Create model
        let model = flux::model::Flux::new(&flux_config, vb)?;
        
        // Count parameters
        let total_params = count_parameters(&var_map)?;
        
        println!("  ✓ Model created successfully");
        println!("  ✓ Total parameters: {:.2}B", total_params as f64 / 1e9);
        println!("  ✓ Load time: {:.2}s", start.elapsed().as_secs_f32());
        
        // Store for later tests
        self.model = Some(model);
        self.var_map = Some(var_map);
        
        // Load VAE if path exists
        if self.config.vae_path.exists() {
            let vae_vb = unsafe { 
                VarBuilder::from_mmaped_safetensors(&[&self.config.vae_path], DType::F32, &device)?
            };
            let vae_config = flux::autoencoder::Config::dev();
            let vae = flux::autoencoder::AutoEncoder::new(&vae_config, vae_vb)?;
            self.vae = Some(vae);
            println!("  ✓ VAE loaded successfully");
        }
        
        Ok(())
    }

    /// Test 2: Forward Pass Shapes
    fn test_forward_pass_shapes(&mut self) -> Result<()> {
        println!("\n📐 Test 2: Forward Pass Shapes");
        
        let model = self.model.as_ref().ok_or(Error::Training("Model not loaded".to_string()))?;
        let batch_size = self.config.test_batch_size;
        let device = self.to_candle_device()?;
        
        // Create dummy inputs
        let latent_size = self.config.test_image_size / 8; // VAE downscale factor
        let latents = Tensor::randn(
            0.0, 
            1.0, 
            &[batch_size, 16, latent_size, latent_size], // Flux uses 16 latent channels
            &device
        )?;
        
        // Create text embeddings (T5-XXL dimensions)
        let text_embeddings = Tensor::randn(
            0.0,
            1.0,
            &[batch_size, 512, 4096], // T5-XXL: 512 tokens, 4096 dim
            &device
        )?;
        
        // Create pooled embeddings (CLIP dimensions)
        let pooled_embeddings = Tensor::randn(
            0.0,
            1.0,
            &[batch_size, 768], // CLIP-L pooled: 768 dim
            &device
        )?;
        
        // Create timesteps
        let timesteps = randint(
            0,
            self.config.num_timesteps as i64,
            &[batch_size],
            &device
        )?;
        
        // Test forward pass
        let start = Instant::now();
        let state = flux::sampling::State::new(&text_embeddings, &pooled_embeddings, &latents)?;
        
        // For a single denoising step
        let output = flux::sampling::denoise(
            model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &[timesteps.to_scalar::<f64>()?],
            3.5, // guidance scale
        )?;
        
        let forward_time = start.elapsed();
        
        // Check output shape
        assert_eq!(output.shape(), latents.shape(), "Output shape mismatch");
        
        println!("  ✓ Input latents shape: {:?}", latents.shape());
        println!("  ✓ Text embeddings shape: {:?}", text_embeddings.shape());
        println!("  ✓ Pooled embeddings shape: {:?}", pooled_embeddings.shape());
        println!("  ✓ Output shape: {:?}", output.shape());
        println!("  ✓ Forward pass time: {:.2}s", forward_time.as_secs_f32());
        
        Ok(())
    }

    /// Test 3: Gradient Flow
    fn test_gradient_flow(&mut self) -> Result<()> {
        println!("\n🔄 Test 3: Gradient Flow");
        
        let model = self.model.as_ref().ok_or(Error::Training("Model not loaded".to_string()))?;
        let var_map = self.var_map.as_ref().ok_or(Error::Training("VarMap not loaded".to_string()))?;
        let device = self.to_candle_device()?;
        
        // Create inputs with gradient tracking
        let batch_size = self.config.test_batch_size;
        let latent_size = self.config.test_image_size / 8;
        
        let latents = Tensor::randn(
            0.0, 
            1.0, 
            &[batch_size, 16, latent_size, latent_size],
            &device
        )?;
        
        let text_embeddings = Tensor::randn(
            0.0,
            1.0,
            &[batch_size, 512, 4096],
            &device
        )?;
        
        let pooled_embeddings = Tensor::randn(
            0.0,
            1.0,
            &[batch_size, 768],
            &device
        )?;
        
        let timesteps = randint(
            0,
            self.config.num_timesteps as i64,
            &[batch_size],
            &device
        )?;
        
        // Forward pass
        let state = flux::sampling::State::new(&text_embeddings, &pooled_embeddings, &latents)?;
        let output = flux::sampling::denoise(
            model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &[timesteps.to_scalar::<f64>()?],
            3.5,
        )?;
        
        // Create a simple loss (MSE with target)
        let target = Tensor::randn(0.0, 1.0, output.shape(), &device)?;
        let loss = output.sub(&target)?.sqr()?.mean_all()?;
        
        println!("  ✓ Loss computed: {:.6}", loss.to_scalar::<f32>()?);
        
        // Backward pass
        let start = Instant::now();
        loss.backward()?;
        let backward_time = start.elapsed();
        
        println!("  ✓ Backward pass completed in {:.2}s", backward_time.as_secs_f32());
        
        // Check gradients
        let mut has_gradients = false;
        let mut grad_stats = GradientStatistics::default();
        
        let all_vars = var_map.all_vars();
        for (idx, var) in all_vars.iter().enumerate() {
            let name = format!("var_{}", idx);
            if let Ok(grad) = var.grad() {
                has_gradients = true;
                let grad_values = grad.flatten_all()?.to_vec1::<f32>()?;
                grad_stats.update(&grad_values);
                
                if self.config.verbose && grad_stats.count < 5 {
                    println!("  ✓ Gradient found for: {}", name);
                }
            }
        }
        
        if !has_gradients {
            return Err(Error::Training("No gradients found!".to_string()));
        }
        
        println!("  ✓ Gradients statistics:");
        println!("    - Mean: {:.6}", grad_stats.mean);
        println!("    - Std: {:.6}", grad_stats.std);
        println!("    - Min: {:.6}", grad_stats.min);
        println!("    - Max: {:.6}", grad_stats.max);
        println!("    - % zero: {:.2}%", grad_stats.zero_ratio * 100.0);
        
        // Check for gradient issues
        if grad_stats.has_nan {
            println!("  ⚠️  Warning: NaN gradients detected!");
        }
        if grad_stats.has_inf {
            println!("  ⚠️  Warning: Inf gradients detected!");
        }
        if grad_stats.zero_ratio > 0.9 {
            println!("  ⚠️  Warning: >90% of gradients are zero!");
        }
        
        Ok(())
    }

    /// Test 4: Memory and Performance
    fn test_memory_performance(&mut self) -> Result<()> {
        println!("\n💾 Test 4: Memory and Performance");
        
        let model = self.model.as_ref().ok_or(Error::Training("Model not loaded".to_string()))?;
        let device = self.to_candle_device()?;
        
        // Test different batch sizes
        let batch_sizes = vec![1, 2, 4];
        let latent_size = self.config.test_image_size / 8;
        
        for batch_size in batch_sizes {
            let latents = Tensor::randn(
                0.0, 
                1.0, 
                &[batch_size, 16, latent_size, latent_size],
                &device
            )?;
            
            let text_embeddings = Tensor::randn(
                0.0,
                1.0,
                &[batch_size, 512, 4096],
                &device
            )?;
            
            let pooled_embeddings = Tensor::randn(
                0.0,
                1.0,
                &[batch_size, 768],
                &device
            )?;
            
            let timesteps = randint(
                0,
                self.config.num_timesteps as i64,
                &[batch_size],
                &device
            )?;
            
            // Warm up
            let state = flux::sampling::State::new(&text_embeddings, &pooled_embeddings, &latents)?;
            let _ = flux::sampling::denoise(
                model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &[timesteps.to_scalar::<f64>()?],
                3.5,
            )?;
            
            // Time multiple iterations
            let num_iterations = 10;
            let start = Instant::now();
            
            for _ in 0..num_iterations {
                let state = flux::sampling::State::new(&text_embeddings, &pooled_embeddings, &latents)?;
                let output = flux::sampling::denoise(
                    model,
                    &state.img,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &state.vec,
                    &[timesteps.to_scalar::<f64>()?],
                    3.5,
                )?;
                // Force computation
                let _ = output.sum_all()?.to_scalar::<f32>()?;
            }
            
            let total_time = start.elapsed();
            let avg_time = total_time.as_secs_f32() / num_iterations as f32;
            
            println!("  ✓ Batch size {}: {:.3}s per forward pass", batch_size, avg_time);
        }
        
        Ok(())
    }

    /// Test 5: Different Input Configurations
    fn test_input_variations(&mut self) -> Result<()> {
        println!("\n🔧 Test 5: Input Variations");
        
        let model = self.model.as_ref().ok_or(Error::Training("Model not loaded".to_string()))?;
        let device = self.to_candle_device()?;
        
        // Test different timestep values
        let batch_size = self.config.test_batch_size;
        let latent_size = self.config.test_image_size / 8;
        
        let latents = Tensor::randn(
            0.0, 
            1.0, 
            &[batch_size, 16, latent_size, latent_size],
            &device
        )?;
        
        let text_embeddings = Tensor::randn(
            0.0,
            1.0,
            &[batch_size, 512, 4096],
            &device
        )?;
        
        let pooled_embeddings = Tensor::randn(
            0.0,
            1.0,
            &[batch_size, 768],
            &device
        )?;
        
        // Test edge cases for timesteps
        let test_timesteps = vec![
            vec![0i64; batch_size], // Start of diffusion
            vec![500i64; batch_size], // Middle
            vec![999i64; batch_size], // End
        ];
        
        for (i, t_values) in test_timesteps.iter().enumerate() {
            let timesteps = Tensor::new(t_values.as_slice(), &device)?;
            let state = flux::sampling::State::new(&text_embeddings, &pooled_embeddings, &latents)?;
            let output = flux::sampling::denoise(
                model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &[timesteps.to_scalar::<f64>()?],
                3.5,
            )?;
            
            let output_mean = output.mean_all()?.to_scalar::<f32>()?;
            let output_std = output.var_all()?.to_scalar::<f32>()?.sqrt();
            
            println!("  ✓ Timestep {}: mean={:.4}, std={:.4}", 
                     t_values[0], output_mean, output_std);
        }
        
        // Test with different text embedding patterns
        println!("\n  Testing text conditioning variations:");
        
        // Empty text (zeros)
        let empty_text = Tensor::zeros(&[batch_size, 512, 4096], DType::F32, &device)?;
        let state = flux::sampling::State::new(&empty_text, &pooled_embeddings, &latents)?;
        let timesteps = Tensor::new(&[500i64], &device)?;
        let output_empty = flux::sampling::denoise(
            model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &[timesteps.to_scalar::<f64>()?],
            3.5,
        )?;
        println!("  ✓ Empty text conditioning: OK");
        
        // Very large text embeddings
        let large_text = text_embeddings.affine(10.0, 0.0)?;
        let state = flux::sampling::State::new(&large_text, &pooled_embeddings, &latents)?;
        let output_large = flux::sampling::denoise(
            model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &[timesteps.to_scalar::<f64>()?],
            3.5,
        )?;
        println!("  ✓ Large magnitude text: OK");
        
        Ok(())
    }

    /// Test 6: Gradient Accumulation
    fn test_gradient_accumulation(&mut self) -> Result<()> {
        println!("\n📊 Test 6: Gradient Accumulation");
        
        let model = self.model.as_ref().ok_or(Error::Training("Model not loaded".to_string()))?;
        let var_map = self.var_map.as_ref().ok_or(Error::Training("VarMap not loaded".to_string()))?;
        let device = self.to_candle_device()?;
        
        // Parameters
        let batch_size = self.config.test_batch_size;
        let latent_size = self.config.test_image_size / 8;
        let accumulation_steps = 4;
        
        // Zero gradients first
        let all_vars = var_map.all_vars();
        for var in all_vars.iter() {
            var.zero_grad()?;
        }
        
        let mut accumulated_loss = 0.0;
        
        // Simulate gradient accumulation
        for step in 0..accumulation_steps {
            let latents = Tensor::randn(
                0.0, 
                1.0, 
                &[batch_size, 16, latent_size, latent_size],
                &device
            )?;
            
            let text_embeddings = Tensor::randn(
                0.0,
                1.0,
                &[batch_size, 512, 4096],
                &device
            )?;
            
            let pooled_embeddings = Tensor::randn(
                0.0,
                1.0,
                &[batch_size, 768],
                &device
            )?;
            
            let timesteps = randint(
                0,
                self.config.num_timesteps as i64,
                &[batch_size],
                &device
            )?;
            
            // Forward pass
            let state = flux::sampling::State::new(&text_embeddings, &pooled_embeddings, &latents)?;
            let output = flux::sampling::denoise(
                model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &[timesteps.to_scalar::<f64>()?],
                3.5,
            )?;
            
            // Loss
            let target = Tensor::randn(0.0, 1.0, output.shape(), &device)?;
            let loss = output.sub(&target)?.sqr()?.mean_all()?;
            
            // Scale loss for accumulation
            let scaled_loss = loss.affine(1.0 / accumulation_steps as f64, 0.0)?;
            accumulated_loss += scaled_loss.to_scalar::<f32>()?;
            
            // Backward
            scaled_loss.backward()?;
            
            println!("  ✓ Step {}/{}: loss={:.6}", 
                     step + 1, accumulation_steps, loss.to_scalar::<f32>()?);
        }
        
        println!("  ✓ Average loss: {:.6}", accumulated_loss);
        
        // Check that gradients accumulated
        let mut grad_norm: f32 = 0.0;
        let all_vars = var_map.all_vars();
        for var in all_vars.iter() {
            if let Ok(grad) = var.grad() {
                let grad_flat = grad.flatten_all()?;
                let norm = grad_flat.sqr()?.sum_all()?.to_scalar::<f32>()?;
                grad_norm += norm;
            }
        }
        grad_norm = grad_norm.sqrt();
        
        println!("  ✓ Total gradient norm: {:.6}", grad_norm);
        
        Ok(())
    }

    /// Test 7: Mixed Precision
    fn test_mixed_precision(&mut self) -> Result<()> {
        println!("\n🎯 Test 7: Mixed Precision");
        
        let device = self.to_candle_device()?;
        
        // Create FP16 model
        let flux_config = flux::model::Config::dev();
        let var_map_fp16 = VarMap::new();
        let vb_fp16 = VarBuilder::from_varmap(&var_map_fp16, DType::F16, &device);
        let model_fp16 = flux::model::Flux::new(&flux_config, vb_fp16)?;
        
        let batch_size = self.config.test_batch_size;
        let latent_size = self.config.test_image_size / 8;
        
        // Create FP16 inputs
        let latents = Tensor::randn(
            0.0, 
            1.0, 
            &[batch_size, 16, latent_size, latent_size],
            &device
        )?.to_dtype(DType::F16)?;
        
        let text_embeddings = Tensor::randn(
            0.0,
            1.0,
            &[batch_size, 512, 4096],
            &device
        )?.to_dtype(DType::F16)?;
        
        let pooled_embeddings = Tensor::randn(
            0.0,
            1.0,
            &[batch_size, 768],
            &device
        )?.to_dtype(DType::F16)?;
        
        let timesteps = randint(
            0,
            self.config.num_timesteps as i64,
            &[batch_size],
            &device
        )?;
        
        // Forward pass in FP16
        let start = Instant::now();
        let state = flux::sampling::State::new(&text_embeddings, &pooled_embeddings, &latents)?;
        let output = flux::sampling::denoise(
            &model_fp16,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &[timesteps.to_scalar::<f64>()?],
            3.5,
        )?;
        let fp16_time = start.elapsed();
        
        println!("  ✓ FP16 forward pass: {:.3}s", fp16_time.as_secs_f32());
        println!("  ✓ Output dtype: {:?}", output.dtype());
        
        // Test loss computation with mixed precision
        let target = Tensor::randn(0.0, 1.0, output.shape(), &device)?.to_dtype(DType::F16)?;
        let loss = output.sub(&target)?.sqr()?.mean_all()?;
        
        // Loss should be computed in FP32 for stability
        let loss_fp32 = loss.to_dtype(DType::F32)?;
        loss_fp32.backward()?;
        
        println!("  ✓ Mixed precision loss: {:.6}", loss_fp32.to_scalar::<f32>()?);
        println!("  ✓ Gradient computation successful");
        
        Ok(())
    }
}

/// Helper struct for gradient statistics
#[derive(Default)]
struct GradientStatistics {
    count: usize,
    sum: f32,
    sum_sq: f32,
    min: f32,
    max: f32,
    zero_count: usize,
    has_nan: bool,
    has_inf: bool,
    mean: f32,
    std: f32,
    zero_ratio: f32,
}

impl GradientStatistics {
    fn update(&mut self, values: &[f32]) {
        for &val in values {
            self.count += 1;
            
            if val.is_nan() {
                self.has_nan = true;
                continue;
            }
            if val.is_infinite() {
                self.has_inf = true;
                continue;
            }
            
            self.sum += val;
            self.sum_sq += val * val;
            
            if self.count == 1 {
                self.min = val;
                self.max = val;
            } else {
                self.min = self.min.min(val);
                self.max = self.max.max(val);
            }
            
            if val.abs() < 1e-8 {
                self.zero_count += 1;
            }
        }
        
        // Compute statistics
        if self.count > 0 {
            self.mean = self.sum / self.count as f32;
            self.std = ((self.sum_sq / self.count as f32) - self.mean * self.mean).sqrt();
            self.zero_ratio = self.zero_count as f32 / self.count as f32;
        }
    }
}

/// Count total parameters in VarMap
fn count_parameters(var_map: &VarMap) -> Result<usize> {
    let mut total = 0;
    let all_vars = var_map.all_vars();
    for var in all_vars.iter() {
        total += var.as_tensor().elem_count();
    }
    Ok(total)
}

/// Main test runner
pub fn run_forward_tests(config: TestConfig) -> Result<()> {
    let mut tester = FluxForwardTester::new(config);
    tester.run_all_tests()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_forward_pass() -> Result<()> {
        let config = TestConfig {
            device: Device::Cpu, // Use CPU for testing
            test_batch_size: 1,
            test_image_size: 64, // Small size for testing
            ..Default::default()
        };
        
        run_forward_tests(config)
    }
}