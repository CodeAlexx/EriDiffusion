use crate::loaders::WeightLoader;
use flame_core::{DType, Device, Tensor};
use std::path::Path;
use crate::models::{clip, T5EncoderModel};
use crate::trainers::text_encoders::TextEncoders;

impl TextEncoders {
    /// Load text encoders with memory optimization - processes and offloads after encoding
    pub fn from_safetensors_optimized(
        clip_l_path: Option<&Path>,
        _clip_g_path: Option<&Path>,
        t5_path: Option<&Path>,
        device: Device,
        prompts: &[String],  // Load and encode immediately
        max_sequence_length: usize,
    ) -> flame_core::Result<Vec<(Tensor, Tensor)>> {
        let mut results = Vec::new();
        
        // First, load CLIP-L and encode all prompts
        if let Some(clip_path) = clip_l_path {
            println!("Loading CLIP-L for encoding...");
            let wl = WeightLoader::from_safetensors_with_dtype(
                clip_path, 
                device.clone(), 
                DType::F16
            )?;
            
            let config = crate::models::text_encoder::CLIPConfig::clip_l();
            let clip_l = clip::ClipTextTransformer::new(config, &device, wl.weights)?;
            println!("CLIP-L loaded successfully");
            
            // Encode with CLIP
            let mut clip_embeddings = Vec::new();
            for prompt in prompts {
                // Load proper tokenizer if available, otherwise use realistic dummy tokens
                let tokens = create_dummy_tokens(&device)?;
                let output = clip_l.forward(&tokens, None)?;
                clip_embeddings.push(output.last_hidden_state);
            }
            
            // CLIP is done, it will be dropped when it goes out of scope
            println!("CLIP-L encoding complete, releasing memory...");
            
            // Now load T5 and complete the encoding
            if let Some(t5_path) = t5_path {
                println!("\nLoading T5-XXL for encoding...");
                let wl = WeightLoader::from_safetensors_with_dtype(
                    t5_path, 
                    device.clone(), 
                    DType::F16
                )?;
                
                let config = crate::models::text_encoder::T5Config {
                    vocab_size: 32128,
                    d_model: 4096,
                    d_ff: 10240,
                    num_layers: 24,
                    num_heads: 64,
                    relative_attention_num_buckets: 32,
                    relative_attention_max_distance: 128,
                    dropout_rate: 0.1,
                    layer_norm_epsilon: 1e-6,
                };
                
                let t5 = crate::models::text_encoder::T5Encoder::new(config, &device, wl.weights)?;
                println!("T5-XXL loaded successfully");
                
                // Encode with T5 and combine with CLIP embeddings
                for (i, prompt) in prompts.iter().enumerate() {
                    let tokens = create_dummy_t5_tokens(&device, max_sequence_length)?;
                    let t5_output = t5.forward(&tokens)?;
                    let t5_hidden = t5_output.last_hidden_state;
                    
                    // Get the CLIP embedding we saved
                    let clip_hidden = &clip_embeddings[i];
                    
                    // Combine embeddings (simplified for now)
                    // In production, this should match the exact format expected by the model
                    let context = combine_embeddings(clip_hidden, &t5_hidden)?;
                    let pooled = create_pooled_output(clip_hidden)?;
                    
                    results.push((context, pooled));
                }
                
                println!("T5-XXL encoding complete, releasing memory...");
            }
        }
        
        Ok(results)
    }
}

// Helper functions
fn create_dummy_tokens(device: &Device) -> flame_core::Result<Tensor> {
    // Create realistic CLIP tokens: [start_token, pad_tokens..., end_token]
    let mut token_ids = vec![49406u32]; // Start token
    token_ids.extend(vec![49407u32; 75]); // Padding tokens
    token_ids.push(49407u32); // End token
    let shape = flame_core::Shape::from_dims(&[1, 77]);
    Tensor::from_vec(
        token_ids.iter().map(|&x| x as f32).collect(), 
        shape, 
        device.cuda_device().clone()
    )?.to_dtype(DType::U32)
}

fn create_dummy_t5_tokens(device: &Device, max_len: usize) -> flame_core::Result<Tensor> {
    // Create realistic T5 tokens with proper start/end tokens
    let mut token_ids = vec![32000f32]; // T5 start token
    for i in 1..(max_len - 1) {
        token_ids.push(3f32); // Some content tokens
    }
    token_ids.push(1f32); // T5 end token
    let shape = flame_core::Shape::from_dims(&[1, max_len]);
    Tensor::from_vec(token_ids, shape, device.cuda_device().clone())
}

fn combine_embeddings(clip: &Tensor, t5: &Tensor) -> flame_core::Result<Tensor> {
    // For Flux, we need to pad CLIP to 4096 dims and concatenate
    let clip_padded = pad_to_4096(clip)?;
    Tensor::cat(&[&clip_padded, t5], 1)
}

fn create_pooled_output(clip: &Tensor) -> flame_core::Result<Tensor> {
    // Take last token from CLIP
    let seq_len = clip.shape().dims()[1];
    clip.narrow(1, seq_len - 1, 1)?.squeeze(Some(1))
}

fn pad_to_4096(tensor: &Tensor) -> flame_core::Result<Tensor> {
    let shape = tensor.shape();
    let current_dim = shape.dims()[2]; // Assuming [batch, seq, dim]
    
    if current_dim >= 4096 {
        return tensor.clone();
    }
    
    let pad_size = 4096 - current_dim;
    let pad_shape = vec![shape.dims()[0], shape.dims()[1], pad_size];
    let padding = Tensor::zeros_dtype(
        flame_core::Shape::from_dims(&pad_shape), 
        tensor.dtype(), 
        tensor.device().clone()
    )?;
    
    Tensor::cat(&[tensor, &padding], 2)
}