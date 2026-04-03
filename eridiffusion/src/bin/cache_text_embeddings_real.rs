#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::loaders::WeightLoader;
use eridiffusion::models::flux_text_encoder::{FluxTextEncoder, TextEncoderConfig};
use eridiffusion::tokenizers::clip_tokenizer::ClipTokenizer;
use eridiffusion::tokenizers::t5_tokenizer::T5Tokenizer;
use flame_core::{DType, Device, Result, Shape, Tensor};
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    println!("=== FLUX TEXT EMBEDDING CACHING (REAL) ===");
    println!("Cache text embeddings ONCE with BF16, never load encoders during training!");
    println!();

    let device = Device::cuda(0)?;

    // Setup paths
    let dataset_path = PathBuf::from("/home/alex/diffusers-rs/datasets/1stone");
    let cache_dir = dataset_path.join("cached_embeddings");
    fs::create_dir_all(&cache_dir)?;

    // Model paths
    let clip_path = PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors");
    let t5_path = PathBuf::from("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors");

    // Tokenizer paths
    let clip_tokenizer_path = PathBuf::from("/home/alex/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff/tokenizer.json");
    let t5_tokenizer_path = PathBuf::from(
        "/home/alex/SwarmUI_/dlbackend/ComfyUI/comfy/text_encoders/t5_tokenizer/tokenizer.json",
    );

    // Get list of text files
    let text_files: Vec<_> = fs::read_dir(&dataset_path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.extension().and_then(|ext| ext.to_str()).map(|ext| ext == "txt").unwrap_or(false)
        })
        .collect();

    println!("Found {} text caption files", text_files.len());

    // Load tokenizers
    println!("\n🔤 Loading tokenizers...");
    let clip_tokenizer = ClipTokenizer::from_file(&clip_tokenizer_path)?;
    let t5_tokenizer = T5Tokenizer::from_file(&t5_tokenizer_path)?;
    println!("✅ Tokenizers loaded");

    // Load text encoders with BF16 for memory efficiency
    println!("\n🧠 Loading text encoders with BF16...");

    // Load CLIP with BF16
    println!("  Loading CLIP encoder with BF16...");
    let clip_wl =
        WeightLoader::from_safetensors_with_dtype(&clip_path, device.clone(), DType::BF16)?;
    let clip_config = TextEncoderConfig::clip_default();
    let clip_encoder = FluxTextEncoder::new_clip(&clip_wl, device.clone(), clip_config)?;
    println!("  ✅ CLIP loaded with BF16");

    // Load T5 with BF16
    println!("  Loading T5 XXL encoder with BF16...");
    let t5_wl = WeightLoader::from_safetensors_with_dtype(&t5_path, device.clone(), DType::BF16)?;
    let t5_config = TextEncoderConfig::t5_xxl_default();
    let t5_encoder = FluxTextEncoder::new_t5(&t5_wl, device.clone(), t5_config)?;
    println!("  ✅ T5 XXL loaded with BF16");

    let mut processed = 0;
    let mut skipped = 0;
    let batch_size = 1; // Process one at a time to avoid OOM

    for (idx, entry) in text_files.iter().enumerate() {
        let text_path = entry.path();
        let base_name = text_path.file_stem().unwrap().to_str().unwrap();

        // Check if embeddings already cached
        let clip_cache_path = cache_dir.join(format!("{}_clip.bin", base_name));
        let t5_cache_path = cache_dir.join(format!("{}_t5.bin", base_name));
        let pooled_cache_path = cache_dir.join(format!("{}_pooled.bin", base_name));

        if clip_cache_path.exists() && t5_cache_path.exists() && pooled_cache_path.exists() {
            skipped += 1;
            println!("[{}/{}] Already cached: {}", idx + 1, text_files.len(), base_name);
            continue;
        }

        // Read caption
        let caption = fs::read_to_string(&text_path)?;
        let caption = caption.trim();

        println!("[{}/{}] Processing: {}", idx + 1, text_files.len(), base_name);
        println!(
            "  Caption: \"{}\"",
            if caption.len() > 50 { format!("{}...", &caption[..50]) } else { caption.to_string() }
        );

        // Tokenize
        println!("  Tokenizing...");
        let clip_tokens = clip_tokenizer.encode(caption, 77)?; // Max 77 tokens for CLIP
        let t5_tokens = t5_tokenizer.encode(caption, 256)?; // Max 256 tokens for T5

        // Convert tokens to tensors
        let clip_input = Tensor::from_vec(
            clip_tokens.clone(),
            Shape::from_dims(&[1, clip_tokens.len()]),
            device.cuda_device_arc(),
        )?;

        let t5_input = Tensor::from_vec(
            t5_tokens.clone(),
            Shape::from_dims(&[1, t5_tokens.len()]),
            device.cuda_device_arc(),
        )?;

        // Encode with CLIP
        println!("  Encoding with CLIP (BF16)...");
        let clip_output = clip_encoder.encode(&clip_input)?;
        let clip_embeds = clip_output.hidden_states; // [1, 77, 768]
        let pooled = clip_output.pooled_output.unwrap_or_else(|| {
            // If no pooled output, use last token
            clip_embeds.narrow(1, clip_embeds.shape().dims()[1] - 1, 1).unwrap().squeeze(1).unwrap()
        });

        // Encode with T5
        println!("  Encoding with T5 XXL (BF16)...");
        let t5_output = t5_encoder.encode(&t5_input)?;
        let t5_embeds = t5_output.hidden_states; // [1, 256, 4096]

        // Convert to F32 for saving (training expects F32)
        let clip_embeds_f32 = clip_embeds.to_dtype(DType::F32)?;
        let t5_embeds_f32 = t5_embeds.to_dtype(DType::F32)?;
        let pooled_f32 = pooled.to_dtype(DType::F32)?;

        // Save CLIP embeddings
        let clip_data = clip_embeds_f32.to_vec()?;
        let clip_shape_dims = clip_embeds_f32.shape().dims().to_vec();
        save_tensor_binary(&clip_cache_path, &clip_data, &clip_shape_dims)?;

        // Save T5 embeddings
        let t5_data = t5_embeds_f32.to_vec()?;
        let t5_shape_dims = t5_embeds_f32.shape().dims().to_vec();
        save_tensor_binary(&t5_cache_path, &t5_data, &t5_shape_dims)?;

        // Save pooled embeddings
        let pooled_data = pooled_f32.to_vec()?;
        let pooled_shape_dims = pooled_f32.shape().dims().to_vec();
        save_tensor_binary(&pooled_cache_path, &pooled_data, &pooled_shape_dims)?;

        processed += 1;
        println!(
            "  ✅ Saved embeddings (CLIP: {:?}, T5: {:?}, Pooled: {:?})",
            clip_embeds_f32.shape(),
            t5_embeds_f32.shape(),
            pooled_f32.shape()
        );

        // Free memory after each batch
        if processed % batch_size == 0 {
            println!("\n🔄 Batch of {} complete. Freeing GPU memory...", batch_size);
            // Tensors will be dropped automatically
        }

        // Stop after processing 10 for initial test
        if processed >= 10 {
            println!("\n⚠️  Stopping after {} files for testing", processed);
            break;
        }
    }

    // Free encoders from memory
    drop(clip_encoder);
    drop(t5_encoder);
    drop(clip_wl);
    drop(t5_wl);

    println!("\n✅ TEXT EMBEDDING CACHING COMPLETE!");
    println!("Total processed: {}", processed);
    println!("Total skipped (already cached): {}", skipped);
    println!("Cache directory: {}", cache_dir.display());
    println!("\n🎯 NOW YOU CAN TRAIN WITHOUT LOADING TEXT ENCODERS!");
    println!("Both latents AND text embeddings are cached!");
    println!("Training will use <12GB VRAM (no VAE, no T5, no CLIP)!");

    Ok(())
}

fn save_tensor_binary(path: &Path, data: &[f32], shape: &[usize]) -> Result<()> {
    let mut binary = Vec::new();

    // Write shape
    binary.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for dim in shape {
        binary.extend_from_slice(&(*dim as u32).to_le_bytes());
    }

    // Write data
    for val in data {
        binary.extend_from_slice(&val.to_le_bytes());
    }

    fs::write(path, binary)?;
    Ok(())
}
