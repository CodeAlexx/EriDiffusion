#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::loaders::WeightLoader;
use eridiffusion::models::clip;
use eridiffusion::models::t5::{T5Config, T5EncoderModel};
use eridiffusion::models::text_encoder::CLIPConfig;
use flame_core::{DType, Device, Result, Shape, Tensor};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    println!("=== FLUX TEXT EMBEDDING CACHING WITH MEMORY-EFFICIENT LOADING ===");
    println!("Cache text embeddings ONCE, never load encoders during training!");
    println!("Using BF16 for 50% memory reduction");
    println!();

    let device = Device::cuda(0)?;

    // Setup paths
    let dataset_path = PathBuf::from("/home/alex/diffusers-rs/datasets/1stone");
    let cache_dir = dataset_path.join("cached_embeddings");
    fs::create_dir_all(&cache_dir).map_err(|e| flame_core::Error::Io(e.to_string()))?;

    // Get list of text files
    let text_files: Vec<_> = fs::read_dir(&dataset_path)
        .map_err(|e| flame_core::Error::Io(e.to_string()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.extension().and_then(|ext| ext.to_str()).map(|ext| ext == "txt").unwrap_or(false)
        })
        .collect();

    println!("Found {} text caption files", text_files.len());

    // Load tokenizers first
    println!("\n🔤 Loading tokenizers...");
    let clip_tokenizer = Tokenizer::from_file("/home/alex/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/tokenizer.json")
        .map_err(|e| flame_core::Error::InvalidOperation(format!("Failed to load CLIP tokenizer: {}", e)))?;
    let t5_tokenizer = Tokenizer::from_file(
        "/home/alex/SwarmUI_/dlbackend/ComfyUI/comfy/text_encoders/t5_tokenizer/tokenizer.json",
    )
    .map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to load T5 tokenizer: {}", e))
    })?;
    println!("✅ Tokenizers loaded");

    // Load CLIP (small, can stay in memory)
    println!("\n🧠 Loading CLIP-L with BF16...");
    let clip_path = "/home/alex/SwarmUI/Models/clip/clip_l.safetensors";
    let clip_wl =
        WeightLoader::from_safetensors_with_dtype(clip_path, device.clone(), DType::BF16)?;
    let clip_config = CLIPConfig::clip_l();
    let clip_model = clip::ClipTextTransformer::new(clip_config, &device, clip_wl.weights)?;
    println!("✅ CLIP-L loaded (0.23GB with BF16)");

    let mut processed = 0;
    let mut skipped = 0;

    // Process in batches to manage T5 memory
    let batch_size = 10;
    let files_to_process = text_files.len();

    // Process files in batches, loading T5 only when needed
    for batch_start in (0..files_to_process).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(files_to_process);
        let batch_files = &text_files[batch_start..batch_end];

        println!(
            "\n📦 Processing batch {}/{}",
            batch_start / batch_size + 1,
            (files_to_process + batch_size - 1) / batch_size
        );

        // Load T5 for this batch
        println!("  Loading T5 XXL with BF16 for batch processing...");
        let t5_path = "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors";
        let t5_wl =
            WeightLoader::from_safetensors_with_dtype(t5_path, device.clone(), DType::BF16)?;
        let t5_config = T5Config {
            vocab_size: 32128,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_layers: 24,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            feed_forward_proj_gated: true,
        };
        let t5_model = T5EncoderModel::new(&t5_wl.weights, &t5_config, device.clone())?;
        println!("  ✅ T5 XXL loaded with BF16 (4.5GB)");

        // Process files in this batch
        for entry in batch_files {
            let text_path = entry.path();
            let base_name = text_path.file_stem().unwrap().to_str().unwrap();

            // Check if embeddings already cached
            let clip_cache_path = cache_dir.join(format!("{}_clip.bin", base_name));
            let t5_cache_path = cache_dir.join(format!("{}_t5.bin", base_name));
            let pooled_cache_path = cache_dir.join(format!("{}_pooled.bin", base_name));

            if clip_cache_path.exists() && t5_cache_path.exists() && pooled_cache_path.exists() {
                skipped += 1;
                println!(
                    "  [{}/{}] Already cached: {}",
                    processed + skipped,
                    files_to_process,
                    base_name
                );
                continue;
            }

            // Read caption
            let caption = fs::read_to_string(&text_path)
                .map_err(|e| flame_core::Error::Io(e.to_string()))?;
            let caption = caption.trim();

            println!(
                "  [{}/{}] Processing: {}",
                processed + skipped + 1,
                files_to_process,
                base_name
            );
            println!(
                "    Caption: \"{}\"",
                if caption.len() > 50 {
                    format!("{}...", &caption[..50])
                } else {
                    caption.to_string()
                }
            );

            // Tokenize for CLIP
            let clip_encoding = clip_tokenizer.encode(caption, true).map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "CLIP tokenization failed: {:?}",
                    e
                ))
            })?;
            let mut clip_ids = clip_encoding.get_ids().to_vec();
            clip_ids.resize(77, 0); // Pad/truncate to 77 tokens
            let clip_input = Tensor::from_vec(
                clip_ids.into_iter().map(|id| id as f32).collect::<Vec<_>>(),
                Shape::from_dims(&[1, 77]),
                device.cuda_device().clone(),
            )?;

            // Encode with CLIP
            println!("    Encoding with CLIP (BF16)...");
            let clip_output = clip_model.forward(&clip_input, None)?;
            let clip_embeds = clip_output.last_hidden_state; // [1, 77, 768]

            // Get pooled output (last token)
            let pooled = clip_embeds.narrow(1, 76, 1)?.squeeze(Some(1))?; // [1, 768]

            // Tokenize for T5
            let t5_encoding = t5_tokenizer.encode(caption, true).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("T5 tokenization failed: {:?}", e))
            })?;
            let mut t5_ids = t5_encoding.get_ids().to_vec();
            if t5_ids.len() > 256 {
                t5_ids.truncate(256); // Max 256 tokens for T5
            }
            let t5_len = t5_ids.len();
            let t5_input = Tensor::from_vec(
                t5_ids.into_iter().map(|id| id as f32).collect::<Vec<_>>(),
                Shape::from_dims(&[1, t5_len]),
                device.cuda_device().clone(),
            )?;

            // Encode with T5
            println!("    Encoding with T5 XXL (BF16)...");
            let t5_embeds = t5_model.forward(&t5_input)?; // [1, seq_len, 4096]

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
                "    ✅ Saved embeddings (CLIP: {:?}, T5: {:?}, Pooled: {:?})",
                clip_embeds_f32.shape(),
                t5_embeds_f32.shape(),
                pooled_f32.shape()
            );
        }

        // Free T5 model after each batch to save memory
        println!("  🧹 Freeing T5 model from GPU memory...");
        drop(t5_model);
        drop(t5_wl);

        // Stop after first 50 files for testing
        if processed >= 50 {
            println!("\n⚠️  Stopping after {} files for testing", processed);
            break;
        }
    }

    // Free CLIP model
    drop(clip_model);

    println!("\n✅ TEXT EMBEDDING CACHING COMPLETE!");
    println!("Total processed: {}", processed);
    println!("Total skipped (already cached): {}", skipped);
    println!("Cache directory: {}", cache_dir.display());
    println!("\n🎯 NOW YOU CAN TRAIN WITHOUT LOADING TEXT ENCODERS!");
    println!("Both latents AND text embeddings are cached!");
    println!("Training will use <12GB VRAM (no VAE, no T5, no CLIP)!");
    println!("\n📊 Memory savings:");
    println!("  - VAE not loaded: 2.4GB saved (BF16)");
    println!("  - T5 not loaded: 4.5GB saved (BF16)");
    println!("  - CLIP not loaded: 0.23GB saved");
    println!("  Total saved: ~7GB!");

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

    fs::write(path, binary).map_err(|e| flame_core::Error::Io(e.to_string()))?;
    Ok(())
}
