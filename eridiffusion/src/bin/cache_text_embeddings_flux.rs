#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::trainers::optimized_text_encoders::OptimizedTextEncoders;
use flame_core::{DType, Device, Error, Result, Shape, Tensor};
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    println!("=== FLUX TEXT EMBEDDING CACHING WITH BF16 ===");
    println!("Cache text embeddings ONCE, never load T5/CLIP during training!");
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

    // Load text encoders with BF16
    println!("\n🧠 Loading CLIP and T5 XXL encoders with BF16...");
    let clip_path = "/home/alex/SwarmUI/Models/clip/clip_l.safetensors";
    let t5_path = "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors";
    let clip_tokenizer = "/home/alex/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff/tokenizer.json";
    let t5_tokenizer = "/home/alex/SwarmUI_/dlbackend/ComfyUI/comfy/text_encoders/t5_tokenizer/tokenizer.json";

    let mut encoders = OptimizedTextEncoders::new(device.clone());
    encoders.load_clip_l(clip_path)?;
    encoders.load_tokenizers(clip_tokenizer, t5_tokenizer)?;
    encoders.ensure_t5_loaded(t5_path)?;
    println!("✅ Text encoders loaded with BF16!");

    let mut processed = 0;
    let mut skipped = 0;

    for (idx, entry) in text_files.iter().enumerate() {
        let text_path = entry.path();
        let base_name = text_path.file_stem().unwrap().to_str().unwrap();

        // Check if embeddings already cached
        let cache_path = cache_dir.join(format!("{}_text_embeds.bin", base_name));
        let pooled_cache_path = cache_dir.join(format!("{}_pooled.bin", base_name));

        if cache_path.exists() && pooled_cache_path.exists() {
            skipped += 1;
            if skipped % 100 == 0 || skipped <= 10 {
                println!("[{}/{}] Already cached: {}", idx + 1, text_files.len(), base_name);
            }
            continue;
        }

        // Read caption
        let caption = fs::read_to_string(&text_path)
            .map_err(|e| flame_core::Error::Io(e.to_string()))?;
        let caption = caption.trim();

        println!("[{}/{}] Processing: {}", idx + 1, text_files.len(), base_name);
        println!(
            "  Caption: \"{}\"",
            if caption.len() > 60 { format!("{}...", &caption[..60]) } else { caption.to_string() }
        );

        // Encode with both CLIP and T5 (concatenated)
        println!("  Encoding with CLIP+T5 (BF16)...");
        let (clip_hidden, t5_hidden) = encoders.encode_flux(caption, t5_path)?;

        // Pad CLIP hidden states to match Flux hidden size (4096) before concatenation
        let clip_f32 = clip_hidden.to_dtype(DType::F32)?;
        let t5_f32 = t5_hidden.to_dtype(DType::F32)?;
        let clip_padded = pad_to_dim(&clip_f32, 4096)?;
        let text_embeds = Tensor::cat(&[&clip_padded, &t5_f32], 1)?;

        // Derive pooled embedding by averaging CLIP token features
        let pooled_embeds = clip_f32.mean_dim(&[1], false)?;

        // text_embeds is already concatenated [CLIP, T5] in BF16
        // Shape should be [1, 77+256, dims] where dims varies
        println!("  Text embeds shape: {:?}", text_embeds.shape());
        println!("  Pooled embeds shape: {:?}", pooled_embeds.shape());

        // Convert to F32 for saving (training expects F32)
        let text_embeds_f32 = text_embeds.to_dtype(DType::F32)?;
        let pooled_embeds_f32 = pooled_embeds.to_dtype(DType::F32)?;

        // Save text embeddings (combined CLIP+T5)
        let text_data = text_embeds_f32.to_vec()?;
        let text_shape_dims = text_embeds_f32.shape().dims().to_vec();
        save_tensor_binary(&cache_path, &text_data, &text_shape_dims)?;

        // Save pooled embeddings
        let pooled_data = pooled_embeds_f32.to_vec()?;
        let pooled_shape_dims = pooled_embeds_f32.shape().dims().to_vec();
        save_tensor_binary(&pooled_cache_path, &pooled_data, &pooled_shape_dims)?;

        processed += 1;
        println!("  ✅ Saved embeddings to cache");

        // Free memory periodically
        if processed % 10 == 0 {
            println!("\n🔄 Processed {} files, continuing...", processed);
        }

        // For initial test, process only first 20 files
        if processed >= 20 {
            println!("\n⚠️  Stopping after {} files for initial test", processed);
            break;
        }
    }

    // Free encoders
    println!("\n🧹 Freeing text encoders from GPU memory...");
    drop(encoders);

    println!("\n✅ TEXT EMBEDDING CACHING COMPLETE!");
    println!("Total processed: {}", processed);
    println!("Total skipped (already cached): {}", skipped);
    println!("Cache directory: {}", cache_dir.display());
    println!("\n🎯 NOW YOU CAN TRAIN WITHOUT TEXT ENCODERS!");
    println!("Training will only need:");
    println!("  - Flux model (~12GB)");
    println!("  - LoRA weights (~200MB)");
    println!("  - Optimizer states (~1GB)");
    println!("  - Cached latents + embeddings (loaded from disk)");
    println!("Total VRAM: <15GB!");

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

fn pad_to_dim(tensor: &Tensor, target_dim: usize) -> Result<Tensor> {
    let shape = tensor.shape();
    let dims = shape.dims();
    if dims.len() != 3 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Expected 3D tensor for padding, got shape {:?}",
            dims
        )));
    }

    let current_dim = dims[2];
    if current_dim == target_dim {
        return Ok(tensor.clone());
    }

    if current_dim > target_dim {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Cannot pad from {} to smaller target {}",
            current_dim, target_dim
        )));
    }

    let pad = target_dim - current_dim;
    let zeros = Tensor::zeros(
        Shape::from_dims(&[dims[0], dims[1], pad]),
        tensor.device().cuda_device_arc(),
    )?
    .to_dtype(tensor.dtype())?;

    Tensor::cat(&[tensor, &zeros], 2)
}
