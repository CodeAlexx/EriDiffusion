//! Example of using FLAME DataLoader

use eridiffusion::data::flame_dataloader_v2::{BucketingDataLoader, FLAMEDataLoader};
use flame::{Device, Result};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let device = Device::cuda(0);

    // Basic DataLoader
    let mut dataloader = FLAMEDataLoader::new(
        Path::new("/path/to/dataset"),
        4, // batch_size
        device,
        512,  // image_size
        true, // shuffle
    )?;

    println!("Starting training...");

    for (i, batch) in dataloader.iter().enumerate() {
        println!(
            "Batch {}: {} images, {} captions",
            i,
            batch.images.shape().dim(0),
            batch.captions.len()
        );

        // Training step would go here

        if i >= 10 {
            break;
        }
    }

    // Bucketing DataLoader for multiple aspect ratios
    let bucket_sizes = vec![(512, 512), (512, 768), (768, 512), (1024, 1024)];

    let mut bucketing_loader =
        BucketingDataLoader::new(Path::new("/path/to/dataset"), 4, device, bucket_sizes)?;

    println!("\nUsing bucketing data loader...");

    for (i, batch) in bucketing_loader.iter().enumerate() {
        println!("Batch {}: Shape {:?}", i, batch.images.shape());

        if i >= 5 {
            break;
        }
    }

    Ok(())
}
