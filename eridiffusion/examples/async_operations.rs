//! Example demonstrating consistent async operations

use eridiffusion_core::{
    async_utils::{AsyncWrapper, AsyncBatch, AsyncIO, AsyncProgress, MaybeAsync},
    Result, Error,
};
use std::path::PathBuf;
use tokio::time::{sleep, Duration};

// Example sync operation that we want to make async
struct DataProcessor {
    name: String,
}

impl DataProcessor {
    fn process_sync(&self, data: Vec<f32>) -> Result<Vec<f32>> {
        // Simulate CPU-intensive work
        let result: Vec<f32> = data.iter()
            .map(|x| x * 2.0 + 1.0)
            .collect();
        Ok(result)
    }
}

// Implement MaybeAsync for flexible sync/async execution
struct ProcessTask {
    processor: DataProcessor,
    data: Vec<f32>,
}

impl MaybeAsync for ProcessTask {
    type Output = Vec<f32>;
    
    fn execute_sync(self) -> Result<Self::Output> {
        self.processor.process_sync(self.data)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Example 1: Using AsyncWrapper for sync operations
    println!("=== AsyncWrapper Example ===");
    
    let processor = DataProcessor {
        name: "Example Processor".to_string(),
    };
    let async_processor = AsyncWrapper::new(processor);
    
    // Run sync operation in async context
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = async_processor.run(|p| p.process_sync(data.clone())).await?;
    println!("Processed data: {:?}", result);
    
    // Example 2: Batch async operations
    println!("\n=== AsyncBatch Example ===");
    
    let mut batch: AsyncBatch<Vec<f32>> = AsyncBatch::new();
    
    // Add multiple operations to batch
    for i in 0..5 {
        let data = vec![i as f32; 100];
        batch.add(data);
    }
    
    // Process all operations concurrently
    let results = batch.execute_all(|data| {
        Box::pin(async move {
            // Simulate async processing
            sleep(Duration::from_millis(100)).await;
            Ok(data.iter().map(|x| x * 2.0).collect())
        })
    }).await?;
    
    println!("Batch processed {} items", results.len());
    
    // Example 3: Limited concurrency
    println!("\n=== Limited Concurrency Example ===");
    
    let mut limited_batch: AsyncBatch<usize> = AsyncBatch::new();
    for i in 0..20 {
        limited_batch.add(i);
    }
    
    let start = std::time::Instant::now();
    let results = limited_batch.execute_limited(|i| {
        Box::pin(async move {
            println!("Processing item {}", i);
            sleep(Duration::from_millis(100)).await;
            Ok(i * 2)
        })
    }, 4).await?; // Limit to 4 concurrent operations
    
    println!("Processed {} items in {:?} with concurrency limit", 
        results.len(), start.elapsed());
    
    // Example 4: Async I/O operations
    println!("\n=== AsyncIO Example ===");
    
    let test_dir = PathBuf::from("./async_test_dir");
    let test_file = test_dir.join("test.txt");
    
    // Create directory
    AsyncIO::create_dir_all(&test_dir).await?;
    println!("Created directory: {:?}", test_dir);
    
    // Write file
    let data = b"Hello from async I/O!";
    AsyncIO::write_file(&test_file, data).await?;
    println!("Wrote file: {:?}", test_file);
    
    // Check existence
    let exists = AsyncIO::exists(&test_file).await;
    println!("File exists: {}", exists);
    
    // Read file
    let read_data = AsyncIO::read_file(&test_file).await?;
    println!("Read data: {}", String::from_utf8_lossy(&read_data));
    
    // Cleanup
    tokio::fs::remove_dir_all(&test_dir).await?;
    
    // Example 5: Async progress tracking
    println!("\n=== AsyncProgress Example ===");
    
    let items = vec!["item1", "item2", "item3", "item4", "item5"];
    let mut progress = AsyncProgress::new(items.len())
        .with_callback(|current, total, item| {
            println!("Progress: {}/{} - Processing: {}", current, total, item);
        });
    
    for item in &items {
        // Simulate processing
        sleep(Duration::from_millis(200)).await;
        progress.update(item).await;
    }
    
    println!("Processing complete: {}", progress.is_complete());
    
    // Example 6: MaybeAsync trait usage
    println!("\n=== MaybeAsync Example ===");
    
    let task = ProcessTask {
        processor: DataProcessor { name: "Flexible Processor".to_string() },
        data: vec![10.0, 20.0, 30.0],
    };
    
    // Execute synchronously
    let sync_result = ProcessTask {
        processor: DataProcessor { name: "Sync".to_string() },
        data: vec![1.0, 2.0, 3.0],
    }.execute_sync()?;
    println!("Sync result: {:?}", sync_result);
    
    // Execute asynchronously
    let async_result = task.execute_async().await?;
    println!("Async result: {:?}", async_result);
    
    // Example 7: Error handling in async context
    println!("\n=== Async Error Handling Example ===");
    
    let error_batch: AsyncBatch<usize> = AsyncBatch::new();
    
    // This will handle errors gracefully
    match error_batch.execute_all(|i| {
        Box::pin(async move {
            if i == 2 {
                Err(Error::Runtime("Simulated error".to_string()))
            } else {
                Ok(i * 2)
            }
        })
    }).await {
        Ok(results) => println!("All succeeded: {:?}", results),
        Err(e) => println!("Batch failed: {}", e),
    }
    
    Ok(())
}