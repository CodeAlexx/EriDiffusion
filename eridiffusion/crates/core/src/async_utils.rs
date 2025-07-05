//! Async utilities for consistent API

use crate::{Result, Error};
use std::future::Future;
use std::pin::Pin;
use tokio::task;

/// Trait for operations that can be either sync or async
pub trait MaybeAsync {
    type Output;
    
    /// Execute synchronously
    fn execute_sync(self) -> Result<Self::Output>;
    
    /// Execute asynchronously
    fn execute_async(self) -> Pin<Box<dyn Future<Output = Result<Self::Output>> + Send + 'static>>
    where
        Self: Sized + Send + 'static,
        Self::Output: Send + 'static,
    {
        Box::pin(async move {
            task::spawn_blocking(move || self.execute_sync())
                .await
                .map_err(|e| Error::Runtime(format!("Task join error: {}", e)))?
        })
    }
}

/// Wrapper for making sync operations consistently async
pub struct AsyncWrapper<T> {
    inner: std::sync::Arc<T>,
}

impl<T> AsyncWrapper<T> {
    pub fn new(inner: T) -> Self {
        Self { inner: std::sync::Arc::new(inner) }
    }
    
    /// Run a sync operation in a blocking task
    pub async fn run<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&T) -> Result<R> + Send + 'static,
        T: Send + Sync + 'static,
        R: Send + 'static,
    {
        // Clone the Arc to move into the task
        let inner = self.inner.clone();
        
        task::spawn_blocking(move || {
            f(&*inner)
        })
        .await
        .map_err(|e| Error::Runtime(format!("Async task failed: {}", e)))?
    }
    
    /// Run a mutable sync operation in a blocking task
    pub async fn run_mut<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&T) -> Result<R> + Send + 'static,
        T: Send + Sync + 'static,
        R: Send + 'static,
    {
        // Note: We can't provide mutable access through Arc
        // This is a limitation - users should use interior mutability if needed
        let inner = self.inner.clone();
        
        task::spawn_blocking(move || {
            f(&*inner)
        })
        .await
        .map_err(|e| Error::Runtime(format!("Async task failed: {}", e)))?
    }
}

/// Extension trait for consistent async operations
pub trait AsyncExt: Sized {
    /// Convert to async wrapper
    fn into_async(self) -> AsyncWrapper<Self> {
        AsyncWrapper::new(self)
    }
}

impl<T> AsyncExt for T {}

/// Batch async operations for efficiency
pub struct AsyncBatch<T> {
    operations: Vec<T>,
}

impl<T> AsyncBatch<T> {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
    
    pub fn add(&mut self, op: T) {
        self.operations.push(op);
    }
    
    /// Execute all operations concurrently
    pub async fn execute_all<F, R>(self, f: F) -> Result<Vec<R>>
    where
        F: Fn(T) -> Pin<Box<dyn Future<Output = Result<R>> + Send>> + Clone,
        T: Send + 'static,
        R: Send + 'static,
    {
        let futures: Vec<_> = self.operations
            .into_iter()
            .map(|op| f.clone()(op))
            .collect();
        
        let results = futures::future::try_join_all(futures).await?;
        Ok(results)
    }
    
    /// Execute with limited concurrency
    pub async fn execute_limited<F, R>(self, f: F, limit: usize) -> Result<Vec<R>>
    where
        F: Fn(T) -> Pin<Box<dyn Future<Output = Result<R>> + Send>> + Clone,
        T: Send + 'static,
        R: Send + 'static,
    {
        use futures::stream::{self, StreamExt, TryStreamExt};
        
        let stream = stream::iter(self.operations)
            .map(move |op| f.clone()(op))
            .buffer_unordered(limit)
            .try_collect::<Vec<_>>();
        
        stream.await
    }
}

/// Helper for async I/O operations
pub struct AsyncIO;

impl AsyncIO {
    /// Read file asynchronously
    pub async fn read_file(path: &std::path::Path) -> Result<Vec<u8>> {
        tokio::fs::read(path)
            .await
            .map_err(|e| Error::Io(e))
    }
    
    /// Write file asynchronously
    pub async fn write_file(path: &std::path::Path, data: &[u8]) -> Result<()> {
        tokio::fs::write(path, data)
            .await
            .map_err(|e| Error::Io(e))
    }
    
    /// Create directory asynchronously
    pub async fn create_dir_all(path: &std::path::Path) -> Result<()> {
        tokio::fs::create_dir_all(path)
            .await
            .map_err(|e| Error::Io(e))
    }
    
    /// Check if path exists asynchronously
    pub async fn exists(path: &std::path::Path) -> bool {
        tokio::fs::metadata(path).await.is_ok()
    }
}

/// Async progress reporter
pub struct AsyncProgress<T> {
    total: usize,
    current: usize,
    callback: Option<Box<dyn Fn(usize, usize, &T) + Send + Sync>>,
}

impl<T> AsyncProgress<T> {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            current: 0,
            callback: None,
        }
    }
    
    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(usize, usize, &T) + Send + Sync + 'static,
    {
        self.callback = Some(Box::new(callback));
        self
    }
    
    pub async fn update(&mut self, item: &T) {
        self.current += 1;
        
        if let Some(callback) = &self.callback {
            callback(self.current, self.total, item);
        }
        
        // Yield occasionally to prevent blocking
        if self.current % 100 == 0 {
            tokio::task::yield_now().await;
        }
    }
    
    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_async_wrapper() {
        let data = vec![1, 2, 3, 4, 5];
        let wrapper = data.into_async();
        
        let sum = wrapper.run(|v| {
            Ok(v.iter().sum::<i32>())
        }).await.unwrap();
        
        assert_eq!(sum, 15);
    }
    
    #[tokio::test]
    async fn test_async_batch() {
        let mut batch = AsyncBatch::new();
        batch.add(1);
        batch.add(2);
        batch.add(3);
        
        let results = batch.execute_all(|x| {
            Box::pin(async move {
                Ok(x * 2)
            })
        }).await.unwrap();
        
        assert_eq!(results, vec![2, 4, 6]);
    }
}