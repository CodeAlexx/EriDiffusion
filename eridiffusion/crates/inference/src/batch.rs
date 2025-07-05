//! Batch inference support

use crate::pipeline::{InferencePipeline, InferenceConfig, InferenceOutput};
use eridiffusion_core::{Result, Error, Device};
use candle_core::Tensor;
use serde::{Serialize, Deserialize};
use tokio::sync::{mpsc, RwLock, Semaphore};
use std::sync::Arc;
use std::collections::VecDeque;

/// Batch request
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub id: String,
    pub request_type: RequestType,
    pub priority: i32,
}

#[derive(Debug, Clone)]
pub enum RequestType {
    TextToImage {
        prompt: String,
        negative_prompt: Option<String>,
        width: usize,
        height: usize,
        num_images: usize,
    },
    ImageToImage {
        image: Tensor,
        prompt: String,
        negative_prompt: Option<String>,
        strength: f32,
        num_images: usize,
    },
    Inpaint {
        image: Tensor,
        mask: Tensor,
        prompt: String,
        negative_prompt: Option<String>,
        num_images: usize,
    },
}

/// Batch response
#[derive(Debug)]
pub struct BatchResponse {
    pub id: String,
    pub result: Result<InferenceOutput>,
    pub duration_ms: u64,
}

/// Batch inference engine
pub struct BatchInferenceEngine {
    pipeline: Arc<InferencePipeline>,
    config: BatchConfig,
    request_queue: Arc<RwLock<VecDeque<BatchRequest>>>,
    response_tx: mpsc::Sender<BatchResponse>,
    response_rx: Arc<RwLock<mpsc::Receiver<BatchResponse>>>,
    semaphore: Arc<Semaphore>,
    stats: Arc<RwLock<BatchStats>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_queue_size: usize,
    pub batch_timeout_ms: u64,
    pub max_concurrent_batches: usize,
    pub dynamic_batching: bool,
    pub priority_queue: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_queue_size: 100,
            batch_timeout_ms: 100,
            max_concurrent_batches: 2,
            dynamic_batching: true,
            priority_queue: true,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct BatchStats {
    total_requests: usize,
    completed_requests: usize,
    failed_requests: usize,
    total_latency_ms: u64,
    queue_wait_ms: u64,
    inference_time_ms: u64,
}

impl BatchInferenceEngine {
    /// Create new batch inference engine
    pub fn new(
        pipeline: InferencePipeline,
        config: BatchConfig,
    ) -> Self {
        let (tx, rx) = mpsc::channel(config.max_queue_size);
        let max_concurrent_batches = config.max_concurrent_batches;
        
        Self {
            pipeline: Arc::new(pipeline),
            config,
            request_queue: Arc::new(RwLock::new(VecDeque::new())),
            response_tx: tx,
            response_rx: Arc::new(RwLock::new(rx)),
            semaphore: Arc::new(Semaphore::new(max_concurrent_batches)),
            stats: Arc::new(RwLock::new(BatchStats::default())),
        }
    }
    
    /// Submit request
    pub async fn submit(&self, request: BatchRequest) -> Result<()> {
        let mut queue = self.request_queue.write().await;
        
        if queue.len() >= self.config.max_queue_size {
            return Err(Error::Runtime("Queue is full".to_string()));
        }
        
        queue.push_back(request);
        Ok(())
    }
    
    /// Get response channel
    pub fn get_response_receiver(&self) -> Arc<RwLock<mpsc::Receiver<BatchResponse>>> {
        self.response_rx.clone()
    }
    
    /// Start processing loop
    pub async fn start(&self) {
        let engine = self.clone();
        
        tokio::spawn(async move {
            loop {
                engine.process_batch().await;
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });
    }
    
    /// Process a batch
    async fn process_batch(&self) {
        // Acquire permit
        let permit = match self.semaphore.acquire().await {
            Ok(p) => p,
            Err(_) => return,
        };
        
        // Collect batch
        let batch = self.collect_batch().await;
        if batch.is_empty() {
            return;
        }
        
        // Process batch
        let start_time = std::time::Instant::now();
        
        for request in batch {
            let pipeline = self.pipeline.clone();
            let tx = self.response_tx.clone();
            let stats = self.stats.clone();
            
            tokio::spawn(async move {
                let request_start = std::time::Instant::now();
                
                let result = match &request.request_type {
                    RequestType::TextToImage { 
                        prompt, 
                        negative_prompt, 
                        width, 
                        height, 
                        num_images 
                    } => {
                        pipeline.text_to_image(
                            prompt,
                            negative_prompt.as_deref(),
                            *width,
                            *height,
                            *num_images,
                        ).await
                    }
                    RequestType::ImageToImage {
                        image,
                        prompt,
                        negative_prompt,
                        strength,
                        num_images,
                    } => {
                        pipeline.image_to_image(
                            image,
                            prompt,
                            negative_prompt.as_deref(),
                            *strength,
                            *num_images,
                        ).await
                    }
                    RequestType::Inpaint {
                        image,
                        mask,
                        prompt,
                        negative_prompt,
                        num_images,
                    } => {
                        pipeline.inpaint(
                            image,
                            mask,
                            prompt,
                            negative_prompt.as_deref(),
                            *num_images,
                        ).await
                    }
                };
                
                let duration_ms = request_start.elapsed().as_millis() as u64;
                
                // Update stats
                {
                    let mut stats = stats.write().await;
                    stats.completed_requests += 1;
                    stats.total_latency_ms += duration_ms;
                    stats.inference_time_ms += duration_ms;
                    
                    if result.is_err() {
                        stats.failed_requests += 1;
                    }
                }
                
                let response = BatchResponse {
                    id: request.id,
                    result,
                    duration_ms,
                };
                
                let _ = tx.send(response).await;
            });
        }
        
        drop(permit);
    }
    
    /// Collect requests for batch
    async fn collect_batch(&self) -> Vec<BatchRequest> {
        let mut queue = self.request_queue.write().await;
        let mut batch = Vec::new();
        
        // Wait for at least one request or timeout
        let deadline = tokio::time::Instant::now() + 
            tokio::time::Duration::from_millis(self.config.batch_timeout_ms);
        
        while batch.len() < self.config.max_batch_size {
            if let Some(request) = queue.pop_front() {
                batch.push(request);
                
                // Check if we should continue batching
                if !self.config.dynamic_batching && !queue.is_empty() {
                    continue;
                }
            }
            
            if queue.is_empty() || tokio::time::Instant::now() >= deadline {
                break;
            }
        }
        
        // Sort by priority if enabled
        if self.config.priority_queue {
            batch.sort_by_key(|r| -r.priority);
        }
        
        batch
    }
    
    /// Get statistics
    pub async fn get_stats(&self) -> BatchStats {
        self.stats.read().await.clone()
    }
}

impl Clone for BatchInferenceEngine {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
            config: self.config.clone(),
            request_queue: self.request_queue.clone(),
            response_tx: self.response_tx.clone(),
            response_rx: self.response_rx.clone(),
            semaphore: self.semaphore.clone(),
            stats: self.stats.clone(),
        }
    }
}

/// Dynamic batching optimizer
pub struct DynamicBatchOptimizer {
    history: VecDeque<BatchMetrics>,
    config: OptimizerConfig,
}

#[derive(Debug, Clone)]
struct BatchMetrics {
    batch_size: usize,
    latency_ms: u64,
    throughput: f32,
    gpu_utilization: f32,
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub history_size: usize,
    pub target_latency_ms: u64,
    pub min_batch_size: usize,
    pub max_batch_size: usize,
}

impl DynamicBatchOptimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            history: VecDeque::with_capacity(config.history_size),
            config,
        }
    }
    
    /// Record batch metrics
    pub fn record_batch(&mut self, metrics: BatchMetrics) {
        if self.history.len() >= self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
    }
    
    /// Get optimal batch size
    pub fn get_optimal_batch_size(&self) -> usize {
        if self.history.len() < 5 {
            return self.config.min_batch_size;
        }
        
        // Analyze recent performance
        let avg_latency: f64 = self.history.iter()
            .map(|m| m.latency_ms as f64)
            .sum::<f64>() / self.history.len() as f64;
        
        let avg_throughput: f64 = self.history.iter()
            .map(|m| m.throughput as f64)
            .sum::<f64>() / self.history.len() as f64;
        
        // Find batch size with best throughput under latency constraint
        let mut best_size = self.config.min_batch_size;
        let mut best_score = 0.0;
        
        for size in self.config.min_batch_size..=self.config.max_batch_size {
            let estimated_latency = avg_latency * (size as f64 / self.config.min_batch_size as f64).sqrt();
            let estimated_throughput = avg_throughput * size as f64;
            
            if estimated_latency <= self.config.target_latency_ms as f64 {
                let score = estimated_throughput / estimated_latency;
                if score > best_score {
                    best_score = score;
                    best_size = size;
                }
            }
        }
        
        best_size
    }
}

/// Request prioritizer
pub struct RequestPrioritizer {
    strategy: PriorityStrategy,
}

pub enum PriorityStrategy {
    FIFO,
    LIFO,
    ShortestFirst,
    Custom(Box<dyn Fn(&BatchRequest, &BatchRequest) -> std::cmp::Ordering + Send + Sync>),
}

impl Clone for PriorityStrategy {
    fn clone(&self) -> Self {
        match self {
            PriorityStrategy::FIFO => PriorityStrategy::FIFO,
            PriorityStrategy::LIFO => PriorityStrategy::LIFO,
            PriorityStrategy::ShortestFirst => PriorityStrategy::ShortestFirst,
            PriorityStrategy::Custom(_) => PriorityStrategy::FIFO, // Can't clone closures, default to FIFO
        }
    }
}

impl std::fmt::Debug for PriorityStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PriorityStrategy::FIFO => write!(f, "FIFO"),
            PriorityStrategy::LIFO => write!(f, "LIFO"),
            PriorityStrategy::ShortestFirst => write!(f, "ShortestFirst"),
            PriorityStrategy::Custom(_) => write!(f, "Custom"),
        }
    }
}

impl RequestPrioritizer {
    pub fn new(strategy: PriorityStrategy) -> Self {
        Self { strategy }
    }
    
    /// Sort requests by priority
    pub fn sort_requests(&self, requests: &mut Vec<BatchRequest>) {
        match &self.strategy {
            PriorityStrategy::FIFO => {
                // Already in FIFO order
            }
            PriorityStrategy::LIFO => {
                requests.reverse();
            }
            PriorityStrategy::ShortestFirst => {
                requests.sort_by_key(|r| match &r.request_type {
                    RequestType::TextToImage { width, height, .. } => width * height,
                    RequestType::ImageToImage { .. } => 1024 * 1024, // Estimate
                    RequestType::Inpaint { .. } => 1024 * 1024, // Estimate
                });
            }
            PriorityStrategy::Custom(cmp) => {
                requests.sort_by(|a, b| cmp(a, b));
            }
        }
    }
}