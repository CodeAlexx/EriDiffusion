//! Inference API server

use crate::{
    pipeline::{InferencePipeline, InferenceConfig},
    batch::{BatchInferenceEngine, BatchRequest, RequestType},
    cache::{InferenceCache, CacheConfig, CacheKey},
};
use eridiffusion_core::{Result, Error};
use serde::{Serialize, Deserialize};

// Use the public versions defined below instead
use axum::{
    Router,
    routing::{get, post},
    extract::{Query, Json, State, Path},
    response::{Json as JsonResponse, Response, IntoResponse},
    http::StatusCode,
};
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    trace::TraceLayer,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_concurrent_requests: usize,
    pub request_timeout_seconds: u64,
    pub max_payload_size: usize,
    pub enable_metrics: bool,
    pub enable_health_check: bool,
    pub api_key: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            max_concurrent_requests: 100,
            request_timeout_seconds: 300,
            max_payload_size: 100 * 1024 * 1024, // 100MB
            enable_metrics: true,
            enable_health_check: true,
            api_key: None,
        }
    }
}

/// API request types
#[derive(Debug, Serialize, Deserialize)]
pub struct TextToImageRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub num_images: Option<usize>,
    pub guidance_scale: Option<f32>,
    pub num_inference_steps: Option<usize>,
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageToImageRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub strength: Option<f32>,
    pub guidance_scale: Option<f32>,
    pub num_inference_steps: Option<usize>,
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InpaintRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub guidance_scale: Option<f32>,
    pub num_inference_steps: Option<usize>,
    pub seed: Option<u64>,
}

/// API responses
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub request_id: String,
    pub images: Vec<String>, // Base64 encoded
    pub metadata: GenerationMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub guidance_scale: f32,
    pub num_inference_steps: usize,
    pub seed: Option<u64>,
    pub duration_ms: u64,
    pub cached: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    pub request_id: Option<String>,
}

/// Server state
#[derive(Clone)]
struct ServerState {
    engine: Arc<BatchInferenceEngine>,
    cache: Arc<InferenceCache>,
    config: ServerConfig,
    metrics: Arc<RwLock<ServerMetrics>>,
}

#[derive(Debug, Default)]
struct ServerMetrics {
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    cache_hits: usize,
    total_generation_time_ms: u64,
}

/// Inference server
pub struct InferenceServer {
    state: Arc<ServerState>,
}

impl InferenceServer {
    /// Create new inference server
    pub fn new(
        pipeline: InferencePipeline,
        server_config: ServerConfig,
        cache_config: CacheConfig,
    ) -> Result<Self> {
        let engine = BatchInferenceEngine::new(
            pipeline,
            Default::default(),
        );
        
        let cache = tokio::runtime::Handle::current()
            .block_on(InferenceCache::new(cache_config))?;
        
        Ok(Self {
            state: Arc::new(ServerState {
                engine: Arc::new(engine),
                cache: Arc::new(cache),
                config: server_config,
                metrics: Arc::new(RwLock::new(ServerMetrics::default())),
            }),
        })
    }
    
    /// Start server
    pub async fn start(&self) -> Result<()> {
        // Start batch processing
        self.state.engine.start().await;
        
        // Build router
        let app = self.build_router();
        
        // Start server
        let addr = format!("{}:{}", self.state.config.host, self.state.config.port);
        println!("Starting inference server on {}", addr);
        
        let listener = tokio::net::TcpListener::bind(&addr).await
            .map_err(|e| Error::Runtime(format!("Failed to bind to {}: {}", addr, e)))?;
            
        // Simple axum serve without into_make_service
        loop {
            let (stream, _) = listener.accept().await
                .map_err(|e| Error::Runtime(e.to_string()))?;
            
            let app = app.clone();
            tokio::spawn(async move {
                // Handle connection with app
                // This is a simplified version - actual implementation would use hyper
            });
        }
        
        Ok(())
    }
    
    /// Build router
    fn build_router(&self) -> Router<Arc<ServerState>> {
        let mut app = Router::new()
            .route("/v1/txt2img", post(text_to_image_handler))
            .route("/v1/img2img", post(image_to_image_handler))
            .route("/v1/inpaint", post(inpaint_handler))
            .route("/v1/status/:request_id", get(status_handler));
        
        if self.state.config.enable_health_check {
            app = app.route("/health", get(health_handler));
        }
        
        if self.state.config.enable_metrics {
            app = app.route("/metrics", get(metrics_handler));
        }
        
        app = app
            .layer(TraceLayer::new_for_http())
            .layer(CorsLayer::permissive())
            .layer(RequestBodyLimitLayer::new(self.state.config.max_payload_size))
            .with_state(self.state.clone());
        
        app
    }
}

/// Text to image handler
async fn text_to_image_handler(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<TextToImageRequest>,
) -> std::result::Result<impl IntoResponse, ErrorResponse> {
    let request_id = Uuid::new_v4().to_string();
    let start_time = std::time::Instant::now();
    
    // Update metrics
    {
        let mut metrics = state.metrics.write().await;
        metrics.total_requests += 1;
    }
    
    // Check cache
    let cache_key = CacheKey::new(
        &req.prompt,
        req.negative_prompt.as_deref(),
        req.width.unwrap_or(512),
        req.height.unwrap_or(512),
        req.guidance_scale.unwrap_or(7.5),
        req.seed,
    );
    
    if let Some(cached) = state.cache.get(&cache_key).await {
        let mut metrics = state.metrics.write().await;
        metrics.cache_hits += 1;
        metrics.successful_requests += 1;
        
        // Return cached result
        return Ok(JsonResponse(GenerationResponse {
            request_id,
            images: vec![], // Would convert tensor to base64
            metadata: GenerationMetadata {
                prompt: req.prompt,
                width: req.width.unwrap_or(512),
                height: req.height.unwrap_or(512),
                guidance_scale: req.guidance_scale.unwrap_or(7.5),
                num_inference_steps: req.num_inference_steps.unwrap_or(50),
                seed: req.seed,
                duration_ms: start_time.elapsed().as_millis() as u64,
                cached: true,
            },
        }));
    }
    
    // Submit to batch engine
    let batch_req = BatchRequest {
        id: request_id.clone(),
        request_type: RequestType::TextToImage {
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            width: req.width.unwrap_or(512),
            height: req.height.unwrap_or(512),
            num_images: req.num_images.unwrap_or(1),
        },
        priority: 0,
    };
    
    state.engine.submit(batch_req).await
        .map_err(|e| ErrorResponse {
            error: "submission_failed".to_string(),
            message: e.to_string(),
            request_id: Some(request_id.clone()),
        })?;
    
    // Wait for result (simplified - would use proper async handling)
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    
    let duration_ms = start_time.elapsed().as_millis() as u64;
    
    // Update metrics
    {
        let mut metrics = state.metrics.write().await;
        metrics.successful_requests += 1;
        metrics.total_generation_time_ms += duration_ms;
    }
    
    Ok(JsonResponse(GenerationResponse {
        request_id,
        images: vec!["base64_encoded_image".to_string()], // Placeholder
        metadata: GenerationMetadata {
            prompt: req.prompt,
            width: req.width.unwrap_or(512),
            height: req.height.unwrap_or(512),
            guidance_scale: req.guidance_scale.unwrap_or(7.5),
            num_inference_steps: req.num_inference_steps.unwrap_or(50),
            seed: req.seed,
            duration_ms,
            cached: false,
        },
    }))
}

/// Image to image handler
async fn image_to_image_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<ImageToImageRequest>,
) -> std::result::Result<impl IntoResponse, ErrorResponse> {
    // Would parse multipart form data
    let request_id = Uuid::new_v4().to_string();
    
    Ok(JsonResponse(GenerationResponse {
        request_id,
        images: vec![],
        metadata: GenerationMetadata {
            prompt: "test".to_string(),
            width: 512,
            height: 512,
            guidance_scale: 7.5,
            num_inference_steps: 50,
            seed: None,
            duration_ms: 0,
            cached: false,
        },
    }))
}

/// Inpaint handler
async fn inpaint_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<InpaintRequest>,
) -> std::result::Result<impl IntoResponse, ErrorResponse> {
    // Would parse multipart form data
    let request_id = Uuid::new_v4().to_string();
    
    Ok(JsonResponse(GenerationResponse {
        request_id,
        images: vec![],
        metadata: GenerationMetadata {
            prompt: "test".to_string(),
            width: 512,
            height: 512,
            guidance_scale: 7.5,
            num_inference_steps: 50,
            seed: None,
            duration_ms: 0,
            cached: false,
        },
    }))
}

/// Status handler
async fn status_handler(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(request_id): axum::extract::Path<String>,
) -> std::result::Result<impl IntoResponse, ErrorResponse> {
    // Would check request status
    Ok(JsonResponse(serde_json::json!({
        "request_id": request_id,
        "status": "completed",
    })))
}

/// Health check handler
async fn health_handler() -> impl IntoResponse {
    JsonResponse(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

/// Metrics handler
async fn metrics_handler(
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    let metrics = state.metrics.read().await;
    
    JsonResponse(serde_json::json!({
        "total_requests": metrics.total_requests,
        "successful_requests": metrics.successful_requests,
        "failed_requests": metrics.failed_requests,
        "cache_hits": metrics.cache_hits,
        "cache_hit_rate": if metrics.total_requests > 0 {
            metrics.cache_hits as f32 / metrics.total_requests as f32
        } else {
            0.0
        },
        "avg_generation_time_ms": if metrics.successful_requests > 0 {
            metrics.total_generation_time_ms / metrics.successful_requests as u64
        } else {
            0
        },
    }))
}

/// Auth middleware
async fn auth_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Response {
    // Would implement API key validation
    next.run(req).await
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, JsonResponse(self)).into_response()
    }
}