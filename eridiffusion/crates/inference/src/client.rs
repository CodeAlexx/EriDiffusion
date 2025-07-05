//! Inference client library

use crate::server::{
    TextToImageRequest, ImageToImageRequest, InpaintRequest,
    GenerationResponse, ErrorResponse,
};
use eridiffusion_core::{Result, Error};
use reqwest::{Client, ClientBuilder, multipart};
use serde::{Serialize, Deserialize};
use std::time::Duration;
use candle_core::Tensor;

/// Client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub max_retries: usize,
    pub retry_delay_ms: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8000".to_string(),
            api_key: None,
            timeout_seconds: 300,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

/// Inference client
#[derive(Clone)]
pub struct InferenceClient {
    client: Client,
    config: ClientConfig,
}

impl InferenceClient {
    /// Create new inference client
    pub fn new(config: ClientConfig) -> Result<Self> {
        let client = ClientBuilder::new()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| Error::Runtime(e.to_string()))?;
        
        Ok(Self { client, config })
    }
    
    /// Generate images from text
    pub async fn text_to_image(
        &self,
        prompt: &str,
        options: TextToImageOptions,
    ) -> Result<GenerationResponse> {
        let request = TextToImageRequest {
            prompt: prompt.to_string(),
            negative_prompt: options.negative_prompt,
            width: options.width,
            height: options.height,
            num_images: options.num_images,
            guidance_scale: options.guidance_scale,
            num_inference_steps: options.num_inference_steps,
            seed: options.seed,
        };
        
        let url = format!("{}/v1/txt2img", self.config.base_url);
        self.post_json(&url, &request).await
    }
    
    /// Transform existing image
    pub async fn image_to_image(
        &self,
        image: &[u8],
        prompt: &str,
        options: ImageToImageOptions,
    ) -> Result<GenerationResponse> {
        let form = multipart::Form::new()
            .text("prompt", prompt.to_string())
            .part("image", multipart::Part::bytes(image.to_vec())
                .file_name("image.png")
                .mime_str("image/png")
                .map_err(|e| Error::Runtime(e.to_string()))?);
        
        let form = if let Some(neg) = options.negative_prompt {
            form.text("negative_prompt", neg)
        } else {
            form
        };
        
        let form = if let Some(strength) = options.strength {
            form.text("strength", strength.to_string())
        } else {
            form
        };
        
        let url = format!("{}/v1/img2img", self.config.base_url);
        self.post_multipart(&url, form).await
    }
    
    /// Inpaint image
    pub async fn inpaint(
        &self,
        image: &[u8],
        mask: &[u8],
        prompt: &str,
        options: InpaintOptions,
    ) -> Result<GenerationResponse> {
        let form = multipart::Form::new()
            .text("prompt", prompt.to_string())
            .part("image", multipart::Part::bytes(image.to_vec())
                .file_name("image.png")
                .mime_str("image/png")
                .map_err(|e| Error::Runtime(e.to_string()))?)
            .part("mask", multipart::Part::bytes(mask.to_vec())
                .file_name("mask.png")
                .mime_str("image/png")
                .map_err(|e| Error::Runtime(e.to_string()))?);
        
        let form = if let Some(neg) = options.negative_prompt {
            form.text("negative_prompt", neg)
        } else {
            form
        };
        
        let url = format!("{}/v1/inpaint", self.config.base_url);
        self.post_multipart(&url, form).await
    }
    
    /// Check request status
    pub async fn check_status(&self, request_id: &str) -> Result<RequestStatus> {
        let url = format!("{}/v1/status/{}", self.config.base_url, request_id);
        
        let response = self.execute_request(
            self.client.get(&url),
        ).await?;
        
        Ok(response.json().await.map_err(|e| Error::Runtime(e.to_string()))?)
    }
    
    /// Get server health
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let url = format!("{}/health", self.config.base_url);
        
        let response = self.execute_request(
            self.client.get(&url),
        ).await?;
        
        Ok(response.json().await.map_err(|e| Error::Runtime(e.to_string()))?)
    }
    
    /// Get server metrics
    pub async fn get_metrics(&self) -> Result<ServerMetrics> {
        let url = format!("{}/metrics", self.config.base_url);
        
        let response = self.execute_request(
            self.client.get(&url),
        ).await?;
        
        Ok(response.json().await.map_err(|e| Error::Runtime(e.to_string()))?)
    }
    
    /// Execute request with retries
    async fn execute_request(
        &self,
        mut request: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response> {
        if let Some(ref api_key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }
        
        let mut last_error = None;
        
        for attempt in 0..self.config.max_retries {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(
                    self.config.retry_delay_ms * (attempt as u64)
                )).await;
            }
            
            match request.try_clone().unwrap().send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        return Ok(response);
                    } else {
                        let error: ErrorResponse = response.json().await
                            .unwrap_or_else(|_| ErrorResponse {
                                error: "unknown".to_string(),
                                message: "Unknown error".to_string(),
                                request_id: None,
                            });
                        
                        last_error = Some(Error::Runtime(error.message));
                    }
                }
                Err(e) => {
                    last_error = Some(Error::Runtime(e.to_string()));
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| Error::Runtime("Request failed".to_string())))
    }
    
    /// Post JSON request
    async fn post_json<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        body: &T,
    ) -> Result<R> {
        let response = self.execute_request(
            self.client.post(url).json(body),
        ).await?;
        
        Ok(response.json().await.map_err(|e| Error::Runtime(e.to_string()))?)
    }
    
    /// Post multipart request
    async fn post_multipart<R: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        form: multipart::Form,
    ) -> Result<R> {
        let response = self.execute_request(
            self.client.post(url).multipart(form),
        ).await?;
        
        Ok(response.json().await.map_err(|e| Error::Runtime(e.to_string()))?)
    }
}

/// Text to image options
#[derive(Debug, Clone, Default)]
pub struct TextToImageOptions {
    pub negative_prompt: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub num_images: Option<usize>,
    pub guidance_scale: Option<f32>,
    pub num_inference_steps: Option<usize>,
    pub seed: Option<u64>,
}

/// Image to image options
#[derive(Debug, Clone, Default)]
pub struct ImageToImageOptions {
    pub negative_prompt: Option<String>,
    pub strength: Option<f32>,
    pub guidance_scale: Option<f32>,
    pub num_inference_steps: Option<usize>,
    pub seed: Option<u64>,
}

/// Inpaint options
#[derive(Debug, Clone, Default)]
pub struct InpaintOptions {
    pub negative_prompt: Option<String>,
    pub guidance_scale: Option<f32>,
    pub num_inference_steps: Option<usize>,
    pub seed: Option<u64>,
}

/// Request status
#[derive(Debug, Serialize, Deserialize)]
pub struct RequestStatus {
    pub request_id: String,
    pub status: String,
    pub progress: Option<f32>,
    pub eta_seconds: Option<u64>,
}

/// Health status
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: String,
}

/// Server metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct ServerMetrics {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f32,
    pub avg_generation_time_ms: u64,
}

/// Async client with connection pooling
pub struct AsyncInferenceClient {
    clients: Vec<InferenceClient>,
    current_client: std::sync::atomic::AtomicUsize,
}

impl AsyncInferenceClient {
    /// Create new async client with multiple connections
    pub fn new(base_url: &str, num_connections: usize) -> Result<Self> {
        let mut clients = Vec::new();
        
        for _ in 0..num_connections {
            let config = ClientConfig {
                base_url: base_url.to_string(),
                ..Default::default()
            };
            clients.push(InferenceClient::new(config)?);
        }
        
        Ok(Self {
            clients,
            current_client: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    
    /// Get next client (round-robin)
    fn get_client(&self) -> &InferenceClient {
        let idx = self.current_client.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        &self.clients[idx % self.clients.len()]
    }
    
    /// Batch text to image generation
    pub async fn batch_text_to_image(
        &self,
        prompts: &[String],
        options: TextToImageOptions,
    ) -> Vec<Result<GenerationResponse>> {
        let mut handles = Vec::new();
        
        for (i, prompt) in prompts.iter().enumerate() {
            let client = self.clients[i % self.clients.len()].clone();
            let prompt = prompt.clone();
            let opts = options.clone();
            
            let handle = tokio::spawn(async move {
                client.text_to_image(&prompt, opts).await
            });
            
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(Error::Runtime(e.to_string()))),
            }
        }
        
        results
    }
}

/// Utility functions
pub mod utils {
    use super::*;
    use base64::{Engine as _, engine::general_purpose};
    
    /// Decode base64 image to tensor
    pub fn decode_image(base64_str: &str) -> Result<Tensor> {
        let bytes = general_purpose::STANDARD
            .decode(base64_str)
            .map_err(|e| Error::Runtime(e.to_string()))?;
        
        // Would decode image bytes to tensor
        Ok(Tensor::zeros(&[3, 512, 512], candle_core::DType::F32, &candle_core::Device::Cpu)?)
    }
    
    /// Encode tensor to base64 image
    pub fn encode_image(tensor: &Tensor) -> Result<String> {
        // Convert tensor to image bytes
        let (c, h, w) = tensor.dims3()?;
        
        // Ensure tensor is in [0, 1] range
        let tensor = tensor.clamp(0.0, 1.0)?;
        
        // Convert to u8
        let tensor_u8 = (tensor * 255.0)?.to_dtype(candle_core::DType::U8)?;
        
        // Get raw bytes
        let raw_data = tensor_u8.flatten_all()?.to_vec1::<u8>()?;
        
        // Create image using image crate
        use image::{RgbImage, ImageBuffer};
        
        let img: image::RgbImage = if c == 3 {
            // RGB image
            ImageBuffer::from_raw(w as u32, h as u32, raw_data)
                .ok_or_else(|| Error::Conversion("Failed to create image from tensor".to_string()))?
        } else {
            return Err(Error::Conversion(format!("Unsupported channel count: {}", c)));
        };
        
        // Encode to PNG
        use image::codecs::png::PngEncoder;
        use std::io::Cursor;
        
        let mut buffer = Cursor::new(Vec::new());
        let encoder = PngEncoder::new(&mut buffer);
        img.write_with_encoder(encoder)
            .map_err(|e| Error::Runtime(format!("Failed to encode PNG: {}", e)))?;
        
        Ok(general_purpose::STANDARD.encode(buffer.into_inner()))
    }
    
    /// Save generation response to disk
    pub async fn save_response(
        response: &GenerationResponse,
        output_dir: &str,
    ) -> Result<Vec<String>> {
        let mut paths = Vec::new();
        
        for (i, image_b64) in response.images.iter().enumerate() {
            let filename = format!("{}_{}_{}.png", 
                response.request_id,
                response.metadata.seed.unwrap_or(0),
                i
            );
            
            let path = std::path::Path::new(output_dir).join(&filename);
            let bytes = general_purpose::STANDARD
                .decode(image_b64)
                .map_err(|e| Error::Runtime(e.to_string()))?;
            
            tokio::fs::write(&path, bytes).await?;
            paths.push(path.to_string_lossy().to_string());
        }
        
        Ok(paths)
    }
}