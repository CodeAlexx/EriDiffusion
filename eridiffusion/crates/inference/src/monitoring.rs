//! Monitoring and profiling

use eridiffusion_core::{Result, Error};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub enable_profiling: bool,
    pub metrics_interval_seconds: u64,
    pub trace_sample_rate: f32,
    pub profile_output_path: Option<String>,
    pub export_format: ExportFormat,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Prometheus,
    OpenTelemetry,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: false,
            enable_profiling: false,
            metrics_interval_seconds: 60,
            trace_sample_rate: 0.1,
            profile_output_path: None,
            export_format: ExportFormat::Json,
        }
    }
}

/// Performance monitor
pub struct PerformanceMonitor {
    config: MonitoringConfig,
    metrics: Arc<RwLock<MetricsCollector>>,
    tracer: Option<Arc<Tracer>>,
    profiler: Option<Arc<Profiler>>,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(config: MonitoringConfig) -> Self {
        let tracer = if config.enable_tracing {
            Some(Arc::new(Tracer::new(config.trace_sample_rate)))
        } else {
            None
        };
        
        let profiler = if config.enable_profiling {
            Some(Arc::new(Profiler::new()))
        } else {
            None
        };
        
        Self {
            config,
            metrics: Arc::new(RwLock::new(MetricsCollector::new())),
            tracer,
            profiler,
        }
    }
    
    /// Record metric
    pub async fn record_metric(&self, name: &str, value: f64, tags: HashMap<String, String>) {
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.record(name, value, tags);
        }
    }
    
    /// Start span
    pub fn start_span(&self, name: &str) -> Option<SpanHandle> {
        self.tracer.as_ref().map(|t| t.start_span(name))
    }
    
    /// Start profiling
    pub fn start_profiling(&self, name: &str) -> Option<ProfileHandle> {
        self.profiler.as_ref().map(|p| p.start_profile(name))
    }
    
    /// Get metrics summary
    pub async fn get_metrics_summary(&self) -> MetricsSummary {
        let metrics = self.metrics.read().await;
        metrics.get_summary()
    }
    
    /// Export metrics
    pub async fn export_metrics(&self) -> Result<String> {
        let summary = self.get_metrics_summary().await;
        
        match self.config.export_format {
            ExportFormat::Json => {
                serde_json::to_string_pretty(&summary)
                    .map_err(|e| Error::Serialization(e.to_string()))
            }
            ExportFormat::Prometheus => {
                Ok(self.format_prometheus(&summary))
            }
            ExportFormat::OpenTelemetry => {
                Ok(self.format_opentelemetry(&summary))
            }
        }
    }
    
    /// Format metrics as Prometheus
    fn format_prometheus(&self, summary: &MetricsSummary) -> String {
        let mut output = String::new();
        
        for (name, metric) in &summary.metrics {
            let safe_name = name.replace(".", "_").replace("-", "_");
            
            output.push_str(&format!("# TYPE {} gauge\n", safe_name));
            output.push_str(&format!("{} {}\n", safe_name, metric.current));
            
            if let Some(min) = metric.min {
                output.push_str(&format!("{}_min {}\n", safe_name, min));
            }
            
            if let Some(max) = metric.max {
                output.push_str(&format!("{}_max {}\n", safe_name, max));
            }
            
            if let Some(avg) = metric.average {
                output.push_str(&format!("{}_avg {}\n", safe_name, avg));
            }
        }
        
        output
    }
    
    /// Format metrics as OpenTelemetry
    fn format_opentelemetry(&self, summary: &MetricsSummary) -> String {
        // Simplified OpenTelemetry format
        serde_json::to_string(summary).unwrap_or_default()
    }
}

/// Metrics collector
struct MetricsCollector {
    metrics: HashMap<String, MetricData>,
    window_size: usize,
}

#[derive(Debug, Clone)]
struct MetricData {
    values: VecDeque<TimestampedValue>,
    tags: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct TimestampedValue {
    value: f64,
    timestamp: SystemTime,
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            window_size: 1000,
        }
    }
    
    fn record(&mut self, name: &str, value: f64, tags: HashMap<String, String>) {
        let entry = self.metrics.entry(name.to_string()).or_insert_with(|| {
            MetricData {
                values: VecDeque::new(),
                tags: tags.clone(),
            }
        });
        
        entry.values.push_back(TimestampedValue {
            value,
            timestamp: SystemTime::now(),
        });
        
        // Trim old values
        while entry.values.len() > self.window_size {
            entry.values.pop_front();
        }
        
        // Update tags
        entry.tags.extend(tags);
    }
    
    fn get_summary(&self) -> MetricsSummary {
        let mut metrics = HashMap::new();
        
        for (name, data) in &self.metrics {
            let values: Vec<f64> = data.values.iter().map(|v| v.value).collect();
            
            if !values.is_empty() {
                let current = values.last().copied().unwrap_or(0.0);
                let min = values.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).copied();
                let max = values.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).copied();
                let average = Some(values.iter().sum::<f64>() / values.len() as f64);
                let count = values.len();
                
                metrics.insert(name.clone(), MetricSummary {
                    current,
                    min,
                    max,
                    average,
                    count,
                    tags: data.tags.clone(),
                });
            }
        }
        
        MetricsSummary {
            timestamp: SystemTime::now(),
            metrics,
        }
    }
}

/// Metrics summary
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub timestamp: SystemTime,
    pub metrics: HashMap<String, MetricSummary>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricSummary {
    pub current: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub average: Option<f64>,
    pub count: usize,
    pub tags: HashMap<String, String>,
}

/// Tracer for distributed tracing
struct Tracer {
    sample_rate: f32,
    spans: Arc<RwLock<HashMap<String, SpanData>>>,
}

#[derive(Debug)]
struct SpanData {
    name: String,
    start_time: Instant,
    end_time: Option<Instant>,
    tags: HashMap<String, String>,
    events: Vec<SpanEvent>,
}

#[derive(Debug)]
struct SpanEvent {
    name: String,
    timestamp: Instant,
    attributes: HashMap<String, String>,
}

impl Tracer {
    fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            spans: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    fn start_span(&self, name: &str) -> SpanHandle {
        let span_id = uuid::Uuid::new_v4().to_string();
        
        let should_sample = rand::random::<f32>() < self.sample_rate;
        
        if should_sample {
            let span_data = SpanData {
                name: name.to_string(),
                start_time: Instant::now(),
                end_time: None,
                tags: HashMap::new(),
                events: Vec::new(),
            };
            
            let spans = self.spans.clone();
            let span_id_clone = span_id.clone();
            tokio::spawn(async move {
                let mut spans = spans.write().await;
                spans.insert(span_id_clone, span_data);
            });
        }
        
        SpanHandle {
            span_id,
            tracer: self.spans.clone(),
            sampled: should_sample,
        }
    }
}

/// Span handle
pub struct SpanHandle {
    span_id: String,
    tracer: Arc<RwLock<HashMap<String, SpanData>>>,
    sampled: bool,
}

impl SpanHandle {
    /// Add tag to span
    pub async fn add_tag(&self, key: &str, value: &str) {
        if !self.sampled {
            return;
        }
        
        let mut spans = self.tracer.write().await;
        if let Some(span) = spans.get_mut(&self.span_id) {
            span.tags.insert(key.to_string(), value.to_string());
        }
    }
    
    /// Add event to span
    pub async fn add_event(&self, name: &str, attributes: HashMap<String, String>) {
        if !self.sampled {
            return;
        }
        
        let mut spans = self.tracer.write().await;
        if let Some(span) = spans.get_mut(&self.span_id) {
            span.events.push(SpanEvent {
                name: name.to_string(),
                timestamp: Instant::now(),
                attributes,
            });
        }
    }
    
    /// End span
    pub async fn end(self) {
        if !self.sampled {
            return;
        }
        
        let mut spans = self.tracer.write().await;
        if let Some(span) = spans.get_mut(&self.span_id) {
            span.end_time = Some(Instant::now());
        }
    }
}

/// Profiler
struct Profiler {
    profiles: Arc<RwLock<HashMap<String, ProfileData>>>,
}

#[derive(Debug)]
struct ProfileData {
    name: String,
    samples: Vec<ProfileSample>,
    start_time: Instant,
    end_time: Option<Instant>,
}

#[derive(Debug)]
struct ProfileSample {
    timestamp: Instant,
    cpu_usage: f32,
    memory_usage: usize,
    gpu_usage: Option<f32>,
    gpu_memory: Option<usize>,
}

impl Profiler {
    fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    fn start_profile(&self, name: &str) -> ProfileHandle {
        let profile_id = uuid::Uuid::new_v4().to_string();
        
        let profile_data = ProfileData {
            name: name.to_string(),
            samples: Vec::new(),
            start_time: Instant::now(),
            end_time: None,
        };
        
        let profiles = self.profiles.clone();
        let profile_id_clone = profile_id.clone();
        
        tokio::spawn(async move {
            let mut profiles = profiles.write().await;
            profiles.insert(profile_id_clone, profile_data);
        });
        
        ProfileHandle {
            profile_id,
            profiler: self.profiles.clone(),
            sampling_handle: None,
        }
    }
}

/// Profile handle
pub struct ProfileHandle {
    profile_id: String,
    profiler: Arc<RwLock<HashMap<String, ProfileData>>>,
    sampling_handle: Option<tokio::task::JoinHandle<()>>,
}

impl ProfileHandle {
    /// Start sampling
    pub fn start_sampling(&mut self, interval_ms: u64) {
        let profile_id = self.profile_id.clone();
        let profiler = self.profiler.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
            
            loop {
                interval.tick().await;
                
                let sample = ProfileSample {
                    timestamp: Instant::now(),
                    cpu_usage: get_cpu_usage(),
                    memory_usage: get_memory_usage(),
                    gpu_usage: get_gpu_usage(),
                    gpu_memory: get_gpu_memory(),
                };
                
                let mut profiles = profiler.write().await;
                if let Some(profile) = profiles.get_mut(&profile_id) {
                    profile.samples.push(sample);
                } else {
                    break;
                }
            }
        });
        
        self.sampling_handle = Some(handle);
    }
    
    /// Stop profiling
    pub async fn stop(mut self) -> Option<ProfileReport> {
        // Stop sampling
        if let Some(handle) = self.sampling_handle.take() {
            handle.abort();
        }
        
        // Mark profile as ended
        let mut profiles = self.profiler.write().await;
        if let Some(profile) = profiles.get_mut(&self.profile_id) {
            profile.end_time = Some(Instant::now());
            
            // Generate report
            Some(ProfileReport {
                name: profile.name.clone(),
                duration: profile.end_time.unwrap().duration_since(profile.start_time),
                samples: profile.samples.len(),
                avg_cpu_usage: profile.samples.iter()
                    .map(|s| s.cpu_usage)
                    .sum::<f32>() / profile.samples.len() as f32,
                peak_memory_usage: profile.samples.iter()
                    .map(|s| s.memory_usage)
                    .max()
                    .unwrap_or(0),
                avg_gpu_usage: profile.samples.iter()
                    .filter_map(|s| s.gpu_usage)
                    .sum::<f32>() / profile.samples.len() as f32,
            })
        } else {
            None
        }
    }
}

/// Profile report
#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileReport {
    pub name: String,
    pub duration: Duration,
    pub samples: usize,
    pub avg_cpu_usage: f32,
    pub peak_memory_usage: usize,
    pub avg_gpu_usage: f32,
}

/// System metric collection functions
fn get_cpu_usage() -> f32 {
    // Would use actual system metrics
    0.5
}

fn get_memory_usage() -> usize {
    // Would use actual system metrics
    1024 * 1024 * 512 // 512 MB
}

fn get_gpu_usage() -> Option<f32> {
    // Would query GPU metrics
    Some(0.7)
}

fn get_gpu_memory() -> Option<usize> {
    // Would query GPU memory
    Some(1024 * 1024 * 1024 * 4) // 4 GB
}