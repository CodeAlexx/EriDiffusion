//! Training metrics tracking

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use eridiffusion_core::{Result, Error, TensorExt};

/// Metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Metric {
    Loss,
    LearningRate,
    Epoch,
    GlobalStep,
    ValidationLoss,
    GradientNorm,
    ParameterNorm,
    Custom(u64), // Hash of custom metric name
}

impl Metric {
    /// Create custom metric from name
    pub fn custom(name: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        Metric::Custom(hasher.finish())
    }
}

/// Metrics history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEntry {
    pub step: usize,
    pub value: f32,
    pub timestamp: u64,
}

/// Metrics tracker
#[derive(Debug, Clone)]
pub struct MetricsTracker {
    metrics: HashMap<String, Vec<MetricEntry>>,
    custom_names: HashMap<u64, String>,
    max_history: usize,
}

impl MetricsTracker {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self::with_max_history(10000)
    }
    
    /// Create with custom history limit
    pub fn with_max_history(max_history: usize) -> Self {
        Self {
            metrics: HashMap::new(),
            custom_names: HashMap::new(),
            max_history,
        }
    }
    
    /// Log a metric value
    pub fn log(&mut self, metric_name: &str, value: f32) {
        self.log_at_step(metric_name, value, self.current_step())
    }
    
    /// Log a metric value at specific step
    pub fn log_at_step(&mut self, metric_name: &str, value: f32, step: usize) {
        let entry = MetricEntry {
            step,
            value,
            timestamp: Self::current_timestamp(),
        };
        
        let history = self.metrics.entry(metric_name.to_string()).or_insert_with(Vec::new);
        history.push(entry);
        
        // Trim history if needed
        if history.len() > self.max_history {
            history.drain(0..history.len() - self.max_history);
        }
    }
    
    /// Get metric history
    pub fn get_history(&self, metric_name: &str) -> Option<&Vec<MetricEntry>> {
        self.metrics.get(metric_name)
    }
    
    /// Get latest value for metric
    pub fn get_latest(&self, metric_name: &str) -> Option<f32> {
        self.metrics.get(metric_name)
            .and_then(|history| history.last())
            .map(|entry| entry.value)
    }
    
    /// Get average over last n steps
    pub fn get_average(&self, metric_name: &str, n: usize) -> Option<f32> {
        self.metrics.get(metric_name).map(|history| {
            let start = history.len().saturating_sub(n);
            let recent = &history[start..];
            
            if recent.is_empty() {
                0.0
            } else {
                recent.iter().map(|e| e.value).sum::<f32>() / recent.len() as f32
            }
        })
    }
    
    /// Get all metric names
    pub fn metric_names(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }
    
    /// Clear all metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
    }
    
    /// Export metrics to JSON
    pub fn export_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.metrics)
            .map_err(|e| Error::Serialization(e.to_string()))
    }
    
    /// Import metrics from JSON
    pub fn import_json(&mut self, json: &str) -> Result<()> {
        let metrics: HashMap<String, Vec<MetricEntry>> = serde_json::from_str(json)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        
        self.metrics = metrics;
        Ok(())
    }
    
    /// Create tensorboard summary
    pub fn create_summary(&self, step: usize) -> TensorboardSummary {
        let mut summary = TensorboardSummary::new(step);
        
        for (name, history) in &self.metrics {
            if let Some(latest) = history.last() {
                if latest.step == step {
                    summary.add_scalar(name, latest.value);
                }
            }
        }
        
        summary
    }
    
    /// Get current step (max step across all metrics)
    fn current_step(&self) -> usize {
        self.metrics.values()
            .flat_map(|history| history.last())
            .map(|entry| entry.step)
            .max()
            .unwrap_or(0)
    }
    
    /// Get current timestamp
    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// Tensorboard summary
#[derive(Debug, Clone)]
pub struct TensorboardSummary {
    step: usize,
    scalars: HashMap<String, f32>,
    histograms: HashMap<String, Vec<f32>>,
    images: HashMap<String, Vec<u8>>,
}

impl TensorboardSummary {
    /// Create new summary
    pub fn new(step: usize) -> Self {
        Self {
            step,
            scalars: HashMap::new(),
            histograms: HashMap::new(),
            images: HashMap::new(),
        }
    }
    
    /// Add scalar value
    pub fn add_scalar(&mut self, name: &str, value: f32) {
        self.scalars.insert(name.to_string(), value);
    }
    
    /// Add histogram
    pub fn add_histogram(&mut self, name: &str, values: Vec<f32>) {
        self.histograms.insert(name.to_string(), values);
    }
    
    /// Add image
    pub fn add_image(&mut self, name: &str, image_data: Vec<u8>) {
        self.images.insert(name.to_string(), image_data);
    }
}

/// Metrics aggregator for distributed training
#[derive(Debug, Clone)]
pub struct MetricsAggregator {
    local_metrics: MetricsTracker,
    world_size: usize,
}

impl MetricsAggregator {
    /// Create new aggregator
    pub fn new(world_size: usize) -> Self {
        Self {
            local_metrics: MetricsTracker::new(),
            world_size,
        }
    }
    
    /// Log local metric
    pub fn log_local(&mut self, metric_name: &str, value: f32) {
        self.local_metrics.log(metric_name, value);
    }
    
    /// Aggregate metrics across all ranks
    pub async fn aggregate(&self) -> Result<HashMap<String, f32>> {
        // In distributed training, would use all-reduce
        // For now, just return local metrics
        let mut aggregated = HashMap::new();
        
        for name in self.local_metrics.metric_names() {
            if let Some(value) = self.local_metrics.get_latest(&name) {
                aggregated.insert(name, value);
            }
        }
        
        Ok(aggregated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();
        
        // Log some metrics
        tracker.log("loss", 0.5);
        tracker.log("loss", 0.4);
        tracker.log("loss", 0.3);
        
        // Check latest
        assert_eq!(tracker.get_latest("loss"), Some(0.3));
        
        // Check average
        assert_eq!(tracker.get_average("loss", 3), Some(0.4));
        
        // Check history
        let history = tracker.get_history("loss").unwrap();
        assert_eq!(history.len(), 3);
    }
    
    #[test]
    fn test_metrics_export_import() {
        let mut tracker = MetricsTracker::new();
        tracker.log("test_metric", 1.0);
        
        // Export
        let json = tracker.export_json().unwrap();
        
        // Import into new tracker
        let mut new_tracker = MetricsTracker::new();
        new_tracker.import_json(&json).unwrap();
        
        assert_eq!(new_tracker.get_latest("test_metric"), Some(1.0));
    }
}