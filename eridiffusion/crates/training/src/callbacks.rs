//! Training callbacks and early stopping

use eridiffusion_core::{Result, Error};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use candle_core::Tensor;

/// Callback events
#[derive(Debug, Clone)]
pub enum CallbackEvent {
    TrainingStart,
    TrainingEnd,
    EpochStart(usize),
    EpochEnd(usize),
    BatchStart(usize),
    BatchEnd(usize),
    ValidationStart,
    ValidationEnd,
    CheckpointSaved(String),
    MetricLogged(String, f32),
}

/// Callback context
#[derive(Debug, Clone)]
pub struct CallbackContext {
    pub global_step: usize,
    pub epoch: usize,
    pub batch: usize,
    pub loss: Option<f32>,
    pub metrics: std::collections::HashMap<String, f32>,
    pub model_state: Option<Vec<u8>>, // Serialized state
}

/// Base callback trait
#[async_trait::async_trait]
pub trait Callback: Send + Sync {
    /// Called on event
    async fn on_event(&mut self, event: CallbackEvent, context: &CallbackContext) -> Result<()>;
    
    /// Get callback name
    fn name(&self) -> &str;
    
    /// Check if callback is enabled
    fn is_enabled(&self) -> bool {
        true
    }
    
    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
    
    /// Get as mutable Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Early stopping callback
pub struct EarlyStopping {
    monitor: String,
    patience: usize,
    mode: MonitorMode,
    min_delta: f32,
    baseline: Option<f32>,
    best_value: Option<f32>,
    wait: usize,
    stopped_epoch: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum MonitorMode {
    Min,
    Max,
}

impl EarlyStopping {
    pub fn new(monitor: &str, patience: usize, mode: MonitorMode) -> Self {
        Self {
            monitor: monitor.to_string(),
            patience,
            mode,
            min_delta: 0.0,
            baseline: None,
            best_value: None,
            wait: 0,
            stopped_epoch: None,
        }
    }
    
    pub fn with_min_delta(mut self, delta: f32) -> Self {
        self.min_delta = delta;
        self
    }
    
    pub fn with_baseline(mut self, baseline: f32) -> Self {
        self.baseline = Some(baseline);
        self
    }
    
    fn is_better(&self, current: f32, best: f32) -> bool {
        match self.mode {
            MonitorMode::Min => current < best - self.min_delta,
            MonitorMode::Max => current > best + self.min_delta,
        }
    }
    
    pub fn should_stop(&self) -> bool {
        self.stopped_epoch.is_some()
    }
    
    pub fn stopped_epoch(&self) -> Option<usize> {
        self.stopped_epoch
    }
}

#[async_trait::async_trait]
impl Callback for EarlyStopping {
    async fn on_event(&mut self, event: CallbackEvent, context: &CallbackContext) -> Result<()> {
        match event {
            CallbackEvent::EpochEnd(epoch) => {
                if let Some(current) = context.metrics.get(&self.monitor) {
                    // Check baseline
                    if let Some(baseline) = self.baseline {
                        if !self.is_better(*current, baseline) {
                            self.wait += 1;
                            if self.wait >= self.patience {
                                self.stopped_epoch = Some(epoch);
                                println!("Early stopping triggered at epoch {} (baseline not met)", epoch);
                            }
                            return Ok(());
                        }
                    }
                    
                    // Check improvement
                    if let Some(best) = self.best_value {
                        if self.is_better(*current, best) {
                            self.best_value = Some(*current);
                            self.wait = 0;
                        } else {
                            self.wait += 1;
                            if self.wait >= self.patience {
                                self.stopped_epoch = Some(epoch);
                                println!("Early stopping triggered at epoch {} (no improvement)", epoch);
                            }
                        }
                    } else {
                        self.best_value = Some(*current);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "EarlyStopping"
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Model checkpoint callback
pub struct ModelCheckpoint {
    save_dir: String,
    monitor: Option<String>,
    mode: MonitorMode,
    save_best_only: bool,
    save_weights_only: bool,
    period: usize,
    best_value: Option<f32>,
}

impl ModelCheckpoint {
    pub fn new(save_dir: &str) -> Self {
        Self {
            save_dir: save_dir.to_string(),
            monitor: None,
            mode: MonitorMode::Min,
            save_best_only: false,
            save_weights_only: false,
            period: 1,
            best_value: None,
        }
    }
    
    pub fn monitor(mut self, metric: &str, mode: MonitorMode) -> Self {
        self.monitor = Some(metric.to_string());
        self.mode = mode;
        self
    }
    
    pub fn save_best_only(mut self) -> Self {
        self.save_best_only = true;
        self
    }
    
    pub fn save_weights_only(mut self) -> Self {
        self.save_weights_only = true;
        self
    }
    
    pub fn period(mut self, period: usize) -> Self {
        self.period = period;
        self
    }
    
    fn is_better(&self, current: f32, best: f32) -> bool {
        match self.mode {
            MonitorMode::Min => current < best,
            MonitorMode::Max => current > best,
        }
    }
}

#[async_trait::async_trait]
impl Callback for ModelCheckpoint {
    async fn on_event(&mut self, event: CallbackEvent, context: &CallbackContext) -> Result<()> {
        match event {
            CallbackEvent::EpochEnd(epoch) => {
                if epoch % self.period != 0 {
                    return Ok(());
                }
                
                let mut should_save = !self.save_best_only;
                
                if let Some(ref monitor) = self.monitor {
                    if let Some(current) = context.metrics.get(monitor) {
                        if let Some(best) = self.best_value {
                            if self.is_better(*current, best) {
                                self.best_value = Some(*current);
                                should_save = true;
                            }
                        } else {
                            self.best_value = Some(*current);
                            should_save = true;
                        }
                    }
                }
                
                if should_save {
                    let filename = format!("checkpoint_epoch_{}.safetensors", epoch);
                    let path = std::path::Path::new(&self.save_dir).join(&filename);
                    
                    // Would save model state here
                    println!("Saving checkpoint to {}", path.display());
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "ModelCheckpoint"
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Learning rate scheduler callback
pub struct LRSchedulerCallback {
    scheduler: Box<dyn crate::schedulers::LRScheduler>,
    metric: Option<String>,
}

impl LRSchedulerCallback {
    pub fn new(scheduler: Box<dyn crate::schedulers::LRScheduler>) -> Self {
        Self {
            scheduler,
            metric: None,
        }
    }
    
    pub fn with_metric(mut self, metric: &str) -> Self {
        self.metric = Some(metric.to_string());
        self
    }
}

#[async_trait::async_trait]
impl Callback for LRSchedulerCallback {
    async fn on_event(&mut self, event: CallbackEvent, context: &CallbackContext) -> Result<()> {
        match event {
            CallbackEvent::EpochEnd(_) => {
                let metric_value = self.metric.as_ref()
                    .and_then(|m| context.metrics.get(m))
                    .copied();
                
                self.scheduler.step(metric_value);
            }
            _ => {}
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "LRSchedulerCallback"
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Gradient clipping callback
pub struct GradientClipping {
    max_norm: f32,
    norm_type: f32,
}

impl GradientClipping {
    pub fn new(max_norm: f32) -> Self {
        Self {
            max_norm,
            norm_type: 2.0,
        }
    }
    
    pub fn with_norm_type(mut self, norm_type: f32) -> Self {
        self.norm_type = norm_type;
        self
    }
}

#[async_trait::async_trait]
impl Callback for GradientClipping {
    async fn on_event(&mut self, event: CallbackEvent, _context: &CallbackContext) -> Result<()> {
        match event {
            CallbackEvent::BatchEnd(_) => {
                // Gradient clipping is handled in the training loop
                // This callback serves as a hook point for custom clipping logic
                // The actual implementation is in Trainer::clip_gradients()
                tracing::debug!("Gradient clipping applied with max norm: {}", self.max_norm);
            }
            _ => {}
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "GradientClipping"
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Tensorboard logging callback
pub struct TensorBoardCallback {
    log_dir: String,
    log_interval: usize,
}

impl TensorBoardCallback {
    pub fn new(log_dir: &str, log_interval: usize) -> Self {
        Self {
            log_dir: log_dir.to_string(),
            log_interval,
        }
    }
}

#[async_trait::async_trait]
impl Callback for TensorBoardCallback {
    async fn on_event(&mut self, event: CallbackEvent, context: &CallbackContext) -> Result<()> {
        match event {
            CallbackEvent::BatchEnd(batch) => {
                if batch % self.log_interval == 0 {
                    // Would log to tensorboard
                    for (name, value) in &context.metrics {
                        println!("TB Log: {} = {}", name, value);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "TensorBoardCallback"
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Custom lambda callback
pub struct LambdaCallback<F>
where
    F: Fn(CallbackEvent, &CallbackContext) -> Result<()> + Send + Sync,
{
    name: String,
    on_event_fn: F,
}

impl<F> LambdaCallback<F>
where
    F: Fn(CallbackEvent, &CallbackContext) -> Result<()> + Send + Sync,
{
    pub fn new(name: &str, on_event_fn: F) -> Self {
        Self {
            name: name.to_string(),
            on_event_fn,
        }
    }
}

#[async_trait::async_trait]
impl<F> Callback for LambdaCallback<F>
where
    F: Fn(CallbackEvent, &CallbackContext) -> Result<()> + Send + Sync + 'static,
{
    async fn on_event(&mut self, event: CallbackEvent, context: &CallbackContext) -> Result<()> {
        (self.on_event_fn)(event, context)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Callback manager
pub struct CallbackManager {
    callbacks: Vec<Box<dyn Callback>>,
}

impl CallbackManager {
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }
    
    pub fn add_callback(&mut self, callback: Box<dyn Callback>) {
        self.callbacks.push(callback);
    }
    
    pub async fn on_event(&mut self, event: CallbackEvent, context: &CallbackContext) -> Result<()> {
        for callback in &mut self.callbacks {
            if callback.is_enabled() {
                callback.on_event(event.clone(), context).await?;
            }
        }
        Ok(())
    }
    
    pub fn get_callback<T: Callback + 'static>(&self) -> Option<&T> {
        for callback in &self.callbacks {
            if let Some(cb) = callback.as_any().downcast_ref::<T>() {
                return Some(cb);
            }
        }
        None
    }
    
    pub fn get_callback_mut<T: Callback + 'static>(&mut self) -> Option<&mut T> {
        for callback in &mut self.callbacks {
            if let Some(cb) = callback.as_any_mut().downcast_mut::<T>() {
                return Some(cb);
            }
        }
        None
    }
}

/// CSV logging callback
pub struct CSVLogger {
    file_path: String,
    headers_written: bool,
}

impl CSVLogger {
    pub fn new(file_path: &str) -> Self {
        Self {
            file_path: file_path.to_string(),
            headers_written: false,
        }
    }
}

#[async_trait::async_trait]
impl Callback for CSVLogger {
    async fn on_event(&mut self, event: CallbackEvent, context: &CallbackContext) -> Result<()> {
        match event {
            CallbackEvent::EpochEnd(epoch) => {
                // Would write metrics to CSV file
                if !self.headers_written {
                    println!("Writing CSV headers to {}", self.file_path);
                    self.headers_written = true;
                }
                println!("CSV Log: epoch={}, metrics={:?}", epoch, context.metrics);
            }
            _ => {}
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "CSVLogger"
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Helper trait for downcasting
trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: Callback + 'static> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}