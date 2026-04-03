use flame_core::{Device, Result, Tensor};
use std::collections::{HashMap, VecDeque};

/// Optimized backward pass scheduling for Flux layer streaming
/// This module pre-computes the optimal layer loading order for backward pass
/// to minimize cache misses and disk I/O

pub struct BackwardOptimizer {
    /// Layers that should be kept in memory during backward pass
    critical_layers: Vec<String>,
    /// Pre-computed backward pass order
    backward_order: Vec<String>,
    /// Memory budget in bytes
    memory_budget: usize,
    /// Estimated layer sizes
    layer_sizes: HashMap<String, usize>,
}

impl BackwardOptimizer {
    pub fn new(memory_limit_gb: f32) -> Self {
        // Reserve 80% for gradients and optimizer states
        // This leaves only 20% for pre-loading layers to avoid OOM
        let effective_limit_gb = memory_limit_gb * 0.2;
        let memory_budget = (effective_limit_gb * 1e9) as usize;

        // Critical layers that should stay in memory if possible
        let critical_layers = vec![
            "final_layer".to_string(),
            "img_in".to_string(),
            "txt_in".to_string(),
            // Keep early double blocks in memory as they're used last in backward
            "double_blocks.0".to_string(),
            "double_blocks.1".to_string(),
        ];

        // Pre-compute backward pass order (reverse of forward)
        let mut backward_order = Vec::new();

        // Final layer first
        backward_order.push("final_layer".to_string());

        // Single blocks in reverse (37 down to 0)
        for i in (0..38).rev() {
            backward_order.push(format!("single_blocks.{}", i));
        }

        // Double blocks in reverse (18 down to 0)
        for i in (0..19).rev() {
            backward_order.push(format!("double_blocks.{}", i));
        }

        // Input projections last
        backward_order.push("txt_in".to_string());
        backward_order.push("img_in".to_string());

        // Estimated layer sizes
        let mut layer_sizes = HashMap::new();

        // Input/output layers
        layer_sizes.insert("img_in".to_string(), 400_000); // ~0.4MB
        layer_sizes.insert("txt_in".to_string(), 25_000_000); // ~25MB
        layer_sizes.insert("final_layer".to_string(), 38_000_000); // ~38MB

        // Double blocks (~680MB each)
        for i in 0..19 {
            layer_sizes.insert(format!("double_blocks.{}", i), 680_000_000);
        }

        // Single blocks (~283MB each)
        for i in 0..38 {
            layer_sizes.insert(format!("single_blocks.{}", i), 283_000_000);
        }

        Self { critical_layers, backward_order, memory_budget, layer_sizes }
    }

    /// Get the optimal loading order for backward pass
    pub fn get_backward_order(&self) -> &[String] {
        &self.backward_order
    }

    /// Determine which layers to pre-load for backward pass
    pub fn get_preload_layers(&self) -> Vec<String> {
        let mut preload = Vec::new();
        let mut used_memory = 0;

        // First, add critical layers
        for layer in &self.critical_layers {
            if let Some(&size) = self.layer_sizes.get(layer) {
                if used_memory + size <= self.memory_budget {
                    preload.push(layer.clone());
                    used_memory += size;
                }
            }
        }

        // Then add layers in backward order until memory is full
        for layer in &self.backward_order {
            if preload.contains(layer) {
                continue;
            }

            if let Some(&size) = self.layer_sizes.get(layer) {
                if used_memory + size <= self.memory_budget {
                    preload.push(layer.clone());
                    used_memory += size;
                } else {
                    // Memory full, stop
                    break;
                }
            }
        }

        println!(
            "🎯 Backward optimizer: pre-loading {} layers ({:.2} GB / {:.2} GB)",
            preload.len(),
            used_memory as f32 / 1e9,
            self.memory_budget as f32 / 1e9
        );

        preload
    }

    /// Get layers that should never be evicted during backward pass
    pub fn get_persistent_layers(&self) -> &[String] {
        &self.critical_layers
    }

    /// Estimate memory needed for a specific layer during backward pass
    /// (includes gradients and optimizer states)
    pub fn estimate_backward_memory(&self, layer_name: &str) -> usize {
        if let Some(&forward_size) = self.layer_sizes.get(layer_name) {
            // Backward pass needs:
            // - Forward activations (1x)
            // - Gradients (1x)
            // - Optimizer states (2x for Adam: momentum + variance)
            forward_size * 4
        } else {
            0
        }
    }

    /// Check if we should keep a layer in memory based on backward pass needs
    pub fn should_keep_layer(&self, layer_name: &str, current_step: &str) -> bool {
        // Always keep critical layers
        if self.critical_layers.contains(&layer_name.to_string()) {
            return true;
        }

        // Check if this layer will be needed soon in backward pass
        if let Some(current_idx) = self.backward_order.iter().position(|l| l == current_step) {
            if let Some(layer_idx) = self.backward_order.iter().position(|l| l == layer_name) {
                // Keep if within next 5 layers
                return layer_idx > current_idx && layer_idx <= current_idx + 5;
            }
        }

        false
    }

    /// Get list of critical layers that should not be evicted
    pub fn get_critical_layers(&self) -> Vec<&str> {
        self.critical_layers.iter().map(|s| s.as_str()).collect()
    }

    /// Get priority of a layer for eviction (higher = keep longer)
    pub fn get_layer_priority(&self, layer_name: &str) -> f32 {
        // Critical layers have highest priority
        if self.critical_layers.contains(&layer_name.to_string()) {
            return 100.0;
        }

        // Find position in backward order
        if let Some(idx) = self.backward_order.iter().position(|l| l == layer_name) {
            // Earlier in backward order = higher priority
            let normalized_pos = idx as f32 / self.backward_order.len() as f32;
            return 90.0 * (1.0 - normalized_pos);
        }

        // Unknown layers get low priority
        1.0
    }
}

/// Extension trait for FluxLayerStreamer to add backward optimization
pub trait BackwardOptimization {
    /// Prepare for backward pass by pre-loading layers
    fn prepare_backward_pass(&mut self, optimizer: &BackwardOptimizer) -> Result<()>;

    /// Optimize layer eviction for backward pass
    fn evict_for_backward(
        &mut self,
        optimizer: &BackwardOptimizer,
        current_layer: &str,
    ) -> Result<()>;
}
