//! Minimal, Flame-compatible 8-bit AdamW stubs to keep builds green.

#[derive(Debug, Clone)]
pub struct AdamW8bitConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub quantile_alpha: f32,
    pub adaptive_quantization: bool,
    pub quantization_warmup_steps: usize,
}

impl Default for AdamW8bitConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            quantile_alpha: 0.99,
            adaptive_quantization: true,
            quantization_warmup_steps: 100,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AdamW8bit {
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
}

impl AdamW8bit {
    pub fn new(
        _vars_unused: (),
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> anyhow::Result<Self> {
        Ok(Self { lr, beta1, beta2, epsilon, weight_decay })
    }

    pub fn learning_rate(&self) -> f64 {
        self.lr
    }
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

#[derive(Debug)]
pub struct OptimizerMemoryReport {
    pub num_params: usize,
    pub param_memory_gb: f64,
    pub state_memory_gb: f64,
    pub total_memory_gb: f64,
    pub memory_reduction_percent: Option<f64>,
    pub description: String,
}

impl std::fmt::Display for OptimizerMemoryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Optimizer Memory Analysis:")?;
        writeln!(
            f,
            "  Parameters: {:.1}B ({:.2} GB)",
            self.num_params as f64 / 1e9,
            self.param_memory_gb
        )?;
        writeln!(f, "  States: {} ({:.2} GB)", self.description, self.state_memory_gb)?;
        writeln!(f, "  Total: {:.2} GB", self.total_memory_gb)
    }
}

pub fn analyze_optimizer_memory(num_params: usize, precision: &str) -> OptimizerMemoryReport {
    let param_bytes = num_params * 4; // FP32 params
    let (state_bytes, description) = match precision {
        "fp32" => (param_bytes * 2, "FP32 momentum states"),
        "fp16" => (param_bytes, "FP16 momentum states"),
        "8bit" => (param_bytes / 2, "8-bit quantized momentum states"),
        "4bit" => (param_bytes / 4, "4-bit quantized momentum states"),
        _ => (param_bytes * 2, "Unknown precision"),
    };
    let total_bytes = param_bytes + state_bytes;
    let memory_reduction = if precision != "fp32" {
        Some(1.0 - (total_bytes as f64 / (param_bytes * 3) as f64))
    } else {
        None
    };
    OptimizerMemoryReport {
        num_params,
        param_memory_gb: param_bytes as f64 / 1e9,
        state_memory_gb: state_bytes as f64 / 1e9,
        total_memory_gb: total_bytes as f64 / 1e9,
        memory_reduction_percent: memory_reduction.map(|r| r * 100.0),
        description: description.to_string(),
    }
}
