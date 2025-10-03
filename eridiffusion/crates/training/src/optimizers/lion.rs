//! Minimal, Flame-compatible Lion stubs to keep builds green.

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LionConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self { lr: 1e-4, beta1: 0.9, beta2: 0.99, weight_decay: 0.0 }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Lion {
    lr: f64,
    beta1: f64,
    beta2: f64,
    weight_decay: f64,
}

impl Lion {
    pub fn new(_vars_unused: Vec<()>, config: LionConfig) -> anyhow::Result<Self> {
        Ok(Self {
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            weight_decay: config.weight_decay,
        })
    }
    pub fn learning_rate(&self) -> f64 {
        self.lr
    }
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

#[allow(dead_code)]
pub enum LionPreset {
    Language,
    Vision,
    FineTune,
    Default,
}

impl LionPreset {
    pub fn config(&self) -> LionConfig {
        match self {
            Self::Language => LionConfig { lr: 3e-4, beta1: 0.95, beta2: 0.98, weight_decay: 0.01 },
            Self::Vision => LionConfig { lr: 1e-4, beta1: 0.9, beta2: 0.99, weight_decay: 0.05 },
            Self::FineTune => {
                LionConfig { lr: 3e-5, beta1: 0.9, beta2: 0.999, weight_decay: 0.001 }
            }
            Self::Default => LionConfig::default(),
        }
    }
}

#[allow(dead_code)]
pub enum LionSchedule {
    Constant,
    Linear { warmup_steps: usize, total_steps: usize },
    Cosine { total_steps: usize, min_lr: f64 },
    ExponentialDecay { decay_rate: f64, decay_steps: usize },
}

impl LionSchedule {
    pub fn get_lr(&self, base_lr: f64, step: usize) -> f64 {
        match self {
            Self::Constant => base_lr,
            Self::Linear { warmup_steps, total_steps } => {
                if step < *warmup_steps {
                    base_lr * (step as f64 / *warmup_steps as f64)
                } else {
                    let progress =
                        (step - warmup_steps) as f64 / (*total_steps - warmup_steps) as f64;
                    base_lr * (1.0 - progress).max(0.0)
                }
            }
            Self::Cosine { total_steps, min_lr } => {
                let progress = (step as f64 / *total_steps as f64).min(1.0);
                let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                min_lr + (base_lr - min_lr) * cosine_decay
            }
            Self::ExponentialDecay { decay_rate, decay_steps } => {
                let exponent = step as f64 / *decay_steps as f64;
                base_lr * decay_rate.powf(exponent)
            }
        }
    }
}

#[allow(dead_code)]
pub struct LionMonitor {
    lr_history: Vec<f64>,
}
impl LionMonitor {
    pub fn new() -> Self {
        Self { lr_history: Vec::new() }
    }
    pub fn record(&mut self, opt: &Lion) {
        self.lr_history.push(opt.learning_rate());
    }
}

#[allow(dead_code)]
pub struct LionTrainingGuide;
impl LionTrainingGuide {
    pub fn convert_adam_lr(adam_lr: f64) -> f64 {
        adam_lr / 10.0
    }
}

pub fn compare_optimizer_memory(_num_params: usize) { /* no-op stub */
}

pub fn create_lion_preset(_vars_unused: Vec<()>, preset: &str) -> anyhow::Result<Lion> {
    let cfg = match preset {
        "language" => LionPreset::Language.config(),
        "vision" => LionPreset::Vision.config(),
        "finetune" => LionPreset::FineTune.config(),
        _ => LionPreset::Default.config(),
    };
    Lion::new(Vec::new(), cfg)
}

pub struct ScheduledLion {
    optimizer: Lion,
    schedule: LionSchedule,
    base_lr: f64,
}
impl ScheduledLion {
    pub fn new(opt: Lion, schedule: LionSchedule, base_lr: f64) -> Self {
        Self { optimizer: opt, schedule, base_lr }
    }
    pub fn current_lr(&self) -> f64 {
        self.optimizer.learning_rate()
    }
}
