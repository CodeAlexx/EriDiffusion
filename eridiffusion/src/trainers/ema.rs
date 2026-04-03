//! Exponential Moving Average (EMA) for model weights.
//! Stable evaluation/inference by maintaining a moving average of parameters.

use std::collections::HashMap;

use flame_core::device::Device;
use flame_core::{DType, Error, Parameter, Result, Shape, Tensor};
use safetensors::tensor::TensorView;

/// Serialized EMA state (checkpointable)
#[derive(Clone)]
pub struct EMAState {
    pub decay: f32,
    pub step: usize,
    pub shadow_params: HashMap<String, Tensor>,
    pub use_bias_correction: bool,
    pub power: f32,
}

/// EMA tracker for model parameters
pub struct EMAModel {
    decay: f32,
    step: usize,
    shadow_params: HashMap<String, Tensor>,
    device: Device,
    use_bias_correction: bool,
    power: f32,
}

impl EMAModel {
    /// New EMA with base decay.
    pub fn new(decay: f32, device: Device) -> Result<Self> {
        if !(0.0 < decay && decay <= 1.0) {
            return Err(Error::InvalidOperation(format!(
                "EMA decay must be in (0,1], got {}",
                decay
            )));
        }
        Ok(Self {
            decay,
            step: 0,
            shadow_params: HashMap::new(),
            device,
            use_bias_correction: true,
            power: 1.0,
        })
    }

    /// New EMA with full config.
    pub fn with_config(
        decay: f32,
        device: Device,
        use_bias_correction: bool,
        power: f32,
    ) -> Result<Self> {
        let mut m = Self::new(decay, device)?;
        m.use_bias_correction = use_bias_correction;
        m.power = power;
        Ok(m)
    }

    /// Initialize EMA shadows from model params (by name).
    /// NOTE: Uses parameter storage directly (no deep copy).
    pub fn init_from_params(&mut self, params: &HashMap<String, Parameter>) -> Result<()> {
        self.shadow_params.clear();
        for (name, p) in params {
            let t = p.tensor()?;
            self.shadow_params.insert(name.clone(), t);
        }
        Ok(())
    }

    /// Update EMA shadows from current model params.
    pub fn update(&mut self, params: &HashMap<String, Parameter>) -> Result<()> {
        self.step += 1;

        let decay = self.compute_effective_decay();
        for (name, p) in params {
            let cur = p.tensor()?;
            if let Some(shadow) = self.shadow_params.get(name) {
                // shadow = decay * shadow + (1 - decay) * cur
                let decay_t = Tensor::full(shadow.shape().clone(), decay, shadow.device().clone())?;
                let one_minus_t =
                    Tensor::full(cur.shape().clone(), 1.0 - decay, cur.device().clone())?;
                let updated = shadow.mul(&decay_t)?.add(&cur.mul(&one_minus_t)?)?;
                self.shadow_params.insert(name.clone(), updated);
            } else {
                // First sighting of this param name
                self.shadow_params.insert(name.clone(), cur);
            }
        }
        Ok(())
    }

    /// Effective decay (with optional power schedule + bias correction).
    fn compute_effective_decay(&self) -> f32 {
        let mut decay = self.decay;

        // Power schedule: gradually anneal early steps if power > 1
        if self.power != 1.0 {
            let progress = (self.step as f32 / 1000.0).min(1.0); // normalize [0,1]
            decay = decay.powf(self.power * (1.0 - progress) + progress);
        }

        // Bias correction for early steps (keep numerically safe)
        if self.use_bias_correction && self.step > 0 && self.step < 1000 {
            // decay_corrected = decay * (1 - decay^(step-1)) / (1 - decay^step)
            let num = 1.0 - decay.powi((self.step.saturating_sub(1)) as i32);
            let den = (1.0 - decay.powi(self.step as i32)).max(1e-8);
            decay = decay * (num / den);
        }

        decay
    }

    /// Copy EMA shadows into the given params (eval/inference).
    pub fn copy_to(&self, params: &mut HashMap<String, Parameter>) -> Result<()> {
        for (name, param) in params {
            if let Some(shadow) = self.shadow_params.get(name) {
                param.set_data(shadow.clone())?;
            } else {
                return Err(Error::InvalidOperation(format!(
                    "No EMA weights for parameter: {}",
                    name
                )));
            }
        }
        Ok(())
    }

    /// Snapshot current model weights (for restoration).
    pub fn store_params(params: &HashMap<String, Parameter>) -> Result<HashMap<String, Tensor>> {
        let mut stored = HashMap::new();
        for (name, p) in params {
            stored.insert(name.clone(), p.as_tensor()?);
        }
        Ok(stored)
    }

    /// Restore model weights from a snapshot.
    pub fn restore_params(
        params: &mut HashMap<String, Parameter>,
        stored: HashMap<String, Tensor>,
    ) -> Result<()> {
        for (name, t) in stored {
            if let Some(p) = params.get_mut(&name) {
                p.set_data(t)?;
            }
        }
        Ok(())
    }

    /// Get a borrowed view of the shadow map.
    pub fn shadow_params(&self) -> &HashMap<String, Tensor> {
        &self.shadow_params
    }

    /// Current step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Force step (e.g., on resume).
    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }

    /// Current effective decay (read-only).
    pub fn current_decay(&self) -> f32 {
        self.compute_effective_decay()
    }

    /// Build a checkpointable state dict.
    pub fn state_dict(&self) -> EMAState {
        EMAState {
            decay: self.decay,
            step: self.step,
            shadow_params: self.shadow_params.clone(),
            use_bias_correction: self.use_bias_correction,
            power: self.power,
        }
    }

    /// Restore from a state dict (shapes/dtypes must match).
    pub fn load_state_dict(&mut self, state: EMAState) -> Result<()> {
        self.decay = state.decay;
        self.step = state.step;
        self.shadow_params = state.shadow_params;
        self.use_bias_correction = state.use_bias_correction;
        self.power = state.power;
        Ok(())
    }

    /// Save EMA state to a `.safetensors` file.
    /// - `path` should end with `.safetensors`
    /// - `dtype` is the on-disk dtype for tensors: "bf16" | "f16" | "f32"
    pub fn save_state(&self, path: &std::path::Path, dtype: &str) -> Result<()> {
        // Build metadata map to embed into the safetensors header.
        let mut meta = std::collections::HashMap::<String, String>::new();
        meta.insert("ema.decay".into(), self.decay.to_string());
        meta.insert("ema.step".into(), self.step.to_string());
        meta.insert(
            "ema.use_bias_correction".into(),
            if self.use_bias_correction { "1" } else { "0" }.into(),
        );
        meta.insert("ema.power".into(), self.power.to_string());

        // Serialize named tensors
        let dt = match dtype.to_ascii_lowercase().as_str() {
            "f32" => safetensors::Dtype::F32,
            "f16" => safetensors::Dtype::F16,
            _ => safetensors::Dtype::BF16,
        };
        let mut storage: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        for (name, t) in self.shadow_params.iter() {
            let v: Vec<f32> = t.to_vec()?;
            let bytes: Vec<u8> = match dt {
                safetensors::Dtype::F32 => {
                    let mut b = vec![0u8; v.len() * 4];
                    for (i, f) in v.iter().enumerate() {
                        b[i * 4..(i + 1) * 4].copy_from_slice(&f.to_le_bytes());
                    }
                    b
                }
                safetensors::Dtype::F16 => {
                    let mut b = vec![0u8; v.len() * 2];
                    for (i, f) in v.iter().enumerate() {
                        let h = half::f16::from_f32(*f);
                        b[i * 2..(i + 1) * 2].copy_from_slice(&h.to_le_bytes());
                    }
                    b
                }
                safetensors::Dtype::BF16 => {
                    let mut b = vec![0u8; v.len() * 2];
                    for (i, f) in v.iter().enumerate() {
                        let h = half::bf16::from_f32(*f);
                        b[i * 2..(i + 1) * 2].copy_from_slice(&h.to_le_bytes());
                    }
                    b
                }
                _ => unreachable!(),
            };
            storage.push((name.clone(), bytes, t.shape().dims().to_vec()));
        }
        let mut map = std::collections::BTreeMap::<String, TensorView>::new();
        for (name, bytes, shape) in storage.iter() {
            let tv = TensorView::new(dt, shape.clone(), bytes)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
            map.insert(name.clone(), tv);
        }
        let bin = safetensors::serialize(map, &Some(meta))
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        std::fs::write(path, bin).map_err(|e| Error::Io(e.to_string()))?;
        Ok(())
    }

    /// Load EMA state from a `.safetensors` file (and metadata).
    pub fn load_state(&mut self, path: &std::path::Path) -> Result<()> {
        let data = std::fs::read(path).map_err(|e| Error::Io(e.to_string()))?;
        let st = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        let mut named: HashMap<String, Tensor> = HashMap::new();
        for k in st.names() {
            let view = st.tensor(k).map_err(|e| Error::InvalidInput(e.to_string()))?;
            let shape = Shape::from_dims(&view.shape().to_vec());
            let t: Tensor = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let mut v = vec![0f32; view.data().len() / 4];
                    for i in 0..v.len() {
                        v[i] = f32::from_le_bytes([
                            view.data()[i * 4],
                            view.data()[i * 4 + 1],
                            view.data()[i * 4 + 2],
                            view.data()[i * 4 + 3],
                        ]);
                    }
                    Tensor::from_slice(&v, shape, self.device.cuda_device_arc())
                        .map_err(|e| Error::from(e))?
                }
                safetensors::Dtype::F16 => {
                    let mut v = vec![0f32; view.data().len() / 2];
                    for i in 0..v.len() {
                        let h =
                            half::f16::from_le_bytes([view.data()[i * 2], view.data()[i * 2 + 1]]);
                        v[i] = h.to_f32();
                    }
                    Tensor::from_slice(&v, shape, self.device.cuda_device_arc())
                        .map_err(|e| Error::from(e))?
                }
                safetensors::Dtype::BF16 => {
                    let mut v = vec![0f32; view.data().len() / 2];
                    for i in 0..v.len() {
                        let h =
                            half::bf16::from_le_bytes([view.data()[i * 2], view.data()[i * 2 + 1]]);
                        v[i] = f32::from(h);
                    }
                    Tensor::from_slice(&v, shape, self.device.cuda_device_arc())
                        .map_err(|e| Error::from(e))?
                }
                _ => {
                    return Err(Error::InvalidOperation(
                        "Unsupported dtype in EMA safetensors".into(),
                    ))
                }
            };
            named.insert(k.to_string(), t);
        }
        self.shadow_params = named;

        Ok(())
    }
}

/// Helper to apply EMA weights in-place (no restore) or via snapshot.
pub struct EMAContext<'a> {
    ema: &'a EMAModel,
    params: HashMap<String, &'a mut Parameter>,
    backup: Option<HashMap<String, Tensor>>,
}

impl<'a> EMAContext<'a> {
    pub fn new(ema: &'a EMAModel, params: HashMap<String, &'a mut Parameter>) -> Result<Self> {
        Ok(Self { ema, params, backup: None })
    }

    pub fn enter(&mut self) -> Result<()> {
        // Snapshot originals
        let mut snap: HashMap<String, Tensor> = HashMap::new();
        for (name, p) in self.params.iter() {
            snap.insert(name.clone(), p.tensor()?);
        }
        self.backup = Some(snap);
        // Apply EMA
        for (name, p) in self.params.iter_mut() {
            if let Some(sh) = self.ema.shadow_params.get(name) {
                p.set_data(sh.clone())?;
            }
        }
        Ok(())
    }

    pub fn exit(&mut self) -> Result<()> {
        if let Some(snap) = self.backup.take() {
            for (name, p) in self.params.iter_mut() {
                if let Some(t) = snap.get(name) {
                    p.set_data(t.clone())?;
                }
            }
        }
        Ok(())
    }
}

/// Decay that halves contributions after `half_life_steps`.
pub fn decay_from_half_life(half_life_steps: usize) -> f32 {
    0.5f32.powf(1.0 / half_life_steps as f32)
}

/// Decay approximating a moving-average window of size `N`.
pub fn decay_from_window(window_size: usize) -> f32 {
    // Standard EMA-window relation: alpha = 2/(N+1) -> decay = 1 - alpha
    1.0 - 2.0 / (window_size as f32 + 1.0)
}
