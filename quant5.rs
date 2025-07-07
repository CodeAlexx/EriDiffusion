// quanto_dynamic_forward.rs - Accurate Quanto-style dynamic quantization

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use anyhow::Result;
use candle_core::{Tensor, Device, DType, Module};

/// Quanto module states
#[derive(Debug, Clone, PartialEq)]
pub enum QuantoState {
    /// Dynamic quantization on every forward pass
    Dynamic,
    /// Frozen - weights are pre-quantized and stored
    Frozen,
}

/// Quanto-style quantized linear layer that quantizes on forward pass
pub struct QuantoDynamicLinear {
    /// Original FP32/FP16 weights (kept until frozen)
    weight: Arc<RwLock<Tensor>>,
    bias: Option<Tensor>,
    /// Quantization config
    qtype: QuantizationType,
    /// Current state
    state: Arc<RwLock<QuantoState>>,
    /// Pre-quantized weights (only when frozen)
    frozen_weight: Arc<RwLock<Option<QuantizedTensor>>>,
    /// Activation scales for calibration
    activation_scale: Arc<RwLock<Option<(f32, f32)>>>, // (min, max)
}

impl QuantoDynamicLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>, qtype: QuantizationType) -> Self {
        Self {
            weight: Arc::new(RwLock::new(weight)),
            bias,
            qtype,
            state: Arc::new(RwLock::new(QuantoState::Dynamic)),
            frozen_weight: Arc::new(RwLock::new(None)),
            activation_scale: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Freeze the layer - convert to static quantization
    pub fn freeze(&self) -> Result<()> {
        let weight = self.weight.read().unwrap();
        let quantized = self.quantize_weight(&weight)?;
        
        *self.frozen_weight.write().unwrap() = Some(quantized);
        *self.state.write().unwrap() = QuantoState::Frozen;
        
        Ok(())
    }
    
    /// Quantize weight (called on every forward if not frozen)
    fn quantize_weight(&self, weight: &Tensor) -> Result<QuantizedTensor> {
        match self.qtype {
            QuantizationType::Int8 => quantize_int8_symmetric(weight),
            QuantizationType::Int4 => quantize_int4_symmetric(weight),
            _ => Err(anyhow::anyhow!("Unsupported quantization type")),
        }
    }
}

impl Module for QuantoDynamicLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let state = self.state.read().unwrap().clone();
        
        // Get weight - either frozen or dynamically quantized
        let weight_dequantized = match state {
            QuantoState::Dynamic => {
                // DYNAMIC QUANTIZATION ON EVERY FORWARD PASS
                let weight = self.weight.read().unwrap();
                let quantized = self.quantize_weight(&weight)?;
                quantized.dequantize(x.device())?
            }
            QuantoState::Frozen => {
                // Use pre-quantized weight
                let frozen = self.frozen_weight.read().unwrap();
                if let Some(ref qtensor) = *frozen {
                    qtensor.dequantize(x.device())?
                } else {
                    return Err(anyhow::anyhow!("Frozen but no quantized weight"));
                }
            }
        };
        
        // Perform matmul
        let mut output = x.matmul(&weight_dequantized.t()?)?;
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.broadcast_add(bias)?;
        }
        
        Ok(output)
    }
}

/// Improved Quanto Manager with proper dynamic behavior
pub struct DynamicQuantoManager {
    device: Device,
    config: QuantoConfig,
    /// Original model weights (FP32/FP16)
    original_weights: RwLock<HashMap<String, Tensor>>,
    /// Quantized modules
    modules: RwLock<HashMap<String, Arc<QuantoDynamicLinear>>>,
    /// Global state
    state: RwLock<QuantoState>,
}

impl DynamicQuantoManager {
    pub fn new(device: Device, config: QuantoConfig) -> Self {
        Self {
            device,
            config,
            original_weights: RwLock::new(HashMap::new()),
            modules: RwLock::new(HashMap::new()),
            state: RwLock::new(QuantoState::Dynamic),
        }
    }
    
    /// Quantize a model (creates dynamic modules, doesn't pre-quantize)
    pub fn quantize_model(&self, weights: HashMap<String, Tensor>) -> Result<()> {
        *self.original_weights.write().unwrap() = weights.clone();
        let mut modules = self.modules.write().unwrap();
        
        for (name, weight) in weights {
            if self.should_quantize(&name) && weight.dims().len() == 2 {
                let module = Arc::new(QuantoDynamicLinear::new(
                    weight,
                    None, // Bias handled separately
                    self.config.weights,
                ));
                modules.insert(name, module);
            }
        }
        
        log::info!("Create
