use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, anyhow};
use flame_core::{Parameter, Tensor, Device, DType, Shape};
use safetensors::tensor::TensorView;

pub struct EMAModel {
    decay: f32,
    step: u64,
    device: Device,
    shadow: HashMap<String, Tensor>,
}

impl EMAModel {
    pub fn new(decay: f32, device: Device) -> Self {
        Self { decay, step: 0, device, shadow: HashMap::new() }
    }

    pub fn init_from_params(&mut self, params: &HashMap<String, Parameter>) -> Result<()> {
        self.shadow.clear();
        for (k, p) in params {
            let t = p.tensor().map_err(|e| anyhow!(e.to_string()))?;
            // Store FP32 copy for stability
            let t32 = if t.dtype() != DType::F32 { t.to_dtype(DType::F32).map_err(|e| anyhow!(e.to_string()))? } else { t };
            self.shadow.insert(k.clone(), t32);
        }
        Ok(())
    }

    pub fn update(&mut self, params: &HashMap<String, Parameter>) -> Result<()> {
        self.step += 1;
        let beta = self.decay;
        for (k, p) in params {
            let src = p.tensor().map_err(|e| anyhow!(e.to_string()))?;
            let src32 = if src.dtype() != DType::F32 { src.to_dtype(DType::F32).map_err(|e| anyhow!(e.to_string()))? } else { src };
            let dst = self.shadow.get_mut(k).ok_or_else(|| anyhow!(format!("EMA missing param key: {}", k)))?;
            let s_scaled = dst.mul_scalar(beta).map_err(|e| anyhow!(e.to_string()))?;
            let p_scaled = src32.mul_scalar(1.0 - beta).map_err(|e| anyhow!(e.to_string()))?;
            *dst = s_scaled.add(&p_scaled).map_err(|e| anyhow!(e.to_string()))?;
        }
        Ok(())
    }

    pub fn save_state(&self, path: &Path, dtype: &str) -> Result<()> {
        // Serialize named tensors as requested dtype
        let dt = match dtype.to_ascii_lowercase().as_str() { "f32" => safetensors::Dtype::F32, "f16" => safetensors::Dtype::F16, _ => safetensors::Dtype::BF16 };
        let mut storage: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        for (name, t) in self.shadow.iter() {
            let v: Vec<f32> = t.to_vec().map_err(|e| anyhow!(e.to_string()))?;
            let bytes: Vec<u8> = match dt {
                safetensors::Dtype::F32 => {
                    let mut b = vec![0u8; v.len()*4];
                    for (i, f) in v.iter().enumerate() { b[i*4..(i+1)*4].copy_from_slice(&f.to_le_bytes()); }
                    b
                }
                safetensors::Dtype::F16 => {
                    let mut b = vec![0u8; v.len()*2];
                    for (i, f) in v.iter().enumerate() { let h = half::f16::from_f32(*f); b[i*2..(i+1)*2].copy_from_slice(&h.to_le_bytes()); }
                    b
                }
                safetensors::Dtype::BF16 => {
                    let mut b = vec![0u8; v.len()*2];
                    for (i, f) in v.iter().enumerate() { let h = half::bf16::from_f32(*f); b[i*2..(i+1)*2].copy_from_slice(&h.to_le_bytes()); }
                    b
                }
                _ => unreachable!(),
            };
            storage.push((name.clone(), bytes, t.shape().dims().to_vec()));
        }
        let mut map = std::collections::BTreeMap::<String, TensorView>::new();
        for (name, bytes, shape) in storage.iter() {
            let tv = TensorView::new(dt, shape.clone(), bytes)?;
            map.insert(name.clone(), tv);
        }
        let bin = safetensors::serialize(map, &None)?;
        std::fs::write(path, bin)?;
        Ok(())
    }

    pub fn load_state(&mut self, path: &Path) -> Result<()> {
        let data = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&data)?;
        let mut new_shadow: HashMap<String, Tensor> = HashMap::new();
        for k in st.names() {
            let view = st.tensor(k)?;
            let shape = Shape::from_dims(&view.shape().to_vec());
            let t: Tensor = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let mut v = vec![0f32; view.data().len()/4];
                    for i in 0..v.len() { v[i] = f32::from_le_bytes([view.data()[i*4], view.data()[i*4+1], view.data()[i*4+2], view.data()[i*4+3]]); }
                    Tensor::from_slice(&v, shape, self.device.cuda_device_arc()).map_err(|e| anyhow!(e.to_string()))?
                }
                safetensors::Dtype::F16 => {
                    let mut v = vec![0f32; view.data().len()/2];
                    for i in 0..v.len() { let h = half::f16::from_le_bytes([view.data()[i*2], view.data()[i*2+1]]); v[i] = h.to_f32(); }
                    Tensor::from_slice(&v, shape, self.device.cuda_device_arc()).map_err(|e| anyhow!(e.to_string()))?
                }
                safetensors::Dtype::BF16 => {
                    let mut v = vec![0f32; view.data().len()/2];
                    for i in 0..v.len() { let h = half::bf16::from_le_bytes([view.data()[i*2], view.data()[i*2+1]]); v[i] = f32::from(h); }
                    Tensor::from_slice(&v, shape, self.device.cuda_device_arc()).map_err(|e| anyhow!(e.to_string()))?
                }
                _ => return Err(anyhow!("Unsupported dtype in EMA load"))
            };
            new_shadow.insert(k.to_string(), t);
        }
        self.shadow = new_shadow;
        Ok(())
    }

    pub fn step(&self) -> u64 { self.step }
    pub fn state_map(&self) -> &HashMap<String, Tensor> { &self.shadow }
}

pub struct EMAHelper;
impl EMAHelper {
    pub fn with_ema_params<F, R>(ema: &EMAModel, params: &mut HashMap<String, Parameter>, f: F) -> Result<R>
    where F: FnOnce() -> Result<R> {
        // Snapshot
        let backup: HashMap<String, Tensor> = params.iter()
            .map(|(k, p)| (k.clone(), p.tensor().map_err(|e| anyhow!(e.to_string())).unwrap()))
            .collect();
        // Copy EMA shadow
        for (k, p) in params.iter_mut() {
            if let Some(sh) = ema.state_map().get(k) {
                p.set_data(sh.clone()).map_err(|e| anyhow!(e.to_string()))?;
            }
        }
        let out = f();
        // Restore
        for (k, p) in params.iter_mut() {
            if let Some(t) = backup.get(k) {
                p.set_data(t.clone()).map_err(|e| anyhow!(e.to_string()))?;
            }
        }
        out
    }
}

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
        let mut snap: HashMap<String, Tensor> = HashMap::new();
        for (name, p) in self.params.iter() {
            snap.insert(name.clone(), p.tensor()?);
        }
        self.backup = Some(snap);
        eprintln!("[ema] context enter: swapping {} params", self.params.len());
        for (name, p) in self.params.iter_mut() {
            if let Some(sh) = self.ema.shadow.get(name) {
                p.set_data(sh.clone())?;
            }
        }
        Ok(())
    }

    pub fn exit(&mut self) -> Result<()> {
        if let Some(snap) = self.backup.take() {
            for (name, p) in self.params.iter_mut() {
                if let Some(t) = snap.get(name) { p.set_data(t.clone())?; }
            }
        }
        eprintln!("[ema] context exit: restored original params");
        Ok(())
    }
}
