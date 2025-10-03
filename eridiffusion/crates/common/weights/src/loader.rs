use anyhow::{Context, Result, bail};
use safetensors::{SafeTensors, tensor::TensorView};
use std::{collections::HashSet, fs, path::PathBuf};
use flame_core::{Tensor, Shape, DType, CudaDevice};
use crate::dtype::map_st_dtype;
use crate::read;

pub struct SafeLoader {
    path: PathBuf,
    st: SafeTensors<'static>,
    used: HashSet<String>,
    allow_prefixes: Vec<String>,
}

impl SafeLoader {
    pub fn open(path: &str) -> Result<Self> {
        let pb = PathBuf::from(path);
        let bytes = fs::read(&pb).with_context(|| format!("reading weights: {}", pb.display()))?;
        // Leak bytes to extend lifetime for SafeTensors (cheap for loader process)
        let boxed = bytes.into_boxed_slice();
        let static_ref: &'static [u8] = Box::leak(boxed);
        let st = SafeTensors::deserialize(static_ref)?;
        Ok(Self { path: pb, st, used: HashSet::new(), allow_prefixes: Vec::new() })
    }

    pub fn list_keys(&self) -> Result<Vec<String>> {
        Ok(self.st.names().into_iter().map(|s| s.to_string()).collect())
    }

    pub fn with_allowlist_prefixes(mut self, prefixes: &[&str]) -> Self {
        self.allow_prefixes = prefixes.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Internal: convert a TensorView to a device Tensor of the target dtype
    fn load_tensor_view(device: std::sync::Arc<CudaDevice>, view: &TensorView<'_>, target: DType) -> Result<Tensor> {
        let shape = Shape::from_dims(view.shape());
        let raw = view.data();
        match (map_st_dtype(&view.dtype())?, target) {
            (DType::F32, DType::F32) => {
                let src: &[f32] = bytemuck::try_cast_slice::<u8, f32>(raw)
                    .map_err(|e| anyhow::anyhow!("bytemuck cast error: {}", e))?;
                Ok(Tensor::from_slice(src, shape, device.clone())?)
            }
            (DType::F32, DType::BF16) => {
                let src: &[f32] = bytemuck::try_cast_slice::<u8, f32>(raw)
                    .map_err(|e| anyhow::anyhow!("bytemuck cast error: {}", e))?;
                let t = Tensor::from_slice(src, shape, device.clone())?;
                Ok(t.to_dtype(DType::BF16)?)
            }
            (DType::BF16, DType::BF16) => {
                let v = read::to_vec_f32_from_bf16(raw)?;
                let t = Tensor::from_slice(&v, shape, device.clone())?;
                Ok(t.to_dtype(DType::BF16)?)
            }
            (DType::F16, DType::BF16) => {
                let v = read::to_vec_f32_from_f16(raw)?;
                let t = Tensor::from_slice(&v, shape, device.clone())?;
                Ok(t.to_dtype(DType::BF16)?)
            }
            (DType::BF16, DType::F32) => {
                let v = read::to_vec_f32_from_bf16(raw)?;
                Ok(Tensor::from_slice(&v, shape, device.clone())?)
            }
            (DType::F32, DType::I32) => {
                let v: &[f32] = bytemuck::try_cast_slice::<u8, f32>(raw)
                    .map_err(|e| anyhow::anyhow!("bytemuck cast error: {}", e))?;
                let t = Tensor::from_slice(v, shape, device.clone())?;
                Ok(t.to_dtype(DType::I32)?)
            }
            (DType::I32, DType::I32) => {
                let v_i32 = read::to_vec_i32(raw)?;
                let v_f32: Vec<f32> = v_i32.into_iter().map(|x| x as f32).collect();
                let t = Tensor::from_slice(&v_f32, shape, device.clone())?;
                Ok(t.to_dtype(DType::I32)?)
            }
            (DType::Bool, DType::Bool) => {
                let v_u8 = read::to_vec_bool_u8(raw)?;
                let v_f32: Vec<f32> = v_u8.into_iter().map(|x| if x == 0 { 0.0 } else { 1.0 }).collect();
                let t = Tensor::from_slice(&v_f32, shape, device.clone())?;
                Ok(t.to_dtype(DType::Bool)?)
            }
            (src, dst) => bail!("unsupported loader conversion: {:?}→{:?}", src, dst),
        }
    }

    /// Fetch a tensor by key and return BF16 param on CUDA:0 device (no_grad)
    pub fn get_bf16(&mut self, key: &str) -> Result<Tensor> {
        self.get_as(key, DType::BF16)
    }

    /// Fetch a tensor by key converted to the requested dtype on CUDA:0
    pub fn get_as(&mut self, key: &str, target: DType) -> Result<Tensor> {
        let tv: TensorView<'_> = self.st.tensor(key)
            .with_context(|| format!("missing key '{}' in {}", key, self.path.display()))?;
        self.used.insert(key.to_string());
        let device = CudaDevice::new(0)?;
        Self::load_tensor_view(device, &tv, target)
    }

    /// Get raw tensor shape from safetensors metadata
    pub fn shape_of(&self, key: &str) -> Result<Vec<usize>> {
        let tv: TensorView<'_> = self.st.tensor(key)?;
        Ok(tv.shape().to_vec())
    }

    /// Verify strict usage policy: all expected keys exist; all file keys are either used or allowlisted.
    pub fn verify_strict(&self, expected: &[&str]) -> Result<()> {
        // Check missing
        for k in expected {
            if self.st.tensor(k).is_err() {
                bail!("strict loader: expected missing key '{}' in {}", k, self.path.display());
            }
        }
        // Check unused
        let mut extras = Vec::new();
        'outer: for name in self.st.names() {
            if self.used.contains(name) { continue; }
            for p in &self.allow_prefixes {
                if name.starts_with(p) { continue 'outer; }
            }
            extras.push(name.to_string());
        }
        if !extras.is_empty() {
            bail!("strict loader: unused keys not allowed ({}), examples: {:?}", extras.len(), &extras[..extras.len().min(5)])
        }
        Ok(())
    }
}
