use flame_core::{DType, Result, Shape, Tensor, CudaDevice};
use std::sync::Arc;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};

pub struct AdamConfig {
    pub learning_rate: f32,
pub beta1: f32,
pub beta2: f32,
pub eps: f32,
}
pub struct FusedAdamOptimizer {
    config: AdamConfig,
t: i32,
m: Tensor,
v: Tensor,
// For mixed precision
}
pub struct FusedLoRAAdamOptimizer {
    config: AdamConfig,
t: i32,
down_m: Tensor,
down_v: Tensor,
up_m: Tensor,
up_v: Tensor,
}

// Rust wrapper for fused Adam optimizer CUDA kernels

// FLAME uses flame_core::device::Device instead of Device

#[cfg(feature = "cuda")]
extern "C" {
fn launch_adam_update(
param: *mut f32,
m: *mut f32,
v: *mut f32,
grad: *const f32,
lr: f32,
beta1: f32,
beta2: f32,
eps: f32,
t: i32,
size: i32,
stream: *mut std::ffi::c_void,
);

fn launch_adam_update_vectorized(
param: *mut f32,
m: *mut f32,
v: *mut f32,
grad: *const f32,
lr: f32,
beta1: f32,
beta2: f32,
eps: f32,
t: i32,
size: i32,
stream: *mut std::ffi::c_void,
);

fn launch_adamw_update(
param: *mut f32,
m: *mut f32,
v: *mut f32,
grad: *const f32,
lr: f32,
beta1: f32,
beta2: f32,
eps: f32,
weight_decay: f32,
t: i32,
size: i32,
stream: *mut std::ffi::c_void,
);

fn launch_adam_mixed_precision(
param_fp16: *mut f16,
master_param: *mut f32,
m: *mut f32,
v: *mut f32,
grad_fp16: *const f16,
lr: f32,
beta1: f32,
beta2: f32,
eps: f32,
t: i32,
size: i32,
stream: *mut std::ffi::c_void,
);

fn launch_lora_adam_fused(
down_param: *mut f32,
down_m: *mut f32,
down_v: *mut f32,
down_grad: *const f32,
up_param: *mut f32,
up_m: *mut f32,
up_v: *mut f32,
up_grad: *const f32,
lr: f32,
beta1: f32,
beta2: f32,
eps: f32,
t: i32,
down_size: i32,
up_size: i32,
stream: *mut std::ffi::c_void,
);
}

/// Configuration for Adam optimizer

// Extension trait for Tensor to add missing methods




fn sum_dim(&self, dim: usize, device: &CudaDevice) -> flame_core::Result<Tensor> {
// Sum along dimension
self.sum_dim(dim)?
}

fn add_scalar(&self, scalar: f32, device: &CudaDevice) -> flame_core::Result<Tensor> {
// Add scalar to all elements
let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
self.add(&scalar_tensor)
}

fn mul_scalar(&self, scalar: f32, device: &CudaDevice) -> flame_core::Result<Tensor> {
// Multiply all elements by scalar
let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
self.mul(&scalar_tensor)
}

fn square(&self) -> flame_core::Result<Tensor> {
// Element-wise square
self.mul(self)
}

pub weight_decay: Option<f32>}

impl Default for AdamConfig {
fn default() -> Self {
Self {
learning_rate: 1e-3,
beta1: 0.9,
beta2: 0.999,
eps: 1e-8,
weight_decay: None},
}

/// Fused Adam optimizer state

impl FusedAdamOptimizer {
/// Create new optimizer for given parameter shape
pub fn new(param_shape: &Shape, dtype: DType, device: &Device, config: AdamConfig) -> flame_core::Result<Self> {
let m = Tensor::zeros(param_shape, device.cuda_device())?;
let v = Tensor::zeros(param_shape, device.cuda_device())?;

let master_weights = if dtype == DType::F16 {
// For mixed precision, we need FP32 master weights
Some(Tensor::zeros(param_shape, device.cuda_device())?)
} else {
None
};

Ok(Self {
config,
t: 0,
m,
v,
master_weights}),
}

/// Update parameters using fused Adam kernel
pub fn step(&mut self, param: &mut Tensor, grad: &Tensor, device: &CudaDevice) -> flame_core::Result<Tensor> {
self.t += 1;

match param.device() {
Device::from(Arc::new(cuda_device) => {
self.step_cuda(param, grad, cuda_device)
}
_ => {
// Fallback to CPU implementation
self.step_cpu(param, grad)
}
}

#[cfg(feature = "cuda")]
fn step_cuda(&mut self, param: &mut Tensor, grad: &Tensor, device: &Device) -> flame_core::Result<Tensor> {
let size = param.shape().dims().iter().product::<usize>() as i32;

// Get CUDA stream
let stream = device.cuda_stream();

unsafe {
match param.dtype() {
DType::F32 => {
// Use vectorized kernel if size is divisible by 4
if size % 4 == 0 {
launch_adam_update_vectorized(
param.as_cuda_ptr_mut()?,
self.m.as_cuda_ptr_mut()?,
self.v.as_cuda_ptr_mut()?,
grad.as_cuda_ptr()?,
self.config.learning_rate,
self.config.beta1,
self.config.beta2,
self.config.eps,
self.t,
size,
stream as *mut std::ffi::c_void,
);
} else if let Some(weight_decay) = self.config.weight_decay {
launch_adamw_update(
param.as_cuda_ptr_mut()?,
self.m.as_cuda_ptr_mut()?,
self.v.as_cuda_ptr_mut()?,
grad.as_cuda_ptr()?,
self.config.learning_rate,
self.config.beta1,
self.config.beta2,
self.config.eps,
weight_decay,
self.t,
size,
stream as *mut std::ffi::c_void,
);
} else {
launch_adam_update(
param.as_cuda_ptr_mut()?,
self.m.as_cuda_ptr_mut()?,
self.v.as_cuda_ptr_mut()?,
grad.as_cuda_ptr()?,
self.config.learning_rate,
self.config.beta1,
self.config.beta2,
self.config.eps,
self.t,
size,
stream as *mut std::ffi::c_void,
);
}
DType::F16 => {
// Mixed precision update
let master = self.master_weights.as_mut()

launch_adam_mixed_precision(
param.as_cuda_ptr_mut::<f16>()?,
master.as_cuda_ptr_mut()?,
self.m.as_cuda_ptr_mut()?,
self.v.as_cuda_ptr_mut()?,
grad.as_cuda_ptr::<f16>()?,
self.config.learning_rate,
self.config.beta1,
self.config.beta2,
self.config.eps,
self.t,
size,
stream as *mut std::ffi::c_void,
);
}
_ => {
}
}

Ok(())
}

fn step_cpu(&mut self, param: &mut Tensor, grad: &Tensor) -> flame_core::Result<()> {
// CPU fallback implementation
let lr = self.config.learning_rate;
let beta1 = self.config.beta1;
let beta2 = self.config.beta2;
let eps = self.config.eps;

// Update biased first moment estimate
self.m = self.m.mul_scalar(beta1 as f32)?.add(&grad.mul_scalar((1.0 - beta1) as f32)?)?;

// Update biased second raw moment estimate
let grad_sq = grad.square()?;
self.v = self.v.mul_scalar(beta2 as f32)?.add(&grad_sq.mul_scalar((1.0 - beta2) as f32)?)?;

// Compute bias-corrected first moment estimate
let m_hat = self.m.mul_scalar((1.0 / (1.0 - beta1.powi(self.t))) as f32)?;

// Compute bias-corrected second raw moment estimate
let v_hat = self.v.mul_scalar((1.0 / (1.0 - beta2.powi(self.t))) as f32)?;

// Update parameters
let v_sqrt = v_hat.sqrt()?;
let update = m_hat.div(&v_sqrt.add_scalar(eps)?)?;
*param = param.sub(&update.mul_scalar(lr as f32)?)?;

Ok(())
}

/// Fused LoRA Adam optimizer - updates both projections in one kernel

impl FusedLoRAAdamOptimizer {
pub fn new(
down_shape: &Shape,
up_shape: &Shape,
device: &Device,
config: AdamConfig,
) -> flame_core::Result<Self> {
let down_m = Tensor::zeros(down_shape, device.cuda_device())?;
let down_v = Tensor::zeros(down_shape, device.cuda_device())?;
let up_m = Tensor::zeros(up_shape, device.cuda_device())?;
let up_v = Tensor::zeros(up_shape, device.cuda_device())?;

Ok(Self {
config,
t: 0,
down_m,
down_v,
up_m,
up_v})
}

/// Update both LoRA projections in a single kernel
#[cfg(feature = "cuda")]
pub fn step(
&mut self,
down_param: &mut Tensor,
down_grad: &Tensor,
up_param: &mut Tensor,
up_grad: &Tensor,
) -> flame_core::Result<Tensor> {
self.t += 1;

match down_param.device() {
Device::from(Arc::new(cuda_device) => {
let stream = cuda_device.cuda_stream();
let down_size = down_param.shape().dims().iter().product::<usize>() as i32;
let up_size = up_param.shape().dims().iter().product::<usize>() as i32;

unsafe {
launch_lora_adam_fused(
down_param.as_cuda_ptr_mut()?,
self.down_m.as_cuda_ptr_mut()?,
self.down_v.as_cuda_ptr_mut()?,
down_grad.as_cuda_ptr()?,
up_param.as_cuda_ptr_mut()?,
self.up_m.as_cuda_ptr_mut()?,
self.up_v.as_cuda_ptr_mut()?,
up_grad.as_cuda_ptr()?,
self.config.learning_rate,
self.config.beta1,
self.config.beta2,
self.config.eps,
self.t,
down_size,
up_size,
stream as *mut std::ffi::c_void,
);
}
Ok(())
}
_ => {
// CPU fallback - just do two separate updates
let mut down_opt = FusedAdamOptimizer {
config: self.config.clone(),
t: self.t,
m: self.down_m.clone(),
v: self.down_v.clone(),;
down_opt.step(down_param, down_grad)?;
self.down_m = down_opt.m;
self.down_v = down_opt.v;

let mut up_opt = FusedAdamOptimizer {
config: self.config.clone(),
t: self.t,
m: self.up_m.clone(),
v: self.up_v.clone(),;
up_opt.step(up_param, up_grad)?;
self.up_m = up_opt.m;
self.up_v = up_opt.v;

Ok(())
}
}

// Helper trait for getting CUDA pointers
trait CudaTensorExt {
fn as_cuda_ptr<T>(&self) -> flame_core::Result<Tensor>;
fn as_cuda_ptr_mut<T>(&mut self) -> flame_core::Result<Tensor>;
}

impl CudaTensorExt for Tensor {
fn as_cuda_ptr<T>(&self) -> flame_core::Result<Tensor> {
// FLAME tensors store data as CudaSlice
let data = self.data();
Ok(data.as_ptr() as *const T)
}

fn as_cuda_ptr_mut<T>(&mut self) -> flame_core::Result<Tensor> {
// For FLAME, we need to get mutable access to the underlying data
// This is a simplification - in practice FLAME may need a different approach
let data = self.data();
Ok(data.as_ptr() as *mut T)
}
}
}
}
}
}
}
}
