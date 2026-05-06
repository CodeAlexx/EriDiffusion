use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelType {
    #[serde(rename = "STABLE_DIFFUSION_15")]
    #[default]
    StableDiffusion15,
    #[serde(rename = "STABLE_DIFFUSION_20")]
    StableDiffusion20,
    #[serde(rename = "STABLE_DIFFUSION_21")]
    StableDiffusion21,
    #[serde(rename = "STABLE_DIFFUSION_XL_10_BASE")]
    StableDiffusionXL10Base,
    #[serde(rename = "STABLE_DIFFUSION_3")]
    StableDiffusion3,
    #[serde(rename = "STABLE_DIFFUSION_35")]
    StableDiffusion35,
    #[serde(rename = "WUERSTCHEN_2")]
    Wuerstchen2,
    #[serde(rename = "STABLE_CASCADE_1")]
    StableCascade1,
    #[serde(rename = "PIXART_ALPHA")]
    PixArtAlpha,
    #[serde(rename = "PIXART_SIGMA")]
    PixArtSigma,
    #[serde(rename = "FLUX_DEV_1")]
    FluxDev1,
    #[serde(rename = "FLUX_FILL_DEV_1")]
    FluxFillDev1,
    #[serde(rename = "FLUX_2")]
    Flux2,
    #[serde(rename = "SANA")]
    Sana,
    #[serde(rename = "HUNYUAN_VIDEO")]
    HunyuanVideo,
    #[serde(rename = "HI_DREAM_FULL")]
    HiDreamFull,
    #[serde(rename = "CHROMA_1")]
    Chroma1,
    #[serde(rename = "QWEN")]
    Qwen,
    #[serde(rename = "Z_IMAGE")]
    ZImage,
    #[serde(rename = "ERNIE")]
    Ernie,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrainingMethod {
    #[serde(rename = "FINE_TUNE")]
    #[default]
    FineTune,
    #[serde(rename = "LORA")]
    Lora,
    #[serde(rename = "EMBEDDING")]
    Embedding,
    #[serde(rename = "FINE_TUNE_VAE")]
    FineTuneVae,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DataType {
    #[default]
    #[serde(rename = "FLOAT_32")]
    Float32,
    #[serde(rename = "FLOAT_16")]
    Float16,
    #[serde(rename = "BFLOAT_16")]
    BFloat16,
    #[serde(rename = "TFLOAT_32")]
    TFloat32,
    #[serde(rename = "NONE")]
    None,
    #[serde(rename = "FLOAT_8")]
    Float8,
    #[serde(rename = "INT_8")]
    Int8,
    #[serde(rename = "NFLOAT_4")]
    NFloat4,
}

impl DataType {
    pub fn to_flame_dtype(&self) -> flame_core::DType {
        match self {
            DataType::Float32 => flame_core::DType::F32,
            DataType::BFloat16 => flame_core::DType::BF16,
            DataType::Float16 => flame_core::DType::F16,
            _ => flame_core::DType::F32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Optimizer {
    #[serde(rename = "ADAMW")]
    #[default]
    AdamW,
    #[serde(rename = "ADAMW_8BIT")]
    AdamW8Bit,
    #[serde(rename = "ADAM")]
    Adam,
    #[serde(rename = "SGD")]
    Sgd,
    #[serde(rename = "ADAFACTOR")]
    Adafactor,
    #[serde(rename = "LION")]
    Lion,
    #[serde(rename = "ADOPT")]
    Adopt,
    #[serde(rename = "MUON")]
    Muon,
    #[serde(rename = "PRODIGY")]
    Prodigy,
    #[serde(rename = "CAME")]
    Came,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LrScheduler {
    #[serde(rename = "CONSTANT")]
    #[default]
    Constant,
    #[serde(rename = "LINEAR")]
    Linear,
    #[serde(rename = "COSINE")]
    Cosine,
    #[serde(rename = "COSINE_WITH_RESTARTS")]
    CosineWithRestarts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TimeUnit {
    #[serde(rename = "EPOCH")]
    Epoch,
    #[serde(rename = "STEP")]
    Step,
    #[serde(rename = "MINUTE")]
    Minute,
    #[serde(rename = "NEVER")]
    #[default]
    Never,
    #[serde(rename = "ALWAYS")]
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelFormat {
    #[serde(rename = "SAFETENSORS")]
    #[default]
    Safetensors,
    #[serde(rename = "DIFFUSERS")]
    Diffusers,
    #[serde(rename = "CKPT")]
    Ckpt,
    #[serde(rename = "INTERNAL")]
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PeftType {
    #[serde(rename = "LORA")]
    #[default]
    Lora,
    #[serde(rename = "LOHA")]
    Loha,
    #[serde(rename = "OFT_2")]
    Oft2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EmAMode {
    #[serde(rename = "OFF")]
    #[default]
    Off,
    #[serde(rename = "GPU")]
    Gpu,
    #[serde(rename = "CPU")]
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TimestepDistribution {
    #[serde(rename = "UNIFORM")]
    #[default]
    Uniform,
    #[serde(rename = "SIGMOID")]
    Sigmoid,
    #[serde(rename = "LOGIT_NORMAL")]
    LogitNormal,
    #[serde(rename = "HEAVY_TAIL")]
    HeavyTail,
    #[serde(rename = "COS_MAP")]
    CosMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LossWeight {
    #[serde(rename = "CONSTANT")]
    #[default]
    Constant,
    #[serde(rename = "P2")]
    P2,
    #[serde(rename = "MIN_SNR_GAMMA")]
    MinSnrGamma,
    #[serde(rename = "SIGMA")]
    Sigma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GradientCheckpointing {
    #[serde(rename = "ON")]
    #[default]
    On,
    #[serde(rename = "OFF")]
    Off,
}
