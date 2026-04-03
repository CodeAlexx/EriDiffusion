use flame_core::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageName {
    Seed,
    FetchBatch,
    TextEncode,
    VaeEncode,
    NoiseSchedule,
    Forward,
    Loss,
    Backward,
    OptimStep,
    ZeroGrad,
    EmaUpdate,
    Checkpoint,
    EvalStep,
}

// Minimal placeholders to avoid pulling heavy types here.
pub struct Batch; // routed from existing dataloader in adapters
pub struct TextEncodings; // routed from existing encoders in adapters

pub struct Ctx {
    pub step: u64,
    pub rng_seed: u64,
    pub batch: Option<Batch>,
    pub text: Option<TextEncodings>,
    pub latents: Option<Tensor>,
    pub noise: Option<Tensor>,
    pub timestep: Option<Tensor>,
    pub pred: Option<Tensor>,
    pub loss: Option<Tensor>,
}
