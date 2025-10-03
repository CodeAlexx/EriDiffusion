use StageName::*;

use super::stages::StageName;

pub const SDXL_RECIPE: &[StageName] = &[
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
];
pub const SD35_RECIPE: &[StageName] = &[
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
];
pub const FLUX_RECIPE: &[StageName] = &[
    Seed,
    FetchBatch,
    TextEncode,
    NoiseSchedule,
    Forward,
    Loss,
    Backward,
    OptimStep,
    ZeroGrad,
    EmaUpdate,
    Checkpoint,
];
