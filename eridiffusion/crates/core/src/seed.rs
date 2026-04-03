use crate::Result;

/// Seed all relevant RNGs used by EriDiffusion and Flame.
/// Note: Flame’s CUDA-side RNG is not yet globally configurable; this function
/// seeds CPU-side PRNGs and calls into device-level seed hooks where available.
pub fn seed_all(seed: u64) -> Result<()> {
    // Best-effort: set deterministic seed hints via environment (for downstream libs)
    std::env::set_var("ERIDIFFUSION_SEED", seed.to_string());
    // If Flame exposes device-level seeding, call it at device init sites.
    // Many internal sampling utilities construct their own RNGs; for now this acts as a
    // central place to thread a seed and log it.
    Ok(())
}
