use anyhow::{anyhow, Result};
use flame_core::Tensor;

#[cfg(feature = "cond_time_ids_mlp")]
use super::make_conditioning::make_conditioning_with_time_ids;
use super::make_conditioning::{make_conditioning_with_sinusoidal, SdxlCond};

/// Builder-style arguments for conditioning helpers.
#[derive(Clone)]
pub struct CondArgs<'a> {
    pub label_emb_in_1280: &'a Tensor,
    pub timesteps_any: Option<&'a Tensor>,
    pub min_t: f32,
    pub max_t: f32,
    pub max_period: f32,
    pub pooled_1280: Option<&'a Tensor>,
    pub time_ids_6: Option<&'a Tensor>,
    pub time_ids_mlp_fwd: Option<&'a dyn Fn(&Tensor) -> Result<Tensor>>,
}

impl<'a> CondArgs<'a> {
    pub fn sinusoidal(
        label_emb_in_1280: &'a Tensor,
        timesteps_any: &'a Tensor,
        min_t: f32,
        max_t: f32,
        max_period: f32,
    ) -> Self {
        Self {
            label_emb_in_1280,
            timesteps_any: Some(timesteps_any),
            min_t,
            max_t,
            max_period,
            pooled_1280: None,
            time_ids_6: None,
            time_ids_mlp_fwd: None,
        }
    }

    #[cfg(feature = "cond_time_ids_mlp")]
    pub fn time_ids(
        label_emb_in_1280: &'a Tensor,
        pooled_1280: &'a Tensor,
        time_ids_6: &'a Tensor,
        time_ids_mlp_fwd: &'a dyn Fn(&Tensor) -> Result<Tensor>,
        min_t: f32,
        max_t: f32,
        max_period: f32,
        timesteps_any: Option<&'a Tensor>,
    ) -> Self {
        Self {
            label_emb_in_1280,
            timesteps_any,
            min_t,
            max_t,
            max_period,
            pooled_1280: Some(pooled_1280),
            time_ids_6: Some(time_ids_6),
            time_ids_mlp_fwd: Some(time_ids_mlp_fwd),
        }
    }
}

/// Dispatch to the active conditioning implementation at compile time.
pub fn make_conditioning(args: CondArgs) -> Result<SdxlCond> {
    #[cfg(feature = "cond_time_ids_mlp")]
    {
        let pooled =
            args.pooled_1280.ok_or_else(|| anyhow!("make_conditioning: pooled_1280 missing"))?;
        let time_ids =
            args.time_ids_6.ok_or_else(|| anyhow!("make_conditioning: time_ids_6 missing"))?;
        let fwd = args
            .time_ids_mlp_fwd
            .ok_or_else(|| anyhow!("make_conditioning: time_ids_mlp_fwd missing"))?;
        return make_conditioning_with_time_ids(args.label_emb_in_1280, pooled, time_ids, fwd);
    }

    #[cfg(all(not(feature = "cond_time_ids_mlp"), feature = "cond_sinusoidal"))]
    {
        let ts = args
            .timesteps_any
            .ok_or_else(|| anyhow!("make_conditioning: timesteps_any missing"))?;
        return make_conditioning_with_sinusoidal(
            args.label_emb_in_1280,
            ts,
            args.min_t,
            args.max_t,
            args.max_period,
        );
    }

    #[cfg(all(not(feature = "cond_time_ids_mlp"), not(feature = "cond_sinusoidal")))]
    {
        compile_error!("Enable either `cond_sinusoidal` (default) or `cond_time_ids_mlp`");
    }
}
