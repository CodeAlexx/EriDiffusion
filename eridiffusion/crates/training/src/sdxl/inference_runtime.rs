use std::sync::Arc;

use anyhow::{anyhow, Result};
use flame_core::Tensor;

use super::registry::{ConditioningBundle, SdxlLayerRegistry, StageTransition};
use super::runtime::{with_attn_chunks, with_kernel_telemetry, AttnChunkConfig};

/// Dedicated inference runtime that pre-materialises SDXL blocks and executes
/// them in a straight-line pass without the training registry bookkeeping.
pub struct SdxlInferenceRuntime {
    registry: Arc<SdxlLayerRegistry>,
    transitions: Vec<Option<StageTransition>>,
    attn_config: AttnChunkConfig,
    kernel_telemetry: bool,
}

impl SdxlInferenceRuntime {
    pub fn new(
        registry: Arc<SdxlLayerRegistry>,
        attn_chunk: Option<usize>,
        kv_chunk: Option<usize>,
        kernel_telemetry: bool,
    ) -> Result<Self> {
        let mut transitions = Vec::with_capacity(registry.block_count());
        for idx in 0..registry.block_count() {
            transitions.push(registry.transition_for(idx).cloned());
        }

        if !transitions.is_empty() {
            // The inference pipeline executes the stem through `input_blocks.4.0`
            // ahead of time, so the registry's first down transition would repeat
            // that work and introduce channel mismatches. Drop it here and rely on
            // the prefilled skip provided at call time.
            transitions[0] = None;
        }

        let attn_config =
            AttnChunkConfig { q_chunk: attn_chunk.unwrap_or(0), kv_chunk: kv_chunk.unwrap_or(0) };

        Ok(Self { registry, transitions, attn_config, kernel_telemetry })
    }

    pub fn forward(
        &self,
        sample: Tensor,
        ctx: &Tensor,
        cond: &ConditioningBundle,
        prefilled_skips: &[Tensor],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let attn_cfg = self.attn_config;
        with_kernel_telemetry(self.kernel_telemetry, || {
            with_attn_chunks(attn_cfg, move || {
                let mut current = sample;
                let mut skip_stack: Vec<Tensor> = Vec::with_capacity(prefilled_skips.len() + 8);
                for skip in prefilled_skips {
                    skip_stack.push(skip.clone_result()?);
                }
                let stage_debug = std::env::var_os("SDXL_STAGE_DEBUG").is_some();
                let mut cat_skips: Vec<Tensor> = Vec::new();

                for idx in 0..self.registry.block_count() {
                    if let Some(trans) = &self.transitions[idx] {
                        if stage_debug {
                            eprintln!(
                                "[inference-stage] entering idx={idx} dims={:?}",
                                current.shape().dims()
                            );
                        }
                        let owned_skip = if trans.consumes_skip() {
                            Some(skip_stack.pop().ok_or_else(|| {
                                anyhow!("skip stack underflow at transition index {idx}")
                            })?)
                        } else {
                            None
                        };
                        let (next, maybe_skip) = trans.apply(&current, owned_skip.as_ref())?;
                        if let Some(skip) = maybe_skip {
                            if trans.produces_skip() {
                                if let Some(aux) = trans.take_down_aux() {
                                    if stage_debug {
                                        eprintln!(
                                            "[inference-stage] captured down aux {:?}",
                                            aux.shape().dims()
                                        );
                                    }
                                    // auxiliary tensor cleared from transition; not retained
                                }
                                skip_stack.push(skip);
                            } else {
                                if stage_debug {
                                    eprintln!(
                                        "[inference-stage] captured cat skip {:?}",
                                        skip.shape().dims()
                                    );
                                }
                                cat_skips.push(skip);
                            }
                        }
                        current = next;
                        if stage_debug {
                            eprintln!(
                                "[inference-stage] after transition idx={} dims={:?}",
                                idx,
                                current.shape().dims()
                            );
                        }
                    }

                    let block = self.registry.load_block_ephemeral(idx)?;
                    current = block.forward_inference(
                        &current,
                        ctx,
                        &cond.driver_1280,
                        Some(&cond.time_proj_1536),
                    )?;
                }

                if stage_debug {
                    eprintln!(
                        "[inference-stage] decoder skip candidates={} residual skip_stack={}",
                        cat_skips.len(),
                        skip_stack.len()
                    );
                }
                let collected = cat_skips;
                Ok((current, collected))
            })
        })
    }
}
