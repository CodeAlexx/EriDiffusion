//! Generic in-training sampler hook.
//!
//! Different trainers/models have very different sampling code paths
//! (Z-Image euler vs Klein flow vs SDXL DDIM vs ERNIE w/ pooled), so this
//! module deliberately stays model-agnostic. It provides:
//!
//! 1. A [`SamplePromptSpec`] — the per-prompt knobs the trainer needs to
//!    pass back into its render closure.
//! 2. An [`InTrainingSampler`] — orchestrates "every N steps + sample at
//!    step 0", iterates prompts, builds output filenames, and logs errors
//!    without aborting the training run.
//! 3. The actual render is supplied by the caller as a closure
//!    `FnMut(&SamplePromptSpec, &Path) -> Result<()>` so each trainer keeps
//!    its existing `sampling::sample_image` (or equivalent) implementation.
//!
//! This lets us share the wiring code (config plumbing, file naming, EMA
//! swap orchestration) without coupling to any model architecture.
//!
//! ## Status
//!
//! **Currently no in-tree trainer uses [`InTrainingSampler`].** Existing
//! trainers (Z-Image, Klein, SDXL, ERNIE) have their own open-coded sample
//! loops with model-specific scheduler logic. This helper is preserved
//! (and `#[deprecated]`-marked) because:
//!
//! 1. The orchestration logic (sample-at-step-0 + every-N + JSONL log) is
//!    the same across all trainers and is the right shape for any future
//!    trainer that doesn't need bespoke scheduling.
//! 2. Removing it would lose the documented JSONL contract that
//!    SerenityBoard relies on.
//!
//! New trainers wanting this pattern can opt-in by suppressing the
//! deprecation warning. If the helper still has zero callers in 6 months,
//! delete it.
//!
//! ## Metrics
//!
//! Render durations are appended as one JSON line per (step, prompt) pair
//! to a sibling `samples.jsonl` file when [`InTrainingSampler::with_metrics_path`]
//! is called. SerenityBoard parses this format alongside `metrics.jsonl`.
//! NOT tensorboard — project rule forbids tensorboard.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct SamplePromptSpec {
    /// Free-form label used in log lines and sample filenames. May be empty.
    pub label: String,
    /// Per-prompt seed. Trainers usually want a stable seed across step
    /// boundaries so the only changing variable is the model state.
    pub seed: u64,
    /// Output image height in pixels.
    pub height: usize,
    /// Output image width in pixels.
    pub width: usize,
    /// Number of denoise steps.
    pub steps: usize,
    /// Classifier-free guidance scale (0 = no CFG).
    pub cfg: f32,
    /// Discrete-flow / DDIM shift, where applicable. Trainers without a
    /// shift knob can ignore this field.
    pub shift: f32,
}

impl SamplePromptSpec {
    pub fn new(label: impl Into<String>, height: usize, width: usize) -> Self {
        Self {
            label: label.into(),
            seed: 42,
            height,
            width,
            steps: 8,
            cfg: 0.0,
            shift: 1.0,
        }
    }
}

#[deprecated(
    since = "0.0.0",
    note = "no in-tree trainer uses InTrainingSampler; trainers have open-coded sample \
            loops with model-specific scheduler code. Helper is preserved for future \
            trainers that don't need bespoke scheduling. Suppress this warning if you're \
            wiring a new trainer to it."
)]
pub struct InTrainingSampler {
    every: usize,
    prompts: Vec<SamplePromptSpec>,
    output_dir: PathBuf,
    sample_at_start: bool,
    /// Optional sibling JSONL log path for SerenityBoard ingest.
    metrics_path: Option<PathBuf>,
}

#[allow(deprecated)]
impl InTrainingSampler {
    /// `every == 0` disables sampling entirely (use [`InTrainingSampler::is_enabled`]
    /// to check before constructing rich auxiliary data).
    pub fn new(every: usize, prompts: Vec<SamplePromptSpec>, output_dir: PathBuf) -> Self {
        Self {
            every,
            prompts,
            output_dir,
            sample_at_start: true,
            metrics_path: None,
        }
    }

    pub fn with_sample_at_start(mut self, on: bool) -> Self {
        self.sample_at_start = on;
        self
    }

    /// Append per-render JSONL records to `path` (one line per step+prompt).
    /// Format matches the project's existing JSONL convention used by
    /// SerenityBoard. NOT tensorboard.
    pub fn with_metrics_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.metrics_path = Some(path.into());
        self
    }

    pub fn is_enabled(&self) -> bool {
        self.every > 0 && !self.prompts.is_empty()
    }

    pub fn every(&self) -> usize {
        self.every
    }

    pub fn prompts(&self) -> &[SamplePromptSpec] {
        &self.prompts
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// True at step 0 (when sample_at_start is on) and every Nth step thereafter.
    pub fn should_sample(&self, step: usize) -> bool {
        if !self.is_enabled() {
            return false;
        }
        if step == 0 {
            return self.sample_at_start;
        }
        step % self.every == 0
    }

    /// Convenience: build the standard `step{N:06}_p{idx:02}.png` path.
    pub fn output_path(&self, step: usize, prompt_idx: usize) -> PathBuf {
        self.output_dir
            .join(format!("step_{:06}_p{:02}.png", step, prompt_idx))
    }

    /// Iterate the prompts and call `render_fn` for each, with a per-prompt
    /// output path. Errors are logged and skipped — sampling failures must
    /// not abort the training run.
    pub fn sample_with<F>(&self, step: usize, mut render_fn: F) -> std::io::Result<()>
    where
        F: FnMut(&SamplePromptSpec, &Path) -> Result<(), Box<dyn std::error::Error>>,
    {
        if !self.should_sample(step) {
            return Ok(());
        }
        std::fs::create_dir_all(&self.output_dir)?;
        for (i, prompt) in self.prompts.iter().enumerate() {
            let out_path = self.output_path(step, i);
            log::info!(
                "sampling step={} prompt={}/{} ({:?}) → {}",
                step,
                i + 1,
                self.prompts.len(),
                prompt.label,
                out_path.display(),
            );
            let started = Instant::now();
            let outcome = render_fn(prompt, &out_path);
            let elapsed = started.elapsed().as_secs_f32();
            match &outcome {
                Ok(()) => log::info!("  wrote {} ({:.1}s)", out_path.display(), elapsed),
                Err(e) => log::warn!(
                    "  sampling FAILED at step={} prompt={}: {}",
                    step,
                    i,
                    e,
                ),
            }
            // Optional sibling JSONL log for SerenityBoard. One line per
            // (step, prompt). Failures here are logged but never abort the
            // training run.
            if let Some(ref mpath) = self.metrics_path {
                let ok = outcome.is_ok();
                let line = format!(
                    "{{\"step\":{},\"prompt_idx\":{},\"label\":\"{}\",\"width\":{},\"height\":{},\"steps\":{},\"cfg\":{:.3},\"shift\":{:.3},\"seed\":{},\"render_seconds\":{:.3},\"ok\":{},\"path\":\"{}\"}}\n",
                    step,
                    i,
                    json_escape(&prompt.label),
                    prompt.width,
                    prompt.height,
                    prompt.steps,
                    prompt.cfg,
                    prompt.shift,
                    prompt.seed,
                    elapsed,
                    ok,
                    json_escape(&out_path.display().to_string()),
                );
                if let Err(e) = append_line(mpath, &line) {
                    log::warn!("samples.jsonl write failed: {}", e);
                }
            }
        }
        Ok(())
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(line.as_bytes())
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn should_sample_step_zero_and_every_n() {
        let s = InTrainingSampler::new(
            5,
            vec![SamplePromptSpec::new("a", 64, 64)],
            PathBuf::from("/tmp"),
        );
        assert!(s.should_sample(0));
        assert!(!s.should_sample(1));
        assert!(!s.should_sample(4));
        assert!(s.should_sample(5));
        assert!(s.should_sample(10));
    }

    #[test]
    fn disabled_when_every_zero_or_no_prompts() {
        let no_prompts = InTrainingSampler::new(5, vec![], PathBuf::from("/tmp"));
        assert!(!no_prompts.is_enabled());
        assert!(!no_prompts.should_sample(0));

        let disabled = InTrainingSampler::new(
            0,
            vec![SamplePromptSpec::new("a", 64, 64)],
            PathBuf::from("/tmp"),
        );
        assert!(!disabled.is_enabled());
        assert!(!disabled.should_sample(0));
    }

    #[test]
    fn sample_at_start_can_be_turned_off() {
        let s = InTrainingSampler::new(
            3,
            vec![SamplePromptSpec::new("a", 64, 64)],
            PathBuf::from("/tmp"),
        )
        .with_sample_at_start(false);
        assert!(!s.should_sample(0));
        assert!(s.should_sample(3));
    }

    /// Brief-spec checks: explicit boundary cases at every / every+1 / 2*every.
    #[test]
    fn should_sample_boundary_cases() {
        let every = 7;
        let s = InTrainingSampler::new(
            every,
            vec![SamplePromptSpec::new("p", 64, 64)],
            PathBuf::from("/tmp"),
        );
        assert!(s.should_sample(0), "step 0 with sample_at_start=true");
        assert!(s.should_sample(every), "step==every");
        assert!(!s.should_sample(every + 1), "step==every+1");
        assert!(s.should_sample(2 * every), "step==2*every");
    }

    /// `output_path` produces the documented `step_NNNNNN_pNN.png` form.
    #[test]
    fn output_path_format() {
        let s = InTrainingSampler::new(
            10,
            vec![
                SamplePromptSpec::new("a", 64, 64),
                SamplePromptSpec::new("b", 64, 64),
            ],
            PathBuf::from("/tmp/out"),
        );
        let p = s.output_path(123, 7);
        assert_eq!(p, PathBuf::from("/tmp/out/step_000123_p07.png"));
    }
}
