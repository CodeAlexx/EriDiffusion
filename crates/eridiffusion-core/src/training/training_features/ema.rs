//! EMA shadow for trainer parameters.
//!
//! Thin wrapper over [`crate::ema::ParameterEma`] that adds the
//! `apply_to(...)` / `restore_from_live(...)` swap protocol so a trainer can
//! temporarily push EMA weights into the live params for sampling and then
//! restore the training weights afterwards.
//!
//! ```text
//! let mut ema = EmaShadow::new(&params, 0.9999)?;
//!
//! // ... in training loop after optimizer.step() ...
//! ema.update(&params)?;
//!
//! // ... before sampling ...
//! let backup = ema.snapshot_live(&params)?;   // take snapshot
//! ema.apply_to(&params)?;                      // copy shadow → params
//! /* ... sample ... */
//! ema.restore_from_live(&params, &backup)?;    // copy backup → params
//! ```
//!
//! If `decay <= 0`, [`EmaShadow::new`] returns `Ok(None)` via the
//! `maybe_new` constructor — trainers should use `maybe_new` to opt in.

use crate::training::ema::ParameterEma;
use flame_core::{parameter::Parameter, Result, Tensor, TensorId};
use std::collections::HashMap;

pub struct EmaShadow {
    inner: ParameterEma,
    /// `param.id() → index in the slice passed at construction time` so
    /// `apply_to` can match shadows back to params even if the order changes
    /// (it shouldn't, but cheap insurance).
    index: HashMap<TensorId, usize>,
}

impl EmaShadow {
    pub fn new(params: &[Parameter], decay: f32) -> Result<Self> {
        let inner = ParameterEma::new(params, decay)?;
        let mut index = HashMap::with_capacity(params.len());
        for (i, p) in params.iter().enumerate() {
            index.insert(p.id(), i);
        }
        Ok(Self { inner, index })
    }

    /// Create an [`EmaShadow`] only when `decay > 0`. Returns `Ok(None)`
    /// otherwise — the standard "ema disabled" path.
    pub fn maybe_new(params: &[Parameter], decay: f32) -> Result<Option<Self>> {
        if decay > 0.0 {
            Ok(Some(Self::new(params, decay)?))
        } else {
            Ok(None)
        }
    }

    pub fn decay(&self) -> f32 {
        self.inner.decay()
    }

    /// Apply one EMA update step. Call after `optimizer.step`.
    pub fn update(&mut self, params: &[Parameter]) -> Result<()> {
        self.inner.update(params)
    }

    /// Take a snapshot of the live param tensors so they can be restored
    /// after a sampling-time `apply_to`.
    pub fn snapshot_live(&self, params: &[Parameter]) -> Result<HashMap<TensorId, Tensor>> {
        let _no_grad = flame_core::autograd::AutogradContext::no_grad();
        let mut out = HashMap::with_capacity(params.len());
        for p in params {
            out.insert(p.id(), p.tensor()?.detach()?);
        }
        Ok(out)
    }

    /// Copy shadow tensors → live params, casting to each param's dtype.
    /// Use with [`EmaShadow::snapshot_live`] for the sample-then-restore cycle.
    pub fn apply_to(&self, params: &[Parameter]) -> Result<()> {
        let _no_grad = flame_core::autograd::AutogradContext::no_grad();
        for p in params {
            let idx = self.index.get(&p.id()).copied().ok_or_else(|| {
                flame_core::Error::Training(format!(
                    "EmaShadow::apply_to: param {:?} not tracked",
                    p.id()
                ))
            })?;
            let shadow = self.inner.shadow(idx);
            let target_dtype = p.dtype()?;
            let cast = if shadow.dtype() == target_dtype {
                shadow.clone()
            } else {
                shadow.to_dtype(target_dtype)?
            };
            p.set_data(cast.detach()?)?;
        }
        Ok(())
    }

    /// Inverse of [`EmaShadow::apply_to`] — restore the snapshot taken before
    /// pushing EMA into the params.
    pub fn restore_from_live(
        &self,
        params: &[Parameter],
        backup: &HashMap<TensorId, Tensor>,
    ) -> Result<()> {
        let _no_grad = flame_core::autograd::AutogradContext::no_grad();
        for p in params {
            let saved = backup.get(&p.id()).ok_or_else(|| {
                flame_core::Error::Training(format!(
                    "EmaShadow::restore_from_live: no backup for param {:?}",
                    p.id()
                ))
            })?;
            // `set_data` keeps the param's dtype invariant — but the snapshot
            // was taken from the param itself, so dtype already matches.
            let target_dtype = p.dtype()?;
            let cast = if saved.dtype() == target_dtype {
                saved.clone()
            } else {
                saved.to_dtype(target_dtype)?
            };
            p.set_data(cast.detach()?)?;
        }
        Ok(())
    }

    /// Number of tracked parameters.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Indexed read of one shadow tensor (F32). Useful for trainers that
    /// want to write EMA copies as separate safetensors checkpoints.
    pub fn shadow(&self, index: usize) -> &Tensor {
        self.inner.shadow(index)
    }

    pub fn shadow_all(&self) -> &[Tensor] {
        self.inner.shadow_all()
    }

    pub fn inner(&self) -> &ParameterEma {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;
    use flame_core::{Shape, Tensor};

    fn cuda_or_skip() -> Option<std::sync::Arc<CudaDevice>> {
        match CudaDevice::new(0) {
            Ok(d) => Some(d),
            Err(_) => {
                eprintln!("skipping: no CUDA device");
                None
            }
        }
    }

    fn make_param(
        device: std::sync::Arc<CudaDevice>,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) -> Parameter {
        let t = Tensor::from_vec(data, Shape::from_dims(&shape), device).unwrap();
        Parameter::new(t.requires_grad_(true))
    }

    fn vec_close(a: &[f32], b: &[f32], eps: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
    }

    /// At construction time, shadow = live param values exactly.
    #[test]
    fn new_initializes_shadow_to_live() {
        let Some(device) = cuda_or_skip() else { return };
        let init = vec![1.0_f32, -2.0, 3.5, 0.0];
        let p = make_param(device.clone(), init.clone(), vec![4]);
        let ema = EmaShadow::new(&[p.clone()], 0.999).unwrap();
        let shadow = ema.shadow(0).to_vec().unwrap();
        assert!(vec_close(&shadow, &init, 1e-6), "shadow should match live at init");
    }

    /// `decay = 0.0` is the disabled-EMA path — `update()` returns early
    /// without modifying the shadow. Documented behavior in `ParameterEma`.
    #[test]
    fn update_with_decay_zero_is_noop() {
        let Some(device) = cuda_or_skip() else { return };
        let init = vec![1.0_f32, 2.0];
        let p = make_param(device.clone(), init.clone(), vec![2]);
        let mut ema = EmaShadow::new(&[p.clone()], 0.0).unwrap();
        // Mutate live param.
        let new_val =
            Tensor::from_vec(vec![10.0, 20.0], Shape::from_dims(&[2]), device.clone()).unwrap();
        p.set_data(new_val).unwrap();
        ema.update(&[p.clone()]).unwrap();
        let shadow = ema.shadow(0).to_vec().unwrap();
        assert!(
            vec_close(&shadow, &init, 1e-6),
            "decay=0 must be a no-op; shadow={:?}, init={:?}",
            shadow,
            init
        );
    }

    /// `decay = 1.0` keeps the shadow unchanged regardless of param updates.
    #[test]
    fn update_with_decay_one_preserves_shadow() {
        let Some(device) = cuda_or_skip() else { return };
        let init = vec![1.0_f32, 2.0];
        let p = make_param(device.clone(), init.clone(), vec![2]);
        let mut ema = EmaShadow::new(&[p.clone()], 1.0).unwrap();
        let new_val =
            Tensor::from_vec(vec![100.0, -50.0], Shape::from_dims(&[2]), device.clone()).unwrap();
        p.set_data(new_val).unwrap();
        ema.update(&[p.clone()]).unwrap();
        let shadow = ema.shadow(0).to_vec().unwrap();
        assert!(
            vec_close(&shadow, &init, 1e-6),
            "decay=1 must preserve shadow; got {:?}",
            shadow
        );
    }

    /// `decay = 0.5` (well-defined non-trivial blend): after one step,
    ///   shadow' = 0.5 * shadow + 0.5 * live.
    /// Hand-checked with init=[1,2], live=[5,6] → shadow'=[3,4].
    #[test]
    fn update_blends_with_decay_half() {
        let Some(device) = cuda_or_skip() else { return };
        let init = vec![1.0_f32, 2.0];
        let p = make_param(device.clone(), init.clone(), vec![2]);
        let mut ema = EmaShadow::new(&[p.clone()], 0.5).unwrap();
        let new_val =
            Tensor::from_vec(vec![5.0, 6.0], Shape::from_dims(&[2]), device.clone()).unwrap();
        p.set_data(new_val).unwrap();
        ema.update(&[p.clone()]).unwrap();
        let shadow = ema.shadow(0).to_vec().unwrap();
        let expected = vec![3.0_f32, 4.0]; // 0.5 * 1 + 0.5 * 5; 0.5 * 2 + 0.5 * 6
        assert!(
            vec_close(&shadow, &expected, 1e-6),
            "decay=0.5 should blend; got {:?}, expected {:?}",
            shadow,
            expected
        );
    }

    /// Hand-checked decay=0.999 single step.
    /// init=[1.0], live=[2.0] → shadow' = 0.999 * 1.0 + 0.001 * 2.0 = 1.001
    #[test]
    fn update_with_decay_999_hand_checked() {
        let Some(device) = cuda_or_skip() else { return };
        let p = make_param(device.clone(), vec![1.0], vec![1]);
        let mut ema = EmaShadow::new(&[p.clone()], 0.999).unwrap();
        let new_val =
            Tensor::from_vec(vec![2.0], Shape::from_dims(&[1]), device.clone()).unwrap();
        p.set_data(new_val).unwrap();
        ema.update(&[p.clone()]).unwrap();
        let shadow = ema.shadow(0).to_vec().unwrap();
        assert!(
            (shadow[0] - 1.001).abs() < 1e-5,
            "decay=0.999: expected 1.001, got {}",
            shadow[0]
        );
    }

    /// snapshot → apply_to → restore_from_live round-trips the live params
    /// back to their original values.
    #[test]
    fn snapshot_apply_restore_roundtrip() {
        let Some(device) = cuda_or_skip() else { return };
        let init = vec![1.0_f32, -2.0, 3.5, 0.0];
        let p = make_param(device.clone(), init.clone(), vec![4]);
        let mut ema = EmaShadow::new(&[p.clone()], 0.5).unwrap();

        // Push the live params somewhere different, then update EMA so
        // shadow ≠ live.
        let new_val = Tensor::from_vec(
            vec![10.0, 20.0, 30.0, 40.0],
            Shape::from_dims(&[4]),
            device.clone(),
        )
        .unwrap();
        p.set_data(new_val).unwrap();
        ema.update(&[p.clone()]).unwrap();

        // Take snapshot of CURRENT live, apply shadow, then restore.
        let live_before: Vec<f32> = p.tensor().unwrap().to_vec().unwrap();
        let backup = ema.snapshot_live(&[p.clone()]).unwrap();
        ema.apply_to(&[p.clone()]).unwrap();
        // Live should now equal the shadow.
        let live_during = p.tensor().unwrap().to_vec().unwrap();
        let shadow = ema.shadow(0).to_vec().unwrap();
        assert!(
            vec_close(&live_during, &shadow, 1e-6),
            "apply_to should overwrite live with shadow",
        );
        // Restore.
        ema.restore_from_live(&[p.clone()], &backup).unwrap();
        let live_after = p.tensor().unwrap().to_vec().unwrap();
        assert!(
            vec_close(&live_after, &live_before, 1e-6),
            "round-trip must restore exactly: before={:?}, after={:?}",
            live_before,
            live_after,
        );
    }

    /// `apply_to` actually overwrites live params with shadow values.
    /// (Asserted independently from the round-trip test above.)
    #[test]
    fn apply_to_overwrites_live_with_shadow() {
        let Some(device) = cuda_or_skip() else { return };
        let init = vec![5.0_f32, 7.0];
        let p = make_param(device.clone(), init.clone(), vec![2]);
        let ema = EmaShadow::new(&[p.clone()], 0.999).unwrap();
        // Mutate live to something different from shadow.
        let new_val =
            Tensor::from_vec(vec![100.0, 200.0], Shape::from_dims(&[2]), device.clone()).unwrap();
        p.set_data(new_val).unwrap();
        // Apply shadow back to live.
        ema.apply_to(&[p.clone()]).unwrap();
        let live = p.tensor().unwrap().to_vec().unwrap();
        assert!(
            vec_close(&live, &init, 1e-6),
            "apply_to should overwrite live with shadow values: got {:?}, expected {:?}",
            live,
            init
        );
    }

    /// `maybe_new` with decay <= 0 returns None.
    #[test]
    fn maybe_new_disables_when_decay_nonpositive() {
        let Some(device) = cuda_or_skip() else { return };
        let p = make_param(device.clone(), vec![1.0], vec![1]);
        assert!(EmaShadow::maybe_new(&[p.clone()], 0.0).unwrap().is_none());
        assert!(EmaShadow::maybe_new(&[p.clone()], -0.1).unwrap().is_none());
        assert!(EmaShadow::maybe_new(&[p], 0.999).unwrap().is_some());
    }
}
