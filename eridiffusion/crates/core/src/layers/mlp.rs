use flame_core::{Tensor, Result as FlameResult};

fn fused_enabled() -> bool {
    std::env::var("ERID_FUSE").ok().as_deref() == Some("1")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationKind { Gelu, Silu }

/// Fused MLP: (x @ w + b) -> act -> dropout(p) -> + residual(optional)
/// Shapes:
/// - x: [*, Din]
/// - w: [Din, Dout]
/// - b: [Dout]
pub fn mlp(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    act: ActivationKind,
    dropout_p: f32,
    training: bool,
    residual: Option<&Tensor>,
) -> FlameResult<Tensor> {
    if fused_enabled() { mlp_fused(x, w, b, act, dropout_p, training, residual) }
    else { mlp_ref(x, w, b, act, dropout_p, training, residual) }
}

fn mlp_ref(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    act: ActivationKind,
    dropout_p: f32,
    training: bool,
    residual: Option<&Tensor>,
) -> FlameResult<Tensor> {
    let mut y = x.matmul(w)?; // [*, Dout]
    if let Some(bias) = b { y = y.add(bias)?; }
    y = match act { ActivationKind::Gelu => y.gelu()?, ActivationKind::Silu => y.silu()?, };
    if training && dropout_p > 0.0 { y = apply_dropout(&y, dropout_p)?; }
    if let Some(r) = residual { y = y.add(r)?; }
    Ok(y)
}

fn mlp_fused(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    act: ActivationKind,
    dropout_p: f32,
    training: bool,
    residual: Option<&Tensor>,
) -> FlameResult<Tensor> {
    // Placeholder fused path delegates to ref path; hook for real kernel later.
    mlp_ref(x, w, b, act, dropout_p, training, residual)
}

fn apply_dropout(x: &Tensor, p: f32) -> FlameResult<Tensor> {
    // Simple host-generated mask per element; scale by 1/(1-p)
    let shape = x.shape().clone();
    let keep_prob = 1.0f32 - p;
    let scale = 1.0f32 / keep_prob.max(1e-6);
    // Generate mask on host deterministically per element
    let mut mask = Vec::with_capacity(shape.elem_count());
    for i in 0..shape.elem_count() {
        // LCG pseudo-random using index
        let r = ((1103515245u64.wrapping_mul(i as u64).wrapping_add(12345)) % 1_000_000) as f32 / 1_000_000.0;
        let m = if r < keep_prob { scale } else { 0.0 };
        mask.push(m);
    }
    let m = Tensor::from_vec(mask, shape, x.device().clone())?;
    x.mul(&m)
}
