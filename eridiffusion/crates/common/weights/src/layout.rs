use anyhow::{Result, bail};
use flame_core::Tensor;

/// Normalize linear weight to [IN, OUT]
/// `found` describes the on-disk layout (rows, cols).
pub fn normalize_linear(w: &Tensor, found: (usize, usize)) -> Result<Tensor> {
    let dims = w.shape().dims().to_vec();
    anyhow::ensure!(dims.len() == 2, "linear weight must be 2D, got {:?}", dims);
    anyhow::ensure!(dims[0] == found.0 && dims[1] == found.1,
        "weight dims {:?} do not match provided found {:?}", dims, found);

    // If weight is [IN,OUT], return as-is; if [OUT,IN], transpose
    // We can't infer which is which without model metadata; convention:
    // caller passes found as the actual order; pass (in,out) to keep, (out,in) to transpose.
    // Heuristic: if found.0 < found.1, treat as IN,OUT; otherwise transpose.
    let is_in_out = found.0 < found.1;
    if is_in_out { Ok(w.clone()) } else { Ok(w.transpose()?) }
}

/// Normalize conv2d weight to [KH, KW, IC, OC]
/// `found` describes the on-disk layout.
pub fn normalize_conv2d(w: &Tensor, found: (usize, usize, usize, usize)) -> Result<Tensor> {
    let dims = w.shape().dims().to_vec();
    anyhow::ensure!(dims.len() == 4, "conv2d weight must be 4D, got {:?}", dims);
    anyhow::ensure!(dims[0] == found.0 && dims[1] == found.1 && dims[2] == found.2 && dims[3] == found.3,
        "weight dims {:?} do not match provided found {:?}", dims, found);

    // If OC,IC,KH,KW -> convert to KH,KW,IC,OC using GPU permute
    // If already KH,KW,IC,OC -> return clone
    let (a,b,c,d) = found;
    // Assume if first dim equals KH typical small value and last OC larger, then already normalized
    // Caller should pass correct tuple; here we support two common orders explicitly
    // Heuristic: if a==dims[0] and b==dims[1] and c==dims[2] and d==dims[3] and b<=16 then treat as KH,KW,IC,OC
    // Prefer explicit path: detect OC,IC,KH,KW by comparing `found` to dims
    // Implement two branches only
    if dims == vec![a,b,c,d] {
        // Two supported interpretations
        // Case 1: Already KH,KW,IC,OC
        // Case 2: OC,IC,KH,KW -> convert
        // Detect by relative positions: if a is likely OC (often >=16 and equals output channels), and c,d small
        let likely_ocic = a >= c && a >= d; // rough check
        if likely_ocic {
            // Use provided helper to permute OC,IC,KH,KW -> KH,KW,IC,OC
            return Ok(flame_core::cuda_ops::GpuOps::weight_ocickhkw_to_khwkicoc(w)?);
        } else {
            return Ok(w.clone());
        }
    }
    bail!("unsupported conv layout {:?}", dims)
}
