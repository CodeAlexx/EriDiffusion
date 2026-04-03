use anyhow::Result;
use hashbrown::HashMap;
use flame_core::{Tensor, Shape, DType, Device};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantKind { None, Q8, NF4 }

#[derive(Debug, Clone)]
pub struct QWeight {
    pub qkind: QuantKind,
    pub qdata: Tensor,
    pub scales: Tensor,
    pub zeros: Option<Tensor>,
    pub shape: (usize, usize), // [in, out]
}

pub fn quantize_weight(w_f32: &Tensor, kind: QuantKind) -> Result<QWeight> {
    // Placeholder: keep data as F32 tensor in qdata; real path packs ints
    let qdata = w_f32.clone();
    let scales = Tensor::zeros_dtype(Shape::from(vec![1]), DType::F32, w_f32.device().clone())?;
    Ok(QWeight { qkind: kind, qdata, scales, zeros: None, shape: (w_f32.shape().dims()[0], w_f32.shape().dims()[1]) })
}

pub fn dequantize_to_bf16(qw: &QWeight) -> Result<Tensor> {
    // Placeholder: return BF16 cast of stored tensor
    let t = qw.qdata.to_bf16()?;
    Ok(t)
}

pub struct ExpertBank {
    pub kind: QuantKind,
    pub experts: HashMap<usize, QWeight>,
}

impl ExpertBank {
    pub fn new(kind: QuantKind) -> Self { Self { kind, experts: HashMap::new() } }
    pub fn insert(&mut self, id: usize, qw: QWeight) { self.experts.insert(id, qw); }
    pub fn fetch_topk_bf16(&self, ids: &[usize]) -> Result<Vec<(usize, Tensor)>> {
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids { if let Some(qw) = self.experts.get(&id) { out.push((id, dequantize_to_bf16(qw)?)); } }
        Ok(out)
    }
}

/// Select the first k expert ids as a simple global-topk stub.
pub fn select_topk_global(total: usize, k: usize) -> Vec<usize> {
    (0..k.min(total)).collect()
}
