use eridiffusion_core::{Error, Result};
use flame_core::{DType, Device as FlameDevice, Parameter, Shape, Tensor};
use std::collections::HashMap;

#[derive(Clone)]
pub struct LoRALinear {
    pub a: Parameter,
    pub b: Parameter,
    pub rank: usize,
    pub alpha: f32,
}

impl LoRALinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        dev: FlameDevice,
    ) -> Result<Self> {
        // Standard LoRA init: A small random, B zeros → initial delta=0 but non-zero grad on B
        let a_t = Tensor::randn(
            Shape::from_dims(&[in_dim, rank]),
            0.0,
            (1.0f32 / rank.max(1) as f32).sqrt(),
            dev.cuda_device_arc(),
        )
        .map_err(eridiffusion_core::Error::from)?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
        let b_t = Tensor::zeros_dtype(
            Shape::from_dims(&[rank, out_dim]),
            DType::BF16,
            dev.cuda_device_arc(),
        )
        .map_err(eridiffusion_core::Error::from)?
        .requires_grad_(true);
        Ok(Self { a: Parameter::new(a_t), b: Parameter::new(b_t), rank, alpha })
    }
    pub fn forward_delta(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.a.tensor().map_err(eridiffusion_core::Error::from)?;
        let b = self.b.tensor().map_err(eridiffusion_core::Error::from)?;
        // Ensure dtype match via autograd-aware Cast so grads flow across dtype boundary
        let a = a.to_dtype(x.dtype())?;
        let b = b.to_dtype(x.dtype())?;
        let h = x.matmul(&a)?;
        let y = h.matmul(&b)?;
        let out = y.affine((self.alpha / self.rank as f32) as f32, 0.0f32)?;
        Ok(out)
    }
    pub fn params(&self) -> Vec<Parameter> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub struct LoraHandles {
    pub per_block: Vec<Vec<LoRALinear>>,
}
impl LoraHandles {
    pub fn new(
        n_blocks: usize,
        _in_dim: usize,
        _out_dim: usize,
        rank: usize,
        alpha: f32,
        dev: FlameDevice,
    ) -> Result<Self> {
        let mut per = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            per.push(vec![
                // q, k, v, o all map 3072 -> 3072
                LoRALinear::new(3072, 3072, rank, alpha, dev.clone())?, // q
                LoRALinear::new(3072, 3072, rank, alpha, dev.clone())?, // k
                LoRALinear::new(3072, 3072, rank, alpha, dev.clone())?, // v
                LoRALinear::new(3072, 3072, rank, alpha, dev.clone())?, // o
                // MLP: fc1 is 3072 -> 12288, fc2 is 12288 -> 3072
                LoRALinear::new(3072, 12288, rank, alpha, dev.clone())?, // fc1
                LoRALinear::new(12288, 3072, rank, alpha, dev.clone())?, // fc2
            ]);
        }
        Ok(Self { per_block: per })
    }
    pub fn for_block(&self, i: usize) -> &Vec<LoRALinear> {
        &self.per_block[i]
    }
    pub fn all_params(&self) -> Vec<Parameter> {
        self.per_block.iter().flat_map(|g| g.iter().flat_map(|l| l.params())).collect()
    }

    pub fn blocks_len(&self) -> usize {
        self.per_block.len()
    }

    pub fn set_block_trainable(&mut self, idx: usize, trainable: bool) {
        if idx >= self.per_block.len() {
            return;
        }
        for ll in &mut self.per_block[idx] {
            // Mutate flags on the actual Parameter handles
            ll.a.set_requires_grad(trainable);
            ll.b.set_requires_grad(trainable);
        }
    }

    pub fn set_all_trainable(&mut self, trainable: bool) {
        for i in 0..self.per_block.len() {
            self.set_block_trainable(i, trainable);
        }
    }

    /// Named Parameter view for EMA init/update (owns Parameter handles into same storage)
    pub fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut out = HashMap::new();
        for (i, group) in self.per_block.iter().enumerate() {
            let q = &group[0];
            let k = &group[1];
            let v = &group[2];
            let o = &group[3];
            let fc1 = &group[4];
            let fc2 = &group[5];
            out.insert(format!("block{}.attn.q.lora.A", i), q.a.clone());
            out.insert(format!("block{}.attn.q.lora.B", i), q.b.clone());
            out.insert(format!("block{}.attn.k.lora.A", i), k.a.clone());
            out.insert(format!("block{}.attn.k.lora.B", i), k.b.clone());
            out.insert(format!("block{}.attn.v.lora.A", i), v.a.clone());
            out.insert(format!("block{}.attn.v.lora.B", i), v.b.clone());
            out.insert(format!("block{}.attn.o.lora.A", i), o.a.clone());
            out.insert(format!("block{}.attn.o.lora.B", i), o.b.clone());
            out.insert(format!("block{}.mlp.fc1.lora.A", i), fc1.a.clone());
            out.insert(format!("block{}.mlp.fc1.lora.B", i), fc1.b.clone());
            out.insert(format!("block{}.mlp.fc2.lora.A", i), fc2.a.clone());
            out.insert(format!("block{}.mlp.fc2.lora.B", i), fc2.b.clone());
        }
        out
    }
    /// Per-block mutable callback to avoid aliasing across whole model
    pub fn for_each_block_named_params_mut<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(&mut HashMap<String, &mut Parameter>) -> Result<()>,
    {
        for (i, blk) in self.per_block.iter_mut().enumerate() {
            // Safe disjoint borrows using split_at_mut chain
            let (b0, rest) = blk.split_at_mut(1);
            let (b1, rest) = rest.split_at_mut(1);
            let (b2, rest) = rest.split_at_mut(1);
            let (b3, rest) = rest.split_at_mut(1);
            let (b4, b5) = rest.split_at_mut(1);
            let q = &mut b0[0];
            let k = &mut b1[0];
            let v = &mut b2[0];
            let o = &mut b3[0];
            let fc1 = &mut b4[0];
            let fc2 = &mut b5[0];
            let mut m: HashMap<String, &mut Parameter> = HashMap::new();
            m.insert(format!("block{}.attn.q.lora.A", i), &mut q.a);
            m.insert(format!("block{}.attn.q.lora.B", i), &mut q.b);
            m.insert(format!("block{}.attn.k.lora.A", i), &mut k.a);
            m.insert(format!("block{}.attn.k.lora.B", i), &mut k.b);
            m.insert(format!("block{}.attn.v.lora.A", i), &mut v.a);
            m.insert(format!("block{}.attn.v.lora.B", i), &mut v.b);
            m.insert(format!("block{}.attn.o.lora.A", i), &mut o.a);
            m.insert(format!("block{}.attn.o.lora.B", i), &mut o.b);
            m.insert(format!("block{}.mlp.fc1.lora.A", i), &mut fc1.a);
            m.insert(format!("block{}.mlp.fc1.lora.B", i), &mut fc1.b);
            m.insert(format!("block{}.mlp.fc2.lora.A", i), &mut fc2.a);
            m.insert(format!("block{}.mlp.fc2.lora.B", i), &mut fc2.b);
            f(&mut m)?;
        }
        Ok(())
    }
    /// Ordered parameter names matching all_params() order
    pub fn param_names_in_order(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, _group) in self.per_block.iter().enumerate() {
            out.push(format!("block{}.attn.q.lora.A", i));
            out.push(format!("block{}.attn.q.lora.B", i));
            out.push(format!("block{}.attn.k.lora.A", i));
            out.push(format!("block{}.attn.k.lora.B", i));
            out.push(format!("block{}.attn.v.lora.A", i));
            out.push(format!("block{}.attn.v.lora.B", i));
            out.push(format!("block{}.attn.o.lora.A", i));
            out.push(format!("block{}.attn.o.lora.B", i));
            out.push(format!("block{}.mlp.fc1.lora.A", i));
            out.push(format!("block{}.mlp.fc1.lora.B", i));
            out.push(format!("block{}.mlp.fc2.lora.A", i));
            out.push(format!("block{}.mlp.fc2.lora.B", i));
        }
        out
    }

    /// Borrowed immutable view for EMA / budgets
    pub fn named_params<'a>(&'a self) -> HashMap<String, &'a Parameter> {
        let mut out = HashMap::new();
        for (i, group) in self.per_block.iter().enumerate() {
            out.insert(format!("block{}.attn.q.lora.A", i), &group[0].a);
            out.insert(format!("block{}.attn.q.lora.B", i), &group[0].b);
            out.insert(format!("block{}.attn.k.lora.A", i), &group[1].a);
            out.insert(format!("block{}.attn.k.lora.B", i), &group[1].b);
            out.insert(format!("block{}.attn.v.lora.A", i), &group[2].a);
            out.insert(format!("block{}.attn.v.lora.B", i), &group[2].b);
            out.insert(format!("block{}.attn.o.lora.A", i), &group[3].a);
            out.insert(format!("block{}.attn.o.lora.B", i), &group[3].b);
            out.insert(format!("block{}.mlp.fc1.lora.A", i), &group[4].a);
            out.insert(format!("block{}.mlp.fc1.lora.B", i), &group[4].b);
            out.insert(format!("block{}.mlp.fc2.lora.A", i), &group[5].a);
            out.insert(format!("block{}.mlp.fc2.lora.B", i), &group[5].b);
        }
        out
    }

    // named_params_mut intentionally omitted; prefer EMAHelper or per-block callbacks

    /// Snapshot current LoRA tensors for restoration
    pub fn snapshot_named_tensors(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        for (i, group) in self.per_block.iter().enumerate() {
            let q = &group[0];
            let k = &group[1];
            let v = &group[2];
            let o = &group[3];
            let fc1 = &group[4];
            let fc2 = &group[5];
            out.insert(format!("block{}.attn.q.lora.A", i), q.a.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.attn.q.lora.B", i), q.b.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.attn.k.lora.A", i), k.a.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.attn.k.lora.B", i), k.b.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.attn.v.lora.A", i), v.a.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.attn.v.lora.B", i), v.b.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.attn.o.lora.A", i), o.a.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.attn.o.lora.B", i), o.b.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.mlp.fc1.lora.A", i), fc1.a.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.mlp.fc1.lora.B", i), fc1.b.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.mlp.fc2.lora.A", i), fc2.a.tensor().map_err(Error::from)?);
            out.insert(format!("block{}.mlp.fc2.lora.B", i), fc2.b.tensor().map_err(Error::from)?);
        }
        Ok(out)
    }

    /// Apply named parameters into LoRA tensors in-place (by replacing tensors)
    pub fn apply_named_parameters(&mut self, params: &HashMap<String, Parameter>) -> Result<()> {
        for (name, p) in params {
            // Parse name: block{i}.attn.q.lora.A etc.
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() != 5 {
                continue;
            }
            let bi = parts[0];
            let cat = parts[1];
            let which = parts[2];
            let _lora = parts[3];
            let ab = parts[4];
            if !bi.starts_with("block") {
                continue;
            }
            let idx: usize = bi.trim_start_matches("block").parse().unwrap_or(0);
            if idx >= self.per_block.len() {
                continue;
            }
            let tnew = p.tensor().map_err(Error::from)?;
            let grp = &mut self.per_block[idx];
            match (cat, which, ab) {
                ("attn", "q", "A") => grp[0].a.set_data(tnew)?,
                ("attn", "q", "B") => grp[0].b.set_data(tnew)?,
                ("attn", "k", "A") => grp[1].a.set_data(tnew)?,
                ("attn", "k", "B") => grp[1].b.set_data(tnew)?,
                ("attn", "v", "A") => grp[2].a.set_data(tnew)?,
                ("attn", "v", "B") => grp[2].b.set_data(tnew)?,
                ("attn", "o", "A") => grp[3].a.set_data(tnew)?,
                ("attn", "o", "B") => grp[3].b.set_data(tnew)?,
                ("mlp", "fc1", "A") => grp[4].a.set_data(tnew)?,
                ("mlp", "fc1", "B") => grp[4].b.set_data(tnew)?,
                ("mlp", "fc2", "A") => grp[5].a.set_data(tnew)?,
                ("mlp", "fc2", "B") => grp[5].b.set_data(tnew)?,
                _ => {}
            }
        }
        Ok(())
    }

    /// Apply named tensors (Tensor values) into LoRA tensors in-place
    pub fn apply_named_tensors(&mut self, params: &HashMap<String, Tensor>) -> Result<()> {
        for (name, t) in params {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() != 5 {
                continue;
            }
            let bi = parts[0];
            let cat = parts[1];
            let which = parts[2];
            let _lora = parts[3];
            let ab = parts[4];
            if !bi.starts_with("block") {
                continue;
            }
            let idx: usize = bi.trim_start_matches("block").parse().unwrap_or(0);
            if idx >= self.per_block.len() {
                continue;
            }
            let grp = &mut self.per_block[idx];
            let tnew = t.clone();
            match (cat, which, ab) {
                ("attn", "q", "A") => grp[0].a.set_data(tnew)?,
                ("attn", "q", "B") => grp[0].b.set_data(tnew)?,
                ("attn", "k", "A") => grp[1].a.set_data(tnew)?,
                ("attn", "k", "B") => grp[1].b.set_data(tnew)?,
                ("attn", "v", "A") => grp[2].a.set_data(tnew)?,
                ("attn", "v", "B") => grp[2].b.set_data(tnew)?,
                ("attn", "o", "A") => grp[3].a.set_data(tnew)?,
                ("attn", "o", "B") => grp[3].b.set_data(tnew)?,
                ("mlp", "fc1", "A") => grp[4].a.set_data(tnew)?,
                ("mlp", "fc1", "B") => grp[4].b.set_data(tnew)?,
                ("mlp", "fc2", "A") => grp[5].a.set_data(tnew)?,
                ("mlp", "fc2", "B") => grp[5].b.set_data(tnew)?,
                _ => {}
            }
        }
        Ok(())
    }
}
