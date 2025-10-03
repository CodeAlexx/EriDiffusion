use anyhow::Result;
use flame_core::{Tensor, Shape, DType, Device};

pub struct HfTokenizer {
    tok: tokenizers::Tokenizer,
    seq: usize,
}

impl HfTokenizer {
    pub fn from_path(path: &str, seq: usize) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(Self { tok, seq })
    }

    /// Returns (ids I32[B,seq], lengths I32[B], pad_id)
    pub fn encode_batch(&self, prompts: &[String]) -> Result<(Tensor, Tensor, i32)> {
        let device = Device::cuda(0)?;
        self.encode_batch_on(prompts, &device)
    }

    /// Same as [`encode_batch`] but allows specifying the target CUDA device explicitly.
    pub fn encode_batch_on(&self, prompts: &[String], device: &Device) -> Result<(Tensor, Tensor, i32)> {
        let b = prompts.len();
        let mut all_ids: Vec<i32> = Vec::with_capacity(b * self.seq);
        let mut lengths: Vec<i32> = Vec::with_capacity(b);
        // Pad ID (fallback 0 if unavailable)
        let mut pad_id: i32 = 0;
        if let Some(pp) = self.tok.get_padding() {
            pad_id = pp.pad_id as i32;
        }
        for p in prompts {
            let enc = self.tok.encode(p.as_str(), true).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let mut ids = enc.get_ids().to_vec();
            let len = ids.len().min(self.seq);
            ids.truncate(self.seq);
            ids.resize(self.seq, pad_id as u32);
            all_ids.extend(ids.iter().map(|&x| x as i32));
            lengths.push(len as i32);
        }
        // Build as F32 then mark dtype as I32 for now (stored as F32 internally)
        let cuda = device.cuda_device().clone();
        let ids_t = Tensor::from_vec(
                all_ids.iter().map(|&x| x as f32).collect(),
                Shape::from_dims(&[b, self.seq]),
                cuda.clone())?
            .to_dtype(DType::I32)?;
        let len_t = Tensor::from_vec(
                lengths.iter().map(|&x| x as f32).collect(),
                Shape::from_dims(&[b]),
                cuda.clone())?
            .to_dtype(DType::I32)?;
        Ok((ids_t, len_t, pad_id))
    }
}

// Back-compat convenience for older callers
#[derive(Debug, Clone)]
pub struct TokenBatch { pub ids: Vec<i32>, pub lengths: Vec<usize>, pub seq: usize, pub batch: usize }

impl HfTokenizer {
    pub fn tokenize(&self, texts: &[String]) -> Result<TokenBatch> {
        let device = Device::cuda(0)?;
        let (ids_t, len_t, _pad_id) = self.encode_batch_on(texts, &device)?;
        let ids_f = ids_t.to_dtype(DType::F32)?.to_vec()?;
        let lens_f = len_t.to_dtype(DType::F32)?.to_vec()?;
        let ids: Vec<i32> = ids_f.into_iter().map(|x| x as i32).collect();
        let lengths: Vec<usize> = lens_f.into_iter().map(|x| x as usize).collect();
        Ok(TokenBatch { ids, lengths, seq: self.seq, batch: texts.len() })
    }
}
