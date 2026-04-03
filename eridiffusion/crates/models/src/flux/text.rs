use anyhow::{anyhow, bail, Context, Result};
use eridiffusion_common_io as cio;
use eridiffusion_common_text::tokenizer::HfTokenizer;
use eridiffusion_core::Device;
use flame_core::{DType, Tensor};
use crate::devtensor::{shape2, shape3, tensor_from_vec_on, zeros_on};

const DEFAULT_SEQ_LEN: usize = 256;
const LAYER_NORM_EPS: f32 = 1e-6;

/// Minimal Flux text stack – embeds tokens with the T5 shared embedding matrix
/// and applies the final layer norm from the checkpoint.  This is sufficient for
/// the restored trainer, which projects the resulting activations downstream.
pub struct FluxText {
    tokenizer: HfTokenizer,
    embedding: Tensor,       // [vocab, hidden] on target device, BF16/F32
    ln_weight: Vec<f32>,     // gamma
    ln_bias: Vec<f32>,       // beta
    hidden: usize,
    max_seq: usize,
    device: Device,
    dtype: DType,
}

impl FluxText {
    pub fn load(tokenizer_path: &str, weights_path: &str, device: Device, dtype: DType) -> Result<Self> {
        if !matches!(device, Device::Cuda(_)) {
            bail!("FluxText requires a CUDA device");
        }
        if !std::path::Path::new(tokenizer_path).exists() {
            bail!("Flux tokenizer missing at {}", tokenizer_path);
        }
        if !std::path::Path::new(weights_path).exists() {
            bail!("Flux text weights missing at {}", weights_path);
        }

        let tokenizer = HfTokenizer::from_path(tokenizer_path, DEFAULT_SEQ_LEN)?;
        let st = cio::STFile::open(weights_path)?;

        let embedding = load_embedding(&st, &device, dtype)?;
        let hidden = embedding.shape().dims()[1];

        let ln_weight = load_layer_norm(&st, &device, "encoder.final_layer_norm.weight", hidden)?
            .unwrap_or_else(|| vec![1.0; hidden]);
        let ln_bias = load_layer_norm(&st, &device, "encoder.final_layer_norm.bias", hidden)?
            .unwrap_or_else(|| vec![0.0; hidden]);

        Ok(Self {
            tokenizer,
            embedding,
            ln_weight,
            ln_bias,
            hidden,
            max_seq: DEFAULT_SEQ_LEN,
            device,
            dtype,
        })
    }

    pub fn hidden_dim(&self) -> usize { self.hidden }
    pub fn max_seq(&self) -> usize { self.max_seq }

    /// Encode a batch of prompts into [B, seq, hidden] activations.
    pub fn encode(&self, prompts: &[String]) -> Result<Tensor> {
        if prompts.is_empty() {
            let empty = zeros_on(
                shape3(0, self.max_seq as i64, self.hidden as i64),
                &self.device,
                self.dtype,
            )
            .map_err(|e| anyhow!("FluxText: empty tensor alloc failed: {e}"))?;
            return Ok(empty);
        }

        let (ids_gpu, lengths_gpu, _pad_id) = self.tokenizer.encode_batch(prompts)?;
        let ids = ids_gpu.to_dtype(DType::I32)?;
        let lengths_vec = lengths_gpu
            .to_dtype(DType::F32)?
            .to_vec()?
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<_>>();

        let gathered = self.embedding.index_select0(&ids)?;
        let dims = gathered.shape().dims().to_vec();
        let (batch, seq, hidden) = (dims[0], dims[1], dims[2]);

        // Bring to host for the layer-norm application.
        let mut activations = gathered.to_dtype(DType::F32)?.to_vec()?;
        apply_layer_norm(
            &mut activations,
            batch,
            seq,
            hidden,
            &self.ln_weight,
            &self.ln_bias,
            &lengths_vec,
        );

        let mut tensor = tensor_from_vec_on(
            activations,
            shape3(batch as i64, seq as i64, hidden as i64),
            &self.device,
            DType::F32,
        )
        .map_err(|e| anyhow!("FluxText: activation upload failed: {e}"))?;
        if self.dtype != DType::F32 {
            tensor = tensor.to_dtype(self.dtype)?;
        }
        Ok(tensor)
    }

    /// Simple mean pooling across the sequence dimension (ignores padding).
    #[allow(dead_code)]
    pub fn pooled_mean(&self, hidden: &Tensor) -> Result<Tensor> {
        let dims = hidden.shape().dims().to_vec();
        if dims.len() != 3 { bail!("expected [B,seq,hidden], got {:?}", dims); }
        let (batch, seq, hid) = (dims[0], dims[1], dims[2]);
        let data = hidden.to_dtype(DType::F32)?.to_vec()?;
        let mut pooled = vec![0f32; batch * hid];
        for b in 0..batch {
            for j in 0..hid {
                let mut sum = 0.0;
                for s in 0..seq { sum += data[(b * seq + s) * hid + j]; }
                pooled[b * hid + j] = sum / (seq as f32);
            }
        }
        let mut tensor = tensor_from_vec_on(
            pooled,
            shape2(batch as i64, hid as i64),
            &self.device,
            DType::F32,
        )
        .map_err(|e| anyhow!("FluxText: pooled upload failed: {e}"))?;
        if self.dtype != DType::F32 {
            tensor = tensor.to_dtype(self.dtype)?;
        }
        Ok(tensor)
    }
}

fn load_embedding(st: &cio::STFile, device: &Device, dtype: DType) -> Result<Tensor> {
    const CANDIDATES: [&str; 4] = [
        "shared.weight",
        "shared.embedding.weight",
        "shared.embed.weight",
        "encoder.embed_tokens.weight",
    ];
    for key in CANDIDATES.iter() {
        if st.tensor(key).is_some() {
            return cio::load_tensor_to_device(st, key, device, Some(dtype))
                .with_context(|| format!("FluxText: failed loading embedding '{key}'"));
        }
    }
    bail!("FluxText: embedding matrix not found in safetensors");
}

fn load_layer_norm(st: &cio::STFile, device: &Device, key: &str, hidden: usize) -> Result<Option<Vec<f32>>> {
    if st.tensor(key).is_none() {
        return Ok(None);
    }
    let tensor = cio::load_tensor_to_device(st, key, device, Some(DType::F32))?;
    let vec = tensor.to_dtype(DType::F32)?.to_vec()?;
    if vec.len() != hidden {
        bail!("FluxText: expected layer-norm '{}' to have {} elements, found {}", key, hidden, vec.len());
    }
    Ok(Some(vec))
}

fn apply_layer_norm(
    activations: &mut [f32],
    batch: usize,
    seq: usize,
    hidden: usize,
    gamma: &[f32],
    beta: &[f32],
    lengths: &[usize],
) {
    for b in 0..batch {
        let valid = lengths.get(b).copied().unwrap_or(seq).min(seq);
        for s in 0..seq {
            let base = (b * seq + s) * hidden;
            if s >= valid {
                for j in 0..hidden { activations[base + j] = 0.0; }
                continue;
            }
            let mut mean = 0.0;
            for j in 0..hidden { mean += activations[base + j]; }
            mean /= hidden as f32;
            let mut var = 0.0;
            for j in 0..hidden {
                let diff = activations[base + j] - mean;
                var += diff * diff;
            }
            var /= hidden as f32;
            let inv_std = 1.0 / (var + LAYER_NORM_EPS).sqrt();
            for j in 0..hidden {
                let norm = (activations[base + j] - mean) * inv_std;
                activations[base + j] = norm * gamma[j] + beta[j];
            }
        }
    }
}
