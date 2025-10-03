use std::collections::HashMap;

use eridiffusion_core::{Error, Result};
use flame_core::{DType, Device as FlameDevice, Parameter, Shape, Tensor};

use crate::chroma::lora::LoRALinear;

const BLOCK_COUNT: usize = 38;
const HIDDEN_DIM: usize = 2432;
const EXPANDED_DIM: usize = HIDDEN_DIM * 4;
const CONTEXT_DIM: usize = 2432;
const FINAL_HEAD_DIM: usize = 64;

/// Available optional head adapters for SD3.5.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sd35LoraHeadTarget {
    FinalLinear,
}

impl Sd35LoraHeadTarget {
    fn key(self) -> &'static str {
        match self {
            Sd35LoraHeadTarget::FinalLinear => "head.final_layer.linear",
        }
    }

    fn dims(self) -> (usize, usize) {
        match self {
            Sd35LoraHeadTarget::FinalLinear => (HIDDEN_DIM, FINAL_HEAD_DIM),
        }
    }
}

/// Policy describing which adapter sites to include.
#[derive(Clone, Debug)]
pub struct Sd35LoraPolicy<'a> {
    /// Target identifiers to enable (e.g. ["q","k","v","o","fc1","fc2","ctx"]).
    pub targets: &'a [&'a str],
    /// Optional head adapters to instantiate.
    pub head_targets: &'a [Sd35LoraHeadTarget],
    /// Whether to zero-initialize the up projection (classic LoRA init).
    pub zero_init_b: bool,
}

impl<'a> Default for Sd35LoraPolicy<'a> {
    fn default() -> Self {
        Self {
            targets: &[],
            head_targets: &[],
            zero_init_b: true,
        }
    }
}

/// Output of the SD3.5 LoRA builder.
pub struct Sd35LoraBuild {
    /// Ordered adapters matching the index map.
    pub adapters: Vec<LoRALinear>,
    /// Mapping from logical site name → adapter index.
    pub index: HashMap<String, usize>,
    /// Trainable parameter bytes across all adapters.
    pub trainable_bytes: usize,
}

fn should_enable(targets: &[&str], key: &str) -> bool {
    if targets.is_empty() {
        return true;
    }
    targets.iter().any(|t| t.eq_ignore_ascii_case(key))
}

fn create_lora(
    dev: &FlameDevice,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    alpha: f32,
    zero_init_b: bool,
) -> Result<(LoRALinear, usize, usize)> {
    if rank == 0 {
        return Err(Error::Config("LoRA rank must be >0".into()));
    }
    let scale = (1.0f32 / rank.max(1) as f32).sqrt();
    let a_t = Tensor::randn(
        Shape::from_dims(&[in_dim, rank]),
        0.0,
        scale,
        dev.cuda_device_arc(),
    )
    .map_err(Error::from)?
    .to_dtype(DType::BF16)
    .map_err(Error::from)?
    .requires_grad_(true);

    let b_shape = Shape::from_dims(&[rank, out_dim]);
    let b_t = if zero_init_b {
        Tensor::zeros_dtype(b_shape, DType::BF16, dev.cuda_device_arc()).map_err(Error::from)?
    } else {
        Tensor::randn(b_shape, 0.0, scale, dev.cuda_device_arc())
            .map_err(Error::from)?
            .to_dtype(DType::BF16)
            .map_err(Error::from)?
    }
    .requires_grad_(true);

    Ok((
        LoRALinear {
            a: Parameter::new(a_t),
            b: Parameter::new(b_t),
            rank,
            alpha,
        },
        in_dim,
        out_dim,
    ))
}

fn register_adapter(
    adapters: &mut Vec<LoRALinear>,
    index: &mut HashMap<String, usize>,
    bytes: &mut usize,
    name: String,
    lora: LoRALinear,
    in_dim: usize,
    out_dim: usize,
) {
    let rank = lora.rank;
    adapters.push(lora);
    let slot = adapters.len() - 1;
    index.insert(name, slot);
    let elems = rank * (in_dim + out_dim);
    *bytes += elems * 2; // BF16 parameters (2 bytes each)
}

fn base_allow<'a>(policy: &Sd35LoraPolicy<'a>, key: &str) -> bool {
    should_enable(policy.targets, key)
}

/// Build all SD3.5 LoRA adapters according to the provided policy.
pub fn build_sd35_loras(
    device: FlameDevice,
    rank: usize,
    alpha: f32,
    policy: Sd35LoraPolicy<'_>,
) -> Result<Sd35LoraBuild> {
    if rank == 0 {
        return Ok(Sd35LoraBuild {
            adapters: Vec::new(),
            index: HashMap::new(),
            trainable_bytes: 0,
        });
    }

    let mut adapters: Vec<LoRALinear> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut bytes: usize = 0;

    let want_q = base_allow(&policy, "q");
    let want_k = base_allow(&policy, "k");
    let want_v = base_allow(&policy, "v");
    let want_o = base_allow(&policy, "o");
    let want_fc1 = base_allow(&policy, "fc1");
    let want_fc2 = base_allow(&policy, "fc2");
    let want_ctx = base_allow(&policy, "ctx") || base_allow(&policy, "context");

    for block in 0..BLOCK_COUNT {
        let prefix = format!("block{block}.");
        if want_q {
            let (lora, in_d, out_d) = create_lora(
                &device,
                HIDDEN_DIM,
                HIDDEN_DIM,
                rank,
                alpha,
                policy.zero_init_b,
            )?;
            register_adapter(
                &mut adapters,
                &mut index,
                &mut bytes,
                format!("{}x.attn.q", prefix),
                lora,
                in_d,
                out_d,
            );
        }
        if want_k {
            let (lora, in_d, out_d) = create_lora(
                &device,
                HIDDEN_DIM,
                HIDDEN_DIM,
                rank,
                alpha,
                policy.zero_init_b,
            )?;
            register_adapter(
                &mut adapters,
                &mut index,
                &mut bytes,
                format!("{}x.attn.k", prefix),
                lora,
                in_d,
                out_d,
            );
        }
        if want_v {
            let (lora, in_d, out_d) = create_lora(
                &device,
                HIDDEN_DIM,
                HIDDEN_DIM,
                rank,
                alpha,
                policy.zero_init_b,
            )?;
            register_adapter(
                &mut adapters,
                &mut index,
                &mut bytes,
                format!("{}x.attn.v", prefix),
                lora,
                in_d,
                out_d,
            );
        }
        if want_o {
            let (lora, in_d, out_d) = create_lora(
                &device,
                HIDDEN_DIM,
                HIDDEN_DIM,
                rank,
                alpha,
                policy.zero_init_b,
            )?;
            register_adapter(
                &mut adapters,
                &mut index,
                &mut bytes,
                format!("{}x.attn.o", prefix),
                lora,
                in_d,
                out_d,
            );
        }
        if want_fc1 {
            let (lora, in_d, out_d) = create_lora(
                &device,
                HIDDEN_DIM,
                EXPANDED_DIM,
                rank,
                alpha,
                policy.zero_init_b,
            )?;
            register_adapter(
                &mut adapters,
                &mut index,
                &mut bytes,
                format!("{}x.mlp.fc1", prefix),
                lora,
                in_d,
                out_d,
            );
        }
        if want_fc2 {
            let (lora, in_d, out_d) = create_lora(
                &device,
                EXPANDED_DIM,
                HIDDEN_DIM,
                rank,
                alpha,
                policy.zero_init_b,
            )?;
            register_adapter(
                &mut adapters,
                &mut index,
                &mut bytes,
                format!("{}x.mlp.fc2", prefix),
                lora,
                in_d,
                out_d,
            );
        }

        if want_ctx {
            if want_q {
                let (lora, in_d, out_d) = create_lora(
                    &device,
                    CONTEXT_DIM,
                    CONTEXT_DIM,
                    rank,
                    alpha,
                    policy.zero_init_b,
                )?;
                register_adapter(
                    &mut adapters,
                    &mut index,
                    &mut bytes,
                    format!("{}ctx.attn.q", prefix),
                    lora,
                    in_d,
                    out_d,
                );
            }
            if want_k {
                let (lora, in_d, out_d) = create_lora(
                    &device,
                    CONTEXT_DIM,
                    CONTEXT_DIM,
                    rank,
                    alpha,
                    policy.zero_init_b,
                )?;
                register_adapter(
                    &mut adapters,
                    &mut index,
                    &mut bytes,
                    format!("{}ctx.attn.k", prefix),
                    lora,
                    in_d,
                    out_d,
                );
            }
            if want_v {
                let (lora, in_d, out_d) = create_lora(
                    &device,
                    CONTEXT_DIM,
                    CONTEXT_DIM,
                    rank,
                    alpha,
                    policy.zero_init_b,
                )?;
                register_adapter(
                    &mut adapters,
                    &mut index,
                    &mut bytes,
                    format!("{}ctx.attn.v", prefix),
                    lora,
                    in_d,
                    out_d,
                );
            }
            if want_o {
                let (lora, in_d, out_d) = create_lora(
                    &device,
                    CONTEXT_DIM,
                    CONTEXT_DIM,
                    rank,
                    alpha,
                    policy.zero_init_b,
                )?;
                register_adapter(
                    &mut adapters,
                    &mut index,
                    &mut bytes,
                    format!("{}ctx.attn.o", prefix),
                    lora,
                    in_d,
                    out_d,
                );
            }
            if want_fc1 {
                let (lora, in_d, out_d) = create_lora(
                    &device,
                    CONTEXT_DIM,
                    EXPANDED_DIM,
                    rank,
                    alpha,
                    policy.zero_init_b,
                )?;
                register_adapter(
                    &mut adapters,
                    &mut index,
                    &mut bytes,
                    format!("{}ctx.mlp.fc1", prefix),
                    lora,
                    in_d,
                    out_d,
                );
            }
            if want_fc2 {
                let (lora, in_d, out_d) = create_lora(
                    &device,
                    EXPANDED_DIM,
                    CONTEXT_DIM,
                    rank,
                    alpha,
                    policy.zero_init_b,
                )?;
                register_adapter(
                    &mut adapters,
                    &mut index,
                    &mut bytes,
                    format!("{}ctx.mlp.fc2", prefix),
                    lora,
                    in_d,
                    out_d,
                );
            }
        }
    }

    for head in policy.head_targets {
        let (in_dim, out_dim) = head.dims();
        let (lora, in_d, out_d) =
            create_lora(&device, in_dim, out_dim, rank, alpha, policy.zero_init_b)?;
        register_adapter(
            &mut adapters,
            &mut index,
            &mut bytes,
            head.key().to_string(),
            lora,
            in_d,
            out_d,
        );
    }

    Ok(Sd35LoraBuild {
        adapters,
        index,
        trainable_bytes: bytes,
    })
}
