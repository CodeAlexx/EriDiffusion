use flame_core::DType;

pub const BLOCK_COUNT: usize = 38;
pub const HIDDEN_SIZE: usize = 2432;
pub const EXPANDED_DIM: usize = HIDDEN_SIZE * 4; // 9728
pub const ADA_DIM_FULL: usize = HIDDEN_SIZE * 6; // 14592
pub const ADA_DIM_LAST_CONTEXT: usize = HIDDEN_SIZE * 2; // 4864
pub const LN_DIM: usize = 64;
pub const LATENT_CHANNELS: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorLoad {
    Direct,
    /// Slice of a fused tensor along an axis (start..start+len)
    Slice {
        axis: usize,
        start: usize,
        len: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub logical_key: String,
    pub tensor_key: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub optional: bool,
    pub load: TensorLoad,
}

impl TensorSpec {
    fn direct(logical: String, shape: Vec<usize>, optional: bool) -> Self {
        Self {
            tensor_key: logical.clone(),
            logical_key: logical,
            shape,
            dtype: DType::BF16,
            optional,
            load: TensorLoad::Direct,
        }
    }

    fn slice(
        logical: String,
        tensor_key: String,
        shape: Vec<usize>,
        axis: usize,
        start: usize,
        len: usize,
        optional: bool,
    ) -> Self {
        Self {
            logical_key: logical,
            tensor_key,
            shape,
            dtype: DType::BF16,
            optional,
            load: TensorLoad::Slice { axis, start, len },
        }
    }
}

fn slice_stride(index: usize) -> usize {
    index * HIDDEN_SIZE
}

fn qkv_slice(meta_name: &str, fused_name: &str, index: usize, optional: bool) -> TensorSpec {
    TensorSpec::slice(
        meta_name.to_string(),
        fused_name.to_string(),
        vec![HIDDEN_SIZE, HIDDEN_SIZE],
        0,
        slice_stride(index),
        HIDDEN_SIZE,
        optional,
    )
}

fn qkv_bias_slice(meta_name: &str, fused_name: &str, index: usize, optional: bool) -> TensorSpec {
    TensorSpec::slice(
        meta_name.to_string(),
        fused_name.to_string(),
        vec![HIDDEN_SIZE],
        0,
        slice_stride(index),
        HIDDEN_SIZE,
        optional,
    )
}

fn x_block_specs(block: usize, prefix: &str) -> Vec<TensorSpec> {
    let mut specs = Vec::new();
    let fused_weight = format!("{prefix}.attn.qkv.weight");
    let fused_bias = format!("{prefix}.attn.qkv.bias");
    specs.push(qkv_slice(&format!("{prefix}.attn.q.weight"), &fused_weight, 0, false));
    specs.push(qkv_slice(&format!("{prefix}.attn.k.weight"), &fused_weight, 1, false));
    specs.push(qkv_slice(&format!("{prefix}.attn.v.weight"), &fused_weight, 2, false));
    specs.push(qkv_bias_slice(&format!("{prefix}.attn.q.bias"), &fused_bias, 0, false));
    specs.push(qkv_bias_slice(&format!("{prefix}.attn.k.bias"), &fused_bias, 1, false));
    specs.push(qkv_bias_slice(&format!("{prefix}.attn.v.bias"), &fused_bias, 2, false));

    specs.push(TensorSpec::direct(
        format!("{prefix}.attn.proj.weight"),
        vec![HIDDEN_SIZE, HIDDEN_SIZE],
        false,
    ));
    specs.push(TensorSpec::direct(format!("{prefix}.attn.proj.bias"), vec![HIDDEN_SIZE], false));

    specs.push(TensorSpec::direct(format!("{prefix}.attn.ln_q.weight"), vec![LN_DIM], false));
    specs.push(TensorSpec::direct(format!("{prefix}.attn.ln_k.weight"), vec![LN_DIM], false));

    specs.push(TensorSpec::direct(
        format!("{prefix}.adaLN_modulation.1.weight"),
        vec![ADA_DIM_FULL, HIDDEN_SIZE],
        false,
    ));
    specs.push(TensorSpec::direct(
        format!("{prefix}.adaLN_modulation.1.bias"),
        vec![ADA_DIM_FULL],
        false,
    ));

    specs.push(TensorSpec::direct(
        format!("{prefix}.mlp.fc1.weight"),
        vec![EXPANDED_DIM, HIDDEN_SIZE],
        false,
    ));
    specs.push(TensorSpec::direct(format!("{prefix}.mlp.fc1.bias"), vec![EXPANDED_DIM], false));
    specs.push(TensorSpec::direct(
        format!("{prefix}.mlp.fc2.weight"),
        vec![HIDDEN_SIZE, EXPANDED_DIM],
        false,
    ));
    specs.push(TensorSpec::direct(format!("{prefix}.mlp.fc2.bias"), vec![HIDDEN_SIZE], false));

    // Context fusion block
    let context_prefix = format!("model.diffusion_model.joint_blocks.{block}.context_block");
    let fused_context_weight = format!("{context_prefix}.attn.qkv.weight");
    let fused_context_bias = format!("{context_prefix}.attn.qkv.bias");

    specs.push(qkv_slice(
        &format!("{context_prefix}.attn.q.weight"),
        &fused_context_weight,
        0,
        false,
    ));
    specs.push(qkv_slice(
        &format!("{context_prefix}.attn.k.weight"),
        &fused_context_weight,
        1,
        false,
    ));
    specs.push(qkv_slice(
        &format!("{context_prefix}.attn.v.weight"),
        &fused_context_weight,
        2,
        false,
    ));
    specs.push(qkv_bias_slice(
        &format!("{context_prefix}.attn.q.bias"),
        &fused_context_bias,
        0,
        false,
    ));
    specs.push(qkv_bias_slice(
        &format!("{context_prefix}.attn.k.bias"),
        &fused_context_bias,
        1,
        false,
    ));
    specs.push(qkv_bias_slice(
        &format!("{context_prefix}.attn.v.bias"),
        &fused_context_bias,
        2,
        false,
    ));

    // The final joint block (index 37) omits context projection/MLP heads.
    let context_has_proj = block != BLOCK_COUNT - 1;
    specs.push(TensorSpec::direct(
        format!("{context_prefix}.attn.proj.weight"),
        vec![HIDDEN_SIZE, HIDDEN_SIZE],
        !context_has_proj,
    ));
    specs.push(TensorSpec::direct(
        format!("{context_prefix}.attn.proj.bias"),
        vec![HIDDEN_SIZE],
        !context_has_proj,
    ));

    specs.push(TensorSpec::direct(
        format!("{context_prefix}.attn.ln_q.weight"),
        vec![LN_DIM],
        false,
    ));
    specs.push(TensorSpec::direct(
        format!("{context_prefix}.attn.ln_k.weight"),
        vec![LN_DIM],
        false,
    ));

    let adaln_dim = if block == BLOCK_COUNT - 1 { ADA_DIM_LAST_CONTEXT } else { ADA_DIM_FULL };
    specs.push(TensorSpec::direct(
        format!("{context_prefix}.adaLN_modulation.1.weight"),
        vec![adaln_dim, HIDDEN_SIZE],
        false,
    ));
    specs.push(TensorSpec::direct(
        format!("{context_prefix}.adaLN_modulation.1.bias"),
        vec![adaln_dim],
        false,
    ));

    specs.push(TensorSpec::direct(
        format!("{context_prefix}.mlp.fc1.weight"),
        vec![EXPANDED_DIM, HIDDEN_SIZE],
        !context_has_proj,
    ));
    specs.push(TensorSpec::direct(
        format!("{context_prefix}.mlp.fc1.bias"),
        vec![EXPANDED_DIM],
        !context_has_proj,
    ));
    specs.push(TensorSpec::direct(
        format!("{context_prefix}.mlp.fc2.weight"),
        vec![HIDDEN_SIZE, EXPANDED_DIM],
        !context_has_proj,
    ));
    specs.push(TensorSpec::direct(
        format!("{context_prefix}.mlp.fc2.bias"),
        vec![HIDDEN_SIZE],
        !context_has_proj,
    ));

    specs
}

pub fn block_tensors(block: usize) -> Vec<TensorSpec> {
    assert!(block < BLOCK_COUNT);
    let x_prefix = format!("model.diffusion_model.joint_blocks.{block}.x_block");
    x_block_specs(block, &x_prefix)
}

pub fn global_tensors() -> Vec<TensorSpec> {
    vec![
        TensorSpec::direct(
            "model.diffusion_model.context_embedder.weight".to_string(),
            vec![HIDDEN_SIZE, 4096],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.context_embedder.bias".to_string(),
            vec![HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.t_embedder.mlp.0.weight".to_string(),
            vec![HIDDEN_SIZE, 256],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.t_embedder.mlp.0.bias".to_string(),
            vec![HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.t_embedder.mlp.2.weight".to_string(),
            vec![HIDDEN_SIZE, HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.t_embedder.mlp.2.bias".to_string(),
            vec![HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.x_embedder.proj.weight".to_string(),
            vec![HIDDEN_SIZE, LATENT_CHANNELS, 2, 2],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.x_embedder.proj.bias".to_string(),
            vec![HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.y_embedder.mlp.0.weight".to_string(),
            vec![HIDDEN_SIZE, 2048],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.y_embedder.mlp.0.bias".to_string(),
            vec![HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.y_embedder.mlp.2.weight".to_string(),
            vec![HIDDEN_SIZE, HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.y_embedder.mlp.2.bias".to_string(),
            vec![HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.pos_embed".to_string(),
            vec![1, 36864, HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.final_layer.adaLN_modulation.1.weight".to_string(),
            vec![ADA_DIM_LAST_CONTEXT, HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.final_layer.adaLN_modulation.1.bias".to_string(),
            vec![ADA_DIM_LAST_CONTEXT],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.final_layer.linear.weight".to_string(),
            vec![64, HIDDEN_SIZE],
            false,
        ),
        TensorSpec::direct(
            "model.diffusion_model.final_layer.linear.bias".to_string(),
            vec![64],
            false,
        ),
    ]
}

pub fn all_tensors() -> Vec<TensorSpec> {
    let mut tensors = Vec::new();
    tensors.extend(global_tensors());
    for block in 0..BLOCK_COUNT {
        tensors.extend(block_tensors(block));
    }
    tensors
}

pub fn enumerate_keys(block: usize) -> Vec<String> {
    block_tensors(block).into_iter().map(|spec| spec.logical_key).collect()
}

pub fn enumerate_global_keys() -> Vec<String> {
    global_tensors().into_iter().map(|spec| spec.logical_key).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_zero_contains_qkv_slices() {
        let tensors = block_tensors(0);
        let find =
            |name: &str| -> &TensorSpec { tensors.iter().find(|t| t.logical_key == name).unwrap() };

        let q = find("model.diffusion_model.joint_blocks.0.x_block.attn.q.weight");
        assert_eq!(q.shape, vec![HIDDEN_SIZE, HIDDEN_SIZE]);
        match &q.load {
            TensorLoad::Slice { axis, start, len } => {
                assert_eq!((*axis, *start, *len), (0, 0, HIDDEN_SIZE));
            }
            other => panic!("expected slice load, got {:?}", other),
        }

        let k = find("model.diffusion_model.joint_blocks.0.x_block.attn.k.weight");
        match &k.load {
            TensorLoad::Slice { axis, start, len } => {
                assert_eq!((*axis, *start, *len), (0, HIDDEN_SIZE, HIDDEN_SIZE));
            }
            other => panic!("expected slice load, got {:?}", other),
        }

        let v = find("model.diffusion_model.joint_blocks.0.x_block.attn.v.weight");
        match &v.load {
            TensorLoad::Slice { axis, start, len } => {
                assert_eq!((*axis, *start, *len), (0, HIDDEN_SIZE * 2, HIDDEN_SIZE));
            }
            other => panic!("expected slice load, got {:?}", other),
        }
    }

    #[test]
    fn optional_context_entries_marked_for_last_block() {
        let tensors = block_tensors(BLOCK_COUNT - 1);
        let find =
            |name: &str| -> &TensorSpec { tensors.iter().find(|t| t.logical_key == name).unwrap() };
        let context_proj = find(&format!(
            "model.diffusion_model.joint_blocks.{}.context_block.attn.proj.weight",
            BLOCK_COUNT - 1
        ));
        assert!(context_proj.optional);
        let context_fc = find(&format!(
            "model.diffusion_model.joint_blocks.{}.context_block.mlp.fc1.weight",
            BLOCK_COUNT - 1
        ));
        assert!(context_fc.optional);
    }

    #[test]
    fn global_tensors_non_empty() {
        let tensors = global_tensors();
        assert!(!tensors.is_empty());
        assert!(tensors
            .iter()
            .any(|t| t.logical_key == "model.diffusion_model.final_layer.linear.weight"));
    }

    #[test]
    fn enumerate_keys_has_entries() {
        let keys = enumerate_keys(0);
        assert!(!keys.is_empty());
        assert!(keys
            .iter()
            .any(|k| k == "model.diffusion_model.joint_blocks.0.x_block.attn.o.weight"));
    }
}
