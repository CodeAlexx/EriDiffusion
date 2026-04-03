//! Flux safetensor key converters.
//! Covers the `double_blocks.{i}.img_*` family used by flux1-dev dumps.

#[derive(Clone, Copy, Debug)]
pub enum Schema {
    /// `double_blocks.{i}.img_attn.qkv.weight`, etc.
    DiffusersFluxDev,
}

#[derive(Clone, Debug)]
pub struct KeyConv {
    pub schema: Schema,
}

impl Default for KeyConv {
    fn default() -> Self { Self { schema: Schema::DiffusersFluxDev } }
}

impl KeyConv {
    #[inline] fn prefix(&self, i: usize) -> String { format!("double_blocks.{i}") }

    pub fn img_attn_qkv_weight(&self, i: usize) -> String {
        format!("{}.img_attn.qkv.weight", self.prefix(i))
    }
    pub fn img_attn_qkv_bias(&self, i: usize) -> String {
        format!("{}.img_attn.qkv.bias", self.prefix(i))
    }
    pub fn img_attn_proj_weight(&self, i: usize) -> String {
        format!("{}.img_attn.proj.weight", self.prefix(i))
    }
    pub fn img_attn_proj_bias(&self, i: usize) -> String {
        format!("{}.img_attn.proj.bias", self.prefix(i))
    }

    pub fn img_mlp_fc1_weight(&self, i: usize) -> String {
        format!("{}.img_mlp.0.weight", self.prefix(i))
    }
    pub fn img_mlp_fc1_bias(&self, i: usize) -> String {
        format!("{}.img_mlp.0.bias", self.prefix(i))
    }
    pub fn img_mlp_fc2_weight(&self, i: usize) -> String {
        format!("{}.img_mlp.2.weight", self.prefix(i))
    }
    pub fn img_mlp_fc2_bias(&self, i: usize) -> String {
        format!("{}.img_mlp.2.bias", self.prefix(i))
    }
}

pub fn default_keyconv() -> KeyConv {
    KeyConv::default()
}
