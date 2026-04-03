pub mod backbone;
pub mod lora;
pub mod loader;

use anyhow::Result;
use flame_core::{Tensor};

#[derive(Clone, Debug)]
pub struct Sd35Cond {
    pub text_hidden: Tensor,
}

