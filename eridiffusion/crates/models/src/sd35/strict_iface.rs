#![cfg(feature = "sd35_strict_loader")]

use std::collections::HashMap;

use flame_core::Tensor;

use super::strict_loader::Sd35StrictBundle;

pub trait Sd35StrictBundleExt {
    fn tensor(&self, key: &str) -> Option<&Tensor>;
    fn has(&self, key: &str) -> bool;
    fn tensors(&self) -> &HashMap<String, Tensor>;
}

impl Sd35StrictBundleExt for Sd35StrictBundle {
    fn tensor(&self, key: &str) -> Option<&Tensor> {
        self.tensors.get(key)
    }

    fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }
}
