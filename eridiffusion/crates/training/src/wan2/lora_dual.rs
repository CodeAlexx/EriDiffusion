
#![allow(dead_code)]
#[derive(Clone, Debug)]
pub struct DualLoraSite { pub a_key: String, pub b_key: String }
#[derive(Clone, Debug)]
pub struct DualLoraMap { pub sites: Vec<DualLoraSite> }
impl DualLoraMap { pub fn new() -> Self { Self { sites: vec![] } } }
