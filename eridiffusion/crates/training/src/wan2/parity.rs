
#![allow(dead_code)]
// Parity hooks (CPU by default). Real harness lives in scripts; this is a compile-safe stub.
#[derive(Clone, Debug)]
pub struct ParityCfg { pub mode: String }
pub fn run_parity(_cfg: &ParityCfg) -> bool { true }
