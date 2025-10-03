
#![allow(dead_code)]
#[derive(Clone, Debug)]
pub struct EmaCfg { pub decay: f32 }
#[derive(Clone, Debug)]
pub struct EmaState { pub step: u64 }
impl EmaState {
    pub fn new() -> Self { Self { step: 0 } }
    pub fn update(&mut self) { self.step += 1; }
}
