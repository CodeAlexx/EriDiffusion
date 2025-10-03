
#![allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum LossKind { L2, SRPO }
pub fn l2_loss(_pred:&[f32], _target:&[f32])->f32 { 0.0 }
pub fn srpo_loss(_pred:&[f32], _target:&[f32])->f32 { 0.0 }
