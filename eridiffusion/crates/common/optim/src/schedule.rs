#[derive(Clone, Copy)]
pub struct CosineSchedule { pub warmup_steps: usize, pub total_steps: usize, pub base_lr: f32 }

impl CosineSchedule {
    pub fn lr_at(&self, step: usize) -> f32 {
        use std::f32::consts::PI;
        if step < self.warmup_steps { return self.base_lr * (step as f32) / (self.warmup_steps.max(1) as f32); }
        let t = ((step - self.warmup_steps) as f32 / (self.total_steps.max(1) as f32 - self.warmup_steps as f32)).min(1.0).max(0.0);
        let cos = (1.0 + (PI * t).cos()) * 0.5; // cosine from 1 to 0
        self.base_lr * cos
    }
}

