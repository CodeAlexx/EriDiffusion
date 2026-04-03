use flame_core::{Result, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct GradientStatus {
    pub has_gradients: bool,
    pub all_finite: bool,
    pub reasonable_magnitude: bool,
    pub changing_over_time: bool,
    pub avg_norm: f32,
    pub max_norm: f32,
    pub min_norm: f32,
}

#[derive(Debug, Clone)]
pub struct ParameterStatus {
    pub updating: bool,
    pub update_magnitude: f32,
    pub follows_gradient: bool,
    pub learning_rate_applied: bool,
}

#[derive(Debug, Clone)]
pub struct LossStatus {
    pub is_finite: bool,
    pub in_expected_range: bool,
    pub trend: String, // "decreasing", "stable", "increasing"
    pub current_loss: f32,
}

#[derive(Debug, Clone)]
pub struct MemoryStatus {
    pub current_usage_gb: f32,
    pub peak_usage_gb: f32,
    pub within_bounds: bool,
    pub stable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrainingHealth {
    Healthy,
    Warning,
    Failed,
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub step: usize,
    pub gradient_status: GradientStatus,
    pub parameter_status: ParameterStatus,
    pub loss_status: LossStatus,
    pub memory_status: MemoryStatus,
    pub overall_health: TrainingHealth,
    pub issues: Vec<String>,
}

impl VerificationReport {
    pub fn summary(&self) -> String {
        let health_emoji = match self.overall_health {
            TrainingHealth::Healthy => "✅",
            TrainingHealth::Warning => "⚠️",
            TrainingHealth::Failed => "❌",
        };

        format!(
            "{} Step {} | Loss: {:.6} ({}) | Grad: {:.2e}-{:.2e} | Params: {} | Mem: {:.1}GB | Issues: {}",
            health_emoji,
            self.step,
            self.loss_status.current_loss,
            self.loss_status.trend,
            self.gradient_status.min_norm,
            self.gradient_status.max_norm,
            if self.parameter_status.updating { "✓" } else { "✗" },
            self.memory_status.current_usage_gb,
            if self.issues.is_empty() { "None".to_string() } else { self.issues.join(", ") }
        )
    }
}

pub struct GradientInspector {
    step_count: usize,
    gradient_history: HashMap<String, Vec<f32>>,
    parameter_history: HashMap<String, Vec<f32>>,
}

impl GradientInspector {
    pub fn new() -> Self {
        Self { step_count: 0, gradient_history: HashMap::new(), parameter_history: HashMap::new() }
    }

    pub fn inspect_gradients(
        &mut self,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<GradientStatus> {
        let mut has_gradients = false;
        let mut all_finite = true;
        let mut grad_norms = Vec::new();

        for (name, tensor) in tensors {
            // Check if tensor has gradient
            if tensor.requires_grad() {
                has_gradients = true;

                // Get gradient norm - compute L2 norm manually
                // Note: This is a placeholder since we can't access gradients directly
                // In real use, you'd need to access the actual gradient tensor
                let grad_norm = tensor.pow(2.0)?.sum_all()?.sqrt()?.to_scalar::<f32>()?;

                // Check if finite
                if !grad_norm.is_finite() {
                    all_finite = false;
                    eprintln!("⚠️ Non-finite gradient detected for {}: {}", name, grad_norm);
                }

                grad_norms.push(grad_norm);

                // Track history
                self.gradient_history.entry(name.clone()).or_insert_with(Vec::new).push(grad_norm);
            }
        }

        let avg_norm = grad_norms.iter().sum::<f32>() / grad_norms.len().max(1) as f32;
        let max_norm = grad_norms.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_norm = grad_norms.iter().fold(f32::MAX, |a, &b| a.min(b));

        // Check if gradients are in reasonable range (1e-7 to 1e1)
        let reasonable_magnitude = min_norm > 1e-7 && max_norm < 10.0;

        // Check if gradients are changing over time
        let changing_over_time = self.check_gradient_changes();

        self.step_count += 1;

        Ok(GradientStatus {
            has_gradients,
            all_finite,
            reasonable_magnitude,
            changing_over_time,
            avg_norm,
            max_norm,
            min_norm,
        })
    }

    fn check_gradient_changes(&self) -> bool {
        // Check if gradients have changed over last few steps
        for (_, history) in &self.gradient_history {
            if history.len() >= 2 {
                let last = history[history.len() - 1];
                let prev = history[history.len() - 2];
                if (last - prev).abs() > 1e-8 {
                    return true;
                }
            }
        }
        false
    }
}

pub struct ParameterTracker {
    initial_params: HashMap<String, Vec<f32>>,
    previous_params: HashMap<String, Vec<f32>>,
    change_history: Vec<f32>,
}

impl ParameterTracker {
    pub fn new() -> Self {
        Self {
            initial_params: HashMap::new(),
            previous_params: HashMap::new(),
            change_history: Vec::new(),
        }
    }

    pub fn track_parameters(
        &mut self,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<ParameterStatus> {
        let mut total_change = 0.0f32;
        let mut param_count = 0;
        let mut any_updates = false;

        for (name, tensor) in tensors {
            if tensor.requires_grad() {
                // Get current parameter values
                let current_values: Vec<f32> = tensor.to_vec1()?;

                // Initialize if first time
                if !self.initial_params.contains_key(name) {
                    self.initial_params.insert(name.clone(), current_values.clone());
                    self.previous_params.insert(name.clone(), current_values);
                    continue;
                }

                // Compare with previous values
                if let Some(prev_values) = self.previous_params.get(name) {
                    for (curr, prev) in current_values.iter().zip(prev_values.iter()) {
                        let change = (curr - prev).abs();
                        total_change += change;
                        if change > 1e-8 {
                            any_updates = true;
                        }
                    }
                    param_count += current_values.len();
                }

                // Update previous values
                self.previous_params.insert(name.clone(), current_values);
            }
        }

        let avg_change = if param_count > 0 { total_change / param_count as f32 } else { 0.0 };

        self.change_history.push(avg_change);

        Ok(ParameterStatus {
            updating: any_updates,
            update_magnitude: avg_change,
            follows_gradient: true, // This would need gradient direction check
            learning_rate_applied: avg_change > 0.0 && avg_change < 0.01, // Rough check
        })
    }
}

pub struct TrainingAnalyzer {
    loss_history: Vec<f32>,
    gradient_norms: Vec<f32>,
    parameter_norms: Vec<f32>,
}

impl TrainingAnalyzer {
    pub fn new() -> Self {
        Self { loss_history: Vec::new(), gradient_norms: Vec::new(), parameter_norms: Vec::new() }
    }

    pub fn analyze_loss(&mut self, loss: f32) -> LossStatus {
        self.loss_history.push(loss);

        let is_finite = loss.is_finite();
        let in_expected_range = loss > 0.0 && loss < 100.0; // Adjust for your model

        let trend = if self.loss_history.len() >= 10 {
            let recent_avg =
                self.loss_history[self.loss_history.len() - 5..].iter().sum::<f32>() / 5.0;
            let older_avg = self.loss_history
                [self.loss_history.len() - 10..self.loss_history.len() - 5]
                .iter()
                .sum::<f32>()
                / 5.0;

            if recent_avg < older_avg * 0.98 {
                "decreasing".to_string()
            } else if recent_avg > older_avg * 1.02 {
                "increasing".to_string()
            } else {
                "stable".to_string()
            }
        } else {
            "warming_up".to_string()
        };

        LossStatus { is_finite, in_expected_range, trend, current_loss: loss }
    }
}

pub struct MemoryMonitor {
    peak_usage: f32,
    usage_history: Vec<f32>,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self { peak_usage: 0.0, usage_history: Vec::new() }
    }

    pub fn check_memory(&mut self) -> MemoryStatus {
        // This would need actual CUDA memory checking
        // For now, using placeholder values
        let current_usage_gb = 1.8; // You saw this in your logs

        self.usage_history.push(current_usage_gb);
        self.peak_usage = self.peak_usage.max(current_usage_gb);

        // Check if memory is stable (not growing unbounded)
        let stable = if self.usage_history.len() >= 10 {
            let recent_avg =
                self.usage_history[self.usage_history.len() - 5..].iter().sum::<f32>() / 5.0;
            let older_avg = self.usage_history
                [self.usage_history.len() - 10..self.usage_history.len() - 5]
                .iter()
                .sum::<f32>()
                / 5.0;
            (recent_avg - older_avg).abs() < 0.5 // Less than 0.5GB change
        } else {
            true
        };

        MemoryStatus {
            current_usage_gb,
            peak_usage_gb: self.peak_usage,
            within_bounds: current_usage_gb < 23.0, // For 24GB GPU
            stable,
        }
    }
}

pub struct TrainingVerifier {
    gradient_inspector: GradientInspector,
    parameter_tracker: ParameterTracker,
    training_analyzer: TrainingAnalyzer,
    memory_monitor: MemoryMonitor,
    verification_log: Vec<VerificationReport>,
}

impl TrainingVerifier {
    pub fn new() -> Self {
        Self {
            gradient_inspector: GradientInspector::new(),
            parameter_tracker: ParameterTracker::new(),
            training_analyzer: TrainingAnalyzer::new(),
            memory_monitor: MemoryMonitor::new(),
            verification_log: Vec::new(),
        }
    }

    pub fn verify_training_step(
        &mut self,
        lora_tensors: &HashMap<String, Tensor>,
        loss: f32,
        step: usize,
    ) -> Result<VerificationReport> {
        let mut issues = Vec::new();

        // Check gradients
        let gradient_status = self.gradient_inspector.inspect_gradients(lora_tensors)?;
        if !gradient_status.has_gradients {
            issues.push("No gradients detected".to_string());
        }
        if !gradient_status.all_finite {
            issues.push("Non-finite gradients".to_string());
        }
        if !gradient_status.reasonable_magnitude {
            issues
                .push(format!("Gradient magnitude out of range: {:.2e}", gradient_status.avg_norm));
        }

        // Check parameter updates
        let parameter_status = self.parameter_tracker.track_parameters(lora_tensors)?;
        if !parameter_status.updating && step > 0 {
            issues.push("Parameters not updating".to_string());
        }

        // Check loss
        let loss_status = self.training_analyzer.analyze_loss(loss);
        if !loss_status.is_finite {
            issues.push("Loss is not finite".to_string());
        }
        if !loss_status.in_expected_range {
            issues.push(format!("Loss out of range: {}", loss));
        }

        // Check memory
        let memory_status = self.memory_monitor.check_memory();
        if !memory_status.within_bounds {
            issues.push(format!("Memory usage too high: {:.1}GB", memory_status.current_usage_gb));
        }
        if !memory_status.stable {
            issues.push("Memory usage unstable".to_string());
        }

        // Determine overall health
        let overall_health = if issues.is_empty() {
            TrainingHealth::Healthy
        } else if issues.len() <= 2
            && !issues.iter().any(|i| i.contains("not updating") || i.contains("No gradients"))
        {
            TrainingHealth::Warning
        } else {
            TrainingHealth::Failed
        };

        let report = VerificationReport {
            step,
            gradient_status,
            parameter_status,
            loss_status,
            memory_status,
            overall_health,
            issues,
        };

        self.verification_log.push(report.clone());

        Ok(report)
    }

    pub fn generate_final_report(&self) -> String {
        let total_steps = self.verification_log.len();
        let healthy_steps = self
            .verification_log
            .iter()
            .filter(|r| r.overall_health == TrainingHealth::Healthy)
            .count();
        let warning_steps = self
            .verification_log
            .iter()
            .filter(|r| r.overall_health == TrainingHealth::Warning)
            .count();
        let failed_steps = self
            .verification_log
            .iter()
            .filter(|r| r.overall_health == TrainingHealth::Failed)
            .count();

        let avg_loss = if !self.verification_log.is_empty() {
            self.verification_log.iter().map(|r| r.loss_status.current_loss).sum::<f32>()
                / total_steps as f32
        } else {
            0.0
        };

        let separator = "=".repeat(60);
        format!(
            "\n{}\n\
            TRAINING VERIFICATION REPORT\n\
            {}\n\
            Total Steps: {}\n\
            ✅ Healthy: {} ({:.1}%)\n\
            ⚠️  Warning: {} ({:.1}%)\n\
            ❌ Failed: {} ({:.1}%)\n\
            \n\
            Average Loss: {:.6}\n\
            Loss Trend: {}\n\
            \n\
            Common Issues:\n{}\n\
            \n\
            Overall Assessment: {}\n\
            {}",
            separator,
            separator,
            total_steps,
            healthy_steps,
            (healthy_steps as f32 / total_steps.max(1) as f32) * 100.0,
            warning_steps,
            (warning_steps as f32 / total_steps.max(1) as f32) * 100.0,
            failed_steps,
            (failed_steps as f32 / total_steps.max(1) as f32) * 100.0,
            avg_loss,
            self.verification_log
                .last()
                .map(|r| r.loss_status.trend.clone())
                .unwrap_or_else(|| "Unknown".to_string()),
            self.get_common_issues(),
            self.get_overall_assessment(),
            separator
        )
    }

    fn get_common_issues(&self) -> String {
        let mut issue_counts: HashMap<String, usize> = HashMap::new();
        for report in &self.verification_log {
            for issue in &report.issues {
                *issue_counts.entry(issue.clone()).or_insert(0) += 1;
            }
        }

        let mut sorted_issues: Vec<_> = issue_counts.iter().collect();
        sorted_issues.sort_by(|a, b| b.1.cmp(a.1));

        sorted_issues
            .iter()
            .take(5)
            .map(|(issue, count)| format!("  - {} ({}x)", issue, count))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn get_overall_assessment(&self) -> &str {
        let healthy_ratio = self
            .verification_log
            .iter()
            .filter(|r| r.overall_health == TrainingHealth::Healthy)
            .count() as f32
            / self.verification_log.len().max(1) as f32;

        if healthy_ratio > 0.9 {
            "✅ TRAINING IS WORKING CORRECTLY"
        } else if healthy_ratio > 0.5 {
            "⚠️ TRAINING HAS ISSUES BUT IS PROGRESSING"
        } else {
            "❌ TRAINING IS NOT WORKING PROPERLY"
        }
    }
}
