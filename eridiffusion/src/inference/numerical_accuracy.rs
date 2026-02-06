use flame_core::device::Device;
use flame_core::GradientMap;
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
// backward is a method on Tensor, not a standalone function
use anyhow::bail;
use flame_core::optimizers::{Adam, SGD};
use std::{fmt::Write, fs::File, io::Write as IoWrite};

/// Numerical comparison results
#[derive(Debug, Clone)]
pub struct AccuracyReport {
    pub name: String,
    pub max_absolute_error: f32,
    pub mean_absolute_error: f32,
    pub max_relative_error: f32,
    pub mean_relative_error: f32,
    pub num_elements: usize,
    pub tolerance: f32,
    pub passed: bool,
}

/// Collection of accuracy reports
#[derive(Default)]
pub struct AccuracyTestSuite {
    reports: Vec<AccuracyReport>,
}

/// Statistical analysis of differences
pub struct DifferenceStats {
    pub num_zeros: usize,
    pub num_small: usize,  // < 1e-6,
    pub num_medium: usize, // 1e-6 to 1e-3,
    pub num_large: usize,  // > 1e-3,
    pub percentile_95: f32,
    pub percentile_99: f32,
}

// Numerical accuracy testing and reporting utilities
//
// This module provides tools for comparing numerical outputs between
// FLAME and FLAME implementations.

// Note: TensorExt trait removed as it's already defined in other files

impl AccuracyReport {
    /// Create a new accuracy report by comparing two vectors
    pub fn from_vectors(
        name: &str,
        expected: &[f32],
        actual: &[f32],
        tolerance: f32,
    ) -> flame_core::Result<Self> {
        if expected.len() != actual.len() {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Vector length mismatch for {}: expected {} vs actual {}",
                name,
                expected.len(),
                actual.len()
            )));
        }

        let num_elements = expected.len();
        let mut max_abs_error = 0.0f32;
        let mut sum_abs_error = 0.0f32;
        let mut max_rel_error = 0.0f32;
        let mut sum_rel_error = 0.0f32;
        let mut rel_error_count = 0;

        for (exp, act) in expected.iter().zip(actual.iter()) {
            let abs_error = (exp - act).abs();
            max_abs_error = max_abs_error.max(abs_error);
            sum_abs_error += abs_error;

            // Relative error (skip if expected is near zero)
            if exp.abs() > 1e-8 {
                let rel_error = abs_error / exp.abs();
                max_rel_error = max_rel_error.max(rel_error);
                sum_rel_error += rel_error;
                rel_error_count += 1;
            }
        }

        let mean_abs_error = sum_abs_error / num_elements as f32;
        let mean_rel_error =
            if rel_error_count > 0 { sum_rel_error / rel_error_count as f32 } else { 0.0 };

        let passed = max_abs_error <= tolerance;

        Ok(AccuracyReport {
            name: name.to_string(),
            max_absolute_error: max_abs_error,
            mean_absolute_error: mean_abs_error,
            max_relative_error: max_rel_error,
            mean_relative_error: mean_rel_error,
            num_elements,
            tolerance,
            passed,
        })
    }

    /// Generate a formatted report string
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        writeln!(&mut report, "\n{}", "=".repeat(60)).unwrap();
        writeln!(&mut report, "Accuracy Report: {}", self.name).unwrap();
        writeln!(&mut report, "{}", "=".repeat(60)).unwrap();

        writeln!(&mut report, "Elements compared: {}", self.num_elements).unwrap();
        writeln!(&mut report, "Tolerance: {:.2e}", self.tolerance).unwrap();
        writeln!(&mut report, "\nAbsolute Error:").unwrap();
        writeln!(&mut report, "  Maximum: {:.6e}", self.max_absolute_error).unwrap();
        writeln!(&mut report, "  Mean:    {:.6e}", self.mean_absolute_error).unwrap();

        writeln!(&mut report, "\nRelative Error:").unwrap();
        writeln!(&mut report, "  Maximum: {:.2}%", self.max_relative_error * 100.0).unwrap();
        writeln!(&mut report, "  Mean:    {:.2}%", self.mean_relative_error * 100.0).unwrap();

        writeln!(&mut report, "\nStatus: {}", if self.passed { "✓ PASSED" } else { "✗ FAILED" })
            .unwrap();

        if !self.passed {
            writeln!(
                &mut report,
                "Max error {:.2e} exceeds tolerance {:.2e}",
                self.max_absolute_error, self.tolerance
            )
            .unwrap();
        }

        report
    }
}

impl AccuracyTestSuite {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_report(&mut self, report: AccuracyReport) {
        self.reports.push(report);
    }

    pub fn add_comparison(
        &mut self,
        name: &str,
        expected: &[f32],
        actual: &[f32],
        tolerance: f32,
    ) -> flame_core::Result<()> {
        let report = AccuracyReport::from_vectors(name, expected, actual, tolerance)?;
        self.add_report(report);
        Ok(())
    }

    pub fn all_passed(&self) -> bool {
        self.reports.iter().all(|r| r.passed)
    }

    pub fn num_passed(&self) -> usize {
        self.reports.iter().filter(|r| r.passed).count()
    }

    pub fn num_failed(&self) -> usize {
        self.reports.iter().filter(|r| !r.passed).count()
    }

    pub fn generate_summary(&self) -> String {
        let mut summary = String::new();

        writeln!(&mut summary, "\n{}", "=".repeat(80)).unwrap();
        writeln!(&mut summary, "NUMERICAL ACCURACY TEST SUMMARY").unwrap();
        writeln!(&mut summary, "{}", "=".repeat(80)).unwrap();

        writeln!(&mut summary, "\nTotal Tests: {}", self.reports.len()).unwrap();
        writeln!(&mut summary, "Passed: {} ✓", self.num_passed()).unwrap();
        writeln!(&mut summary, "Failed: {} ✗", self.num_failed()).unwrap();

        if !self.reports.is_empty() {
            writeln!(
                &mut summary,
                "\n{:<40} {:>15} {:>15} {:>10}",
                "Test Name", "Max Abs Error", "Mean Abs Error", "Status"
            )
            .unwrap();
            writeln!(&mut summary, "{}", "-".repeat(80)).unwrap();

            for report in &self.reports {
                writeln!(
                    &mut summary,
                    "{:<40} {:>15.2e} {:>15.2e} {:>10}",
                    truncate_string(&report.name, 40),
                    report.max_absolute_error,
                    report.mean_absolute_error,
                    if report.passed { "✓" } else { "✗" }
                )
                .unwrap();
            }
        }

        if self.all_passed() {
            writeln!(&mut summary, "\n🎉 All tests passed!").unwrap();
        } else {
            writeln!(&mut summary, "\n⚠️  Some tests failed. See individual reports for details.")
                .unwrap();
        }

        summary
    }

    pub fn save_to_file(&self, path: &str) -> flame_core::Result<()> {
        let mut file =
            File::create(path).map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        // Write summary
        writeln!(file, "{}", self.generate_summary())
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        // Write individual reports
        for report in &self.reports {
            writeln!(file, "{}", report.format_report())
                .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        }

        Ok(())
    }
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Helper to compare tensor shapes
pub fn compare_shapes(name: &str, shape1: &[usize], shape2: &[usize]) -> flame_core::Result<()> {
    if shape1 != shape2 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Shape mismatch for {}: {:?} vs {:?}",
            name, shape1, shape2
        )));
    }
    Ok(())
}

impl DifferenceStats {
    pub fn from_differences(diffs: &[f32]) -> Self {
        let mut sorted_diffs: Vec<f32> = diffs.to_vec();
        sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let num_zeros = diffs.iter().filter(|&&d| d == 0.0).count();
        let num_small = diffs.iter().filter(|&&d| d > 0.0 && d < 1e-6).count();
        let num_medium = diffs.iter().filter(|&&d| d >= 1e-6 && d < 1e-3).count();
        let num_large = diffs.iter().filter(|&&d| d >= 1e-3).count();

        let idx_95 = ((diffs.len() as f32 * 0.95) as usize).min(diffs.len() - 1);
        let idx_99 = ((diffs.len() as f32 * 0.99) as usize).min(diffs.len() - 1);

        Self {
            num_zeros,
            num_small,
            num_medium,
            num_large,
            percentile_95: sorted_diffs[idx_95],
            percentile_99: sorted_diffs[idx_99],
        }
    }

    pub fn format_report(&self, total: usize) -> String {
        let mut report = String::new();

        writeln!(&mut report, "\nDifference Distribution:").unwrap();
        writeln!(
            &mut report,
            "  Exact matches (0):     {:6} ({:5.1}%)",
            self.num_zeros,
            100.0 * self.num_zeros as f32 / total as f32
        )
        .unwrap();
        writeln!(
            &mut report,
            "  Very small (<1e-6):    {:6} ({:5.1}%)",
            self.num_small,
            100.0 * self.num_small as f32 / total as f32
        )
        .unwrap();
        writeln!(
            &mut report,
            "  Small (1e-6 to 1e-3): {:6} ({:5.1}%)",
            self.num_medium,
            100.0 * self.num_medium as f32 / total as f32
        )
        .unwrap();
        writeln!(
            &mut report,
            "  Large (>1e-3):         {:6} ({:5.1}%)",
            self.num_large,
            100.0 * self.num_large as f32 / total as f32
        )
        .unwrap();
        writeln!(&mut report, "\n95th percentile: {:.2e}", self.percentile_95).unwrap();
        writeln!(&mut report, "99th percentile: {:.2e}", self.percentile_99).unwrap();

        report
    }
}
