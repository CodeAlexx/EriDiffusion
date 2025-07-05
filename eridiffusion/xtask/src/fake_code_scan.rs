//! Fake code detection module

use anyhow::{Context, Result};
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct Issue {
    pub file: PathBuf,
    pub line: usize,
    pub pattern: String,
    pub content: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    Critical,
    Warning,
}

pub struct FakeCodeScanner {
    critical_patterns: Vec<(Regex, &'static str)>,
    warning_patterns: Vec<(Regex, &'static str)>,
    exclude_patterns: Vec<Regex>,
}

impl FakeCodeScanner {
    pub fn new() -> Result<Self> {
        // Critical patterns
        let critical_patterns = vec![
            (Regex::new(r"\btodo!\s*\(")?, "todo! macro"),
            (Regex::new(r"\bunimplemented!\s*\(")?, "unimplemented! macro"),
            (Regex::new(r#"panic!\s*\(\s*"not implemented"#)?, "panic! with not implemented"),
            (Regex::new(r"vec!\s*\[\s*0u8\s*;\s*\d+\s*\].*//.*(?:placeholder|dummy|mock)")?, "dummy byte vector"),
            (Regex::new(r"Tensor::zeros.*//.*(?:dummy|mock|placeholder)")?, "dummy tensor"),
            (Regex::new(r"Tensor::randn.*//.*(?:mock|fake|placeholder)")?, "mock tensor"),
            (Regex::new(r"return\s+Ok\s*\(\s*[^)]*\.clone\s*\(\s*\)\s*\).*//.*(?:placeholder|dummy|mock|fake)")?, "suspicious clone with fake comment"),
        ];
        
        // Warning patterns
        let warning_patterns = vec![
            (Regex::new(r"\bTODO\b")?, "TODO comment"),
            (Regex::new(r"\bFIXME\b")?, "FIXME comment"),
            (Regex::new(r"\bXXX\b")?, "XXX comment"),
            (Regex::new(r"\bHACK\b")?, "HACK comment"),
            (Regex::new(r"\bBUG\b")?, "BUG comment"),
            (Regex::new(r"(?i)\bplaceholder\b")?, "placeholder"),
            (Regex::new(r"(?i)\bdummy\b")?, "dummy"),
            (Regex::new(r"(?i)\bmock\b")?, "mock"),
            (Regex::new(r"(?i)\bfake\b")?, "fake"),
        ];
        
        // Exclude patterns
        let exclude_patterns = vec![
            Regex::new(r"#\[cfg\(test\)\]")?,
            Regex::new(r"#\[test\]")?,
            Regex::new(r"mod\s+tests\s*\{")?,
            Regex::new(r"MockDataset")?,
            Regex::new(r"test_")?,
        ];
        
        Ok(Self {
            critical_patterns,
            warning_patterns,
            exclude_patterns,
        })
    }
    
    pub fn scan_file(&self, path: &Path) -> Result<Vec<Issue>> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        
        let mut issues = Vec::new();
        
        for (line_num, line) in content.lines().enumerate() {
            // Skip excluded patterns
            if self.exclude_patterns.iter().any(|p| p.is_match(line)) {
                continue;
            }
            
            // Check critical patterns
            for (pattern, description) in &self.critical_patterns {
                if pattern.is_match(line) {
                    issues.push(Issue {
                        file: path.to_path_buf(),
                        line: line_num + 1,
                        pattern: description.to_string(),
                        content: line.trim().to_string(),
                        severity: Severity::Critical,
                    });
                }
            }
            
            // Check warning patterns
            for (pattern, description) in &self.warning_patterns {
                if pattern.is_match(line) {
                    issues.push(Issue {
                        file: path.to_path_buf(),
                        line: line_num + 1,
                        pattern: description.to_string(),
                        content: line.trim().to_string(),
                        severity: Severity::Warning,
                    });
                }
            }
        }
        
        Ok(issues)
    }
    
    pub fn scan_directory(&self, dir: &Path) -> Result<Vec<Issue>> {
        let mut all_issues = Vec::new();
        
        for entry in WalkDir::new(dir)
            .into_iter()
            .filter_entry(|e| !Self::should_skip(e.path()))
        {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().map_or(false, |ext| ext == "rs") {
                let issues = self.scan_file(path)?;
                all_issues.extend(issues);
            }
        }
        
        Ok(all_issues)
    }
    
    fn should_skip(path: &Path) -> bool {
        let skip_dirs = ["target", ".git", "tests", "benches", "examples"];
        
        path.components().any(|c| {
            c.as_os_str().to_str()
                .map_or(false, |s| skip_dirs.contains(&s))
        })
    }
    
    pub fn generate_report(&self, issues: &[Issue]) -> String {
        let mut report = String::new();
        
        let critical: Vec<_> = issues.iter()
            .filter(|i| i.severity == Severity::Critical)
            .collect();
        let warnings: Vec<_> = issues.iter()
            .filter(|i| i.severity == Severity::Warning)
            .collect();
        
        report.push_str("FAKE CODE DETECTION REPORT\n");
        report.push_str("==========================\n\n");
        
        if !critical.is_empty() {
            report.push_str(&format!("❌ CRITICAL ISSUES ({})\n", critical.len()));
            report.push_str("----------------------\n");
            for issue in &critical {
                report.push_str(&format!(
                    "{}:{}\n  Pattern: {}\n  Content: {}\n\n",
                    issue.file.display(),
                    issue.line,
                    issue.pattern,
                    issue.content
                ));
            }
        }
        
        if !warnings.is_empty() {
            report.push_str(&format!("⚠️  WARNINGS ({})\n", warnings.len()));
            report.push_str("----------------\n");
            for issue in warnings.iter().take(10) {
                report.push_str(&format!(
                    "{}:{}\n  Pattern: {}\n  Content: {}\n\n",
                    issue.file.display(),
                    issue.line,
                    issue.pattern,
                    issue.content
                ));
            }
            
            if warnings.len() > 10 {
                report.push_str(&format!("... and {} more warnings\n", warnings.len() - 10));
            }
        }
        
        if issues.is_empty() {
            report.push_str("✅ No fake code patterns detected!\n");
        } else {
            report.push_str(&format!(
                "\nTotal issues: {}\nCritical: {}\nWarnings: {}\n",
                issues.len(),
                critical.len(),
                warnings.len()
            ));
        }
        
        report
    }
}