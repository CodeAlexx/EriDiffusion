mod fake_code_scan;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Development tasks for eridiffusion-rs")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan for fake code patterns
    ScanFakeCode {
        /// Path to scan (default: workspace root)
        #[arg(short, long)]
        path: Option<PathBuf>,
        
        /// Fail on warnings
        #[arg(long)]
        fail_on_warning: bool,
        
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::ScanFakeCode { path, fail_on_warning, output } => {
            scan_fake_code(path, fail_on_warning, output)
        }
    }
}

fn scan_fake_code(
    path: Option<PathBuf>,
    fail_on_warning: bool,
    output: Option<PathBuf>,
) -> Result<()> {
    let scanner = fake_code_scan::FakeCodeScanner::new()?;
    
    let scan_path = path.unwrap_or_else(|| {
        std::env::current_dir()
            .expect("Failed to get current directory")
    });
    
    println!("🔍 Scanning {} for fake code patterns...", scan_path.display());
    
    let issues = if scan_path.is_file() {
        scanner.scan_file(&scan_path)?
    } else {
        scanner.scan_directory(&scan_path)?
    };
    
    let report = scanner.generate_report(&issues);
    
    // Write report
    if let Some(output_path) = output {
        std::fs::write(&output_path, &report)?;
        println!("Report written to {}", output_path.display());
    } else {
        println!("{}", report);
    }
    
    // Determine exit code
    let critical_count = issues.iter()
        .filter(|i| i.severity == fake_code_scan::Severity::Critical)
        .count();
    let warning_count = issues.iter()
        .filter(|i| i.severity == fake_code_scan::Severity::Warning)
        .count();
    
    if critical_count > 0 {
        std::process::exit(2);
    } else if warning_count > 0 && fail_on_warning {
        std::process::exit(1);
    }
    
    Ok(())
}