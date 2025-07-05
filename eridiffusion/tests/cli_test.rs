//! CLI integration tests

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("eridiffusion").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("AI-Toolkit: Production-ready diffusion model training and inference"));
}

#[test]
fn test_cli_version() {
    let mut cmd = Command::cargo_bin("eridiffusion").unwrap();
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("eridiffusion"));
}

#[test]
fn test_list_command() {
    let mut cmd = Command::cargo_bin("eridiffusion").unwrap();
    cmd.arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("SD15"))
        .stdout(predicate::str::contains("SDXL"))
        .stdout(predicate::str::contains("Flux"));
}

#[test]
fn test_list_detailed() {
    let mut cmd = Command::cargo_bin("eridiffusion").unwrap();
    cmd.args(&["list", "--detailed"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Default steps"))
        .stdout(predicate::str::contains("Latent channels"));
}

#[test]
fn test_train_missing_config() {
    let mut cmd = Command::cargo_bin("eridiffusion").unwrap();
    cmd.arg("train")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn test_generate_missing_args() {
    let mut cmd = Command::cargo_bin("eridiffusion").unwrap();
    cmd.arg("generate")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn test_plugin_list() {
    let mut cmd = Command::cargo_bin("eridiffusion").unwrap();
    cmd.args(&["plugin", "list"])
        .assert()
        .success();
}

#[test]
fn test_verbose_flag() {
    let mut cmd = Command::cargo_bin("eridiffusion").unwrap();
    cmd.args(&["-vvv", "list"])
        .assert()
        .success();
}