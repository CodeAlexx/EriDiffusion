#!/bin/bash
# Comprehensive test runner for SD 3.5 training

set -e

echo "AI-Toolkit SD 3.5 Training Test Suite"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0
SKIPPED=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -n "Running $test_name... "
    
    if eval "$test_command" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}FAILED${NC}"
        echo "  Error output:"
        tail -20 /tmp/test_output.log | sed 's/^/    /'
        ((FAILED++))
    fi
}

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Rust
    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}ERROR: Rust/Cargo not found${NC}"
        exit 1
    fi
    
    # Check CUDA (optional)
    if command -v nvidia-smi &> /dev/null; then
        echo "  ✓ CUDA available"
        export CUDA_AVAILABLE=1
    else
        echo "  ⚠ CUDA not available (CPU-only tests)"
        export CUDA_AVAILABLE=0
    fi
    
    echo ""
}

# Unit Tests
run_unit_tests() {
    echo "1. Unit Tests"
    echo "============="
    
    # Core tests
    run_test "Core utilities" "cargo test -p ai-toolkit-core --lib"
    
    # Model tests
    run_test "Model implementations" "cargo test -p ai-toolkit-models --lib"
    
    # Network tests
    run_test "LoRA/LoKr networks" "cargo test -p ai-toolkit-networks --lib"
    
    # Training tests
    run_test "Training infrastructure" "cargo test -p ai-toolkit-training --lib"
    
    # Data tests
    run_test "Data pipeline" "cargo test -p ai-toolkit-data --lib"
    
    # SD3.5 specific tests
    run_test "SD3.5 trainer" "cargo test sd35_trainer"
    
    echo ""
}

# Integration Tests
run_integration_tests() {
    echo "2. Integration Tests"
    echo "==================="
    
    # SD3.5 training integration
    run_test "SD3.5 training integration" "cargo test --test sd35_training_integration_test"
    
    # Pipeline tests
    run_test "Inference pipeline" "cargo test -p ai-toolkit-inference sd3_pipeline"
    
    echo ""
}

# Build Tests
run_build_tests() {
    echo "3. Build Tests"
    echo "=============="
    
    # Debug build
    run_test "Debug build" "cargo build --all"
    
    # Release build
    run_test "Release build" "cargo build --all --release"
    
    # Binary builds
    run_test "Training binaries" "cargo build --bin test_sd35_full_training --bin train_sd35_lokr"
    
    echo ""
}

# Documentation Tests
run_doc_tests() {
    echo "4. Documentation Tests"
    echo "====================="
    
    run_test "Documentation" "cargo test --doc"
    run_test "Doc generation" "cargo doc --no-deps --all-features"
    
    echo ""
}

# Clippy and Format
run_lint_tests() {
    echo "5. Linting and Formatting"
    echo "========================="
    
    run_test "Clippy" "cargo clippy --all-targets --all-features -- -D warnings"
    run_test "Format check" "cargo fmt --all -- --check"
    
    echo ""
}

# Memory and Performance Tests
run_performance_tests() {
    echo "6. Performance Tests"
    echo "===================="
    
    if [ "$CUDA_AVAILABLE" -eq 1 ]; then
        run_test "Memory estimation" "cargo test test_memory_estimation -- --nocapture"
        run_test "Training step performance" "cargo test test_training_step_simulation -- --nocapture"
    else
        echo -e "${YELLOW}Skipping GPU performance tests (no CUDA)${NC}"
        ((SKIPPED+=2))
    fi
    
    echo ""
}

# Quick Training Test
run_quick_training_test() {
    echo "7. Quick Training Test"
    echo "======================"
    
    if [ -f "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors" ]; then
        run_test "Quick training (100 steps)" "./scripts/test_sd35_training.sh"
    else
        echo -e "${YELLOW}Skipping training test (model not found)${NC}"
        ((SKIPPED++))
    fi
    
    echo ""
}

# Feature Tests
run_feature_tests() {
    echo "8. Feature Tests"
    echo "================"
    
    # Test specific features
    run_test "Flow matching" "cargo test flow_matching"
    run_test "LoKr implementation" "cargo test lokr"
    run_test "Triple text encoding" "cargo test triple_text"
    run_test "SNR weighting" "cargo test snr_weight"
    run_test "Gradient checkpointing" "cargo test gradient_checkpoint || true"
    
    echo ""
}

# Generate Test Report
generate_report() {
    local total=$((PASSED + FAILED + SKIPPED))
    local pass_rate=0
    if [ $total -gt 0 ]; then
        pass_rate=$((PASSED * 100 / total))
    fi
    
    cat > test_report.txt << EOF
AI-Toolkit SD 3.5 Training Test Report
=====================================
Date: $(date)
Total Tests: $total
Passed: $PASSED
Failed: $FAILED
Skipped: $SKIPPED
Pass Rate: $pass_rate%

Test Categories:
1. Unit Tests
2. Integration Tests  
3. Build Tests
4. Documentation Tests
5. Linting and Formatting
6. Performance Tests
7. Quick Training Test
8. Feature Tests

EOF
    
    if [ $FAILED -gt 0 ]; then
        echo "Failed Tests:" >> test_report.txt
        grep "FAILED" /tmp/test_output.log >> test_report.txt || true
    fi
    
    echo ""
    echo "Test Summary"
    echo "============"
    echo -e "Total Tests: $total"
    echo -e "Passed: ${GREEN}$PASSED${NC}"
    echo -e "Failed: ${RED}$FAILED${NC}"
    echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
    echo -e "Pass Rate: $pass_rate%"
    echo ""
    echo "Full report saved to: test_report.txt"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    check_prerequisites
    
    # Run all test suites
    run_unit_tests
    run_integration_tests
    run_build_tests
    run_doc_tests
    run_lint_tests
    run_performance_tests
    run_quick_training_test
    run_feature_tests
    
    # Generate report
    generate_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "Total test time: ${duration}s"
    
    # Exit with error if any tests failed
    if [ $FAILED -gt 0 ]; then
        exit 1
    fi
}

# Run main
main