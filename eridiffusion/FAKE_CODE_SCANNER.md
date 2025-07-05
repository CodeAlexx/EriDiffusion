# Fake Code Scanner

This project includes multiple layers of protection against fake/mock/placeholder code:

## Automated Scanning

### 1. GitHub Actions (CI/CD)
- Automatically runs on every push and pull request
- Scans all Rust files for fake code patterns
- Fails the build if critical issues are found
- Posts a comment on PRs with detected issues

### 2. Pre-commit Hook
Enable the pre-commit hook to scan before every commit:
```bash
# Enable git hooks
git config core.hooksPath .githooks
```

### 3. Manual Scanning

#### Using Python Script
```bash
# Scan entire project
./scripts/detect_fake_code.py

# Scan specific directory
./scripts/detect_fake_code.py crates/models/src

# Output to file
./scripts/detect_fake_code.py --output scan_report.txt

# JSON output
./scripts/detect_fake_code.py --format json

# Fail on warnings
./scripts/detect_fake_code.py --fail-on-warning
```

#### Using Cargo xtask
```bash
# Build xtask
cargo build -p xtask

# Scan entire workspace
cargo xtask scan-fake-code

# Scan specific path
cargo xtask scan-fake-code --path crates/models

# Save report
cargo xtask scan-fake-code --output report.txt

# Fail on warnings
cargo xtask scan-fake-code --fail-on-warning
```

## Patterns Detected

### Critical Patterns (Build Fails)
- `todo!()` macro
- `unimplemented!()` macro  
- `panic!("not implemented")`
- Dummy byte vectors with comments
- Mock tensors with comments
- Suspicious clone returns

### Warning Patterns
- TODO/FIXME/XXX/HACK/BUG comments
- Words: placeholder, dummy, mock, fake, stub
- "not implemented" mentions
- Hardcoded values
- Simplified implementations

## Exclusions
The scanner excludes:
- Test files and modules
- Benchmark files
- Example files
- Files in target directory
- Test-related code patterns

## Integration with Development Workflow

1. **Pre-commit**: Catches issues before they enter git
2. **CI/CD**: Catches issues before merge
3. **Manual**: Run anytime during development

## Customization

Edit patterns in:
- `.github/workflows/fake-code-detector.yml` - CI patterns
- `.githooks/pre-commit` - Pre-commit patterns
- `scripts/detect_fake_code.py` - Python scanner patterns
- `xtask/src/fake_code_scan.rs` - Rust scanner patterns

## Bypassing (NOT RECOMMENDED)

- Skip pre-commit: `git commit --no-verify`
- Skip CI: Add `[skip ci]` to commit message

**WARNING**: Bypassing these checks means fake code enters the codebase!