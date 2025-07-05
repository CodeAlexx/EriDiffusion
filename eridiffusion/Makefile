.PHONY: build test clean doc fmt lint install run-example

# Build the project
build:
	cargo build --release

# Run tests
test:
	cargo test

# Clean build artifacts
clean:
	cargo clean

# Generate documentation
doc:
	cargo doc --no-deps --open

# Format code
fmt:
	cargo fmt

# Run linter
lint:
	cargo clippy -- -D warnings

# Install the CLI tool
install:
	cargo install --path .

# Run example training
run-example:
	./examples/train_lora.sh

# Run all checks before committing
pre-commit: fmt lint test
	@echo "All checks passed!"

# Build for all targets
build-all:
	cargo build --release --target x86_64-unknown-linux-gnu
	cargo build --release --target x86_64-apple-darwin
	cargo build --release --target x86_64-pc-windows-gnu

# Run benchmarks
bench:
	cargo bench

# Check for outdated dependencies
outdated:
	cargo outdated

# Update dependencies
update:
	cargo update

# Run security audit
audit:
	cargo audit