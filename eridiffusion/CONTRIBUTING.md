# Contributing to AI-Toolkit-RS

Thank you for your interest in contributing to AI-Toolkit-RS! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming environment for all contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ai-toolkit-rs.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `cargo test`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository
git clone https://github.com/ai-toolkit-rs/ai-toolkit.git
cd ai-toolkit-rs

# Build the project
cargo build

# Run tests
cargo test

# Run with verbose logging
RUST_LOG=debug cargo run -- --help
```

## Project Structure

- `crates/core/` - Core traits and utilities
- `crates/models/` - Model implementations
- `crates/networks/` - LoRA and adapter implementations
- `crates/training/` - Training infrastructure
- `crates/data/` - Data loading and processing
- `crates/inference/` - Inference engine
- `crates/web/` - Web UI
- `crates/extensions/` - Plugin system
- `src/` - CLI application
- `examples/` - Example configurations and scripts
- `tests/` - Integration tests

## Coding Standards

### Rust Style

- Follow standard Rust naming conventions
- Use `rustfmt` for formatting: `cargo fmt`
- Use `clippy` for linting: `cargo clippy`
- Add documentation comments for public APIs
- Write unit tests for new functionality

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 50 characters
- Add detailed description if needed

Example:
```
Add LoRA rank adaptation algorithm

- Implement adaptive rank calculation based on weight importance
- Add configuration options for min/max rank
- Include unit tests and benchmarks
```

## Testing

### Unit Tests

Place unit tests in the same file as the code being tested:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // Test implementation
    }
}
```

### Integration Tests

Place integration tests in the `tests/` directory.

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
```

## Documentation

- Add doc comments to all public APIs
- Include examples in doc comments
- Update README.md for significant features
- Add configuration examples for new features

Example:
```rust
/// Applies LoRA adaptation to a model
///
/// # Arguments
/// * `model` - The base model to adapt
/// * `config` - LoRA configuration
///
/// # Example
/// ```
/// let lora = LoRA::new(config);
/// let adapted_model = lora.apply(model)?;
/// ```
pub fn apply_lora(model: &Model, config: LoRAConfig) -> Result<Model> {
    // Implementation
}
```

## Performance Considerations

- Use zero-copy operations where possible
- Minimize allocations in hot paths
- Use async/await for I/O operations
- Profile performance-critical code
- Add benchmarks for new algorithms

## Adding New Features

### New Model Architecture

1. Add model definition in `crates/models/src/architectures/`
2. Implement the `DiffusionModel` trait
3. Add to model registry
4. Add tests and examples
5. Update documentation

### New Network Adapter

1. Add adapter in `crates/networks/src/`
2. Implement the `NetworkAdapter` trait
3. Add configuration options
4. Add tests and benchmarks
5. Update README with usage example

### New Training Feature

1. Add feature in `crates/training/src/`
2. Integrate with `Trainer` class
3. Add configuration options
4. Add tests
5. Update training examples

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add tests for new functionality
4. Ensure code follows style guidelines
5. Update CHANGELOG.md
6. Request review from maintainers

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Suggestions for improvements

Thank you for contributing!