#!/bin/bash
# Create test dataset using our Rust image generator

echo "Building and running test dataset creator..."

# Build the Rust binary
cargo build --release --bin create_test_dataset

# Run it
./target/release/create_test_dataset