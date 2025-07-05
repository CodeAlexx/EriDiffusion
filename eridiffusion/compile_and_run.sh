#!/bin/bash
set -e

echo "🚀 Compiling SD 3.5 LoKr Trainer..."

# Create a Cargo.toml just for this binary
cat > Cargo.toml.tmp <<EOF
[package]
name = "sd35-lokr-trainer"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "train_sd35_lokr"
path = "train_sd35_lokr_working.rs"

[dependencies]
candle-core = { version = "0.9", features = ["cuda"] }
candle-nn = "0.9"
safetensors = "0.4"
EOF

# Move the working file to be part of the temporary project
mkdir -p src/bin
cp train_sd35_lokr_working.rs src/bin/

# Create minimal Cargo.toml
mv Cargo.toml.tmp Cargo.toml

# Build with cargo
echo "Building with Cargo..."
cargo build --release --bin train_sd35_lokr

echo "✅ Build successful!"
echo "🏃 Starting training..."
echo

# Run the trainer
./target/release/train_sd35_lokr