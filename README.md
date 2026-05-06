# EriDiffusion v2

Pure-Rust diffusion-model training framework on top of [flame-core](https://github.com/CodeAlexx/Flame).
Modeled after OneTrainer's pipeline; no Python at runtime.

## Status

Active. Companion repos:

- [`flame-core`](https://github.com/CodeAlexx/Flame) — tensor library + CUDA kernels
- [`inference-flame`](https://github.com/CodeAlexx/inference-flame) — inference-only counterpart

## Trainers

| Model | `crates/eridiffusion-cli/src/bin/` | Status |
| --- | --- | --- |
| Z-Image | `train_zimage`, `prepare_zimage`, `sample_zimage` | tested |
| ERNIE-Image | `train_ernie`, `prepare_ernie`, `sample_ernie` | tested |
| Klein (FLUX.2) | `train_klein`, `prepare_klein`, `sample_klein` | works |
| FLUX.1 | `train_flux`, `prepare_flux`, `sample_flux` | works |
| SD3.5 Medium | `train_sd35`, `prepare_sd35`, `sample_sd35` | works |
| SDXL | `train_sdxl`, `prepare_sdxl`, `sample_sdxl` | works |
| Anima (Cosmos-Predict2 + LLM-Adapter) | `train_anima`, `prepare_anima`, `sample_anima` | rank-32 smoke clean |
| Qwen-Image-2512 | `train_qwenimage`, `prepare_qwenimage`, `sample_qwenimage` | end-to-end |
| LTX-2 | `train_ltx2`, `prepare_ltx2`, `sample_ltx2` | works |
| ACE-Step | `train_acestep` | model port; needs Python prep tensors |
| Chroma | model + sampler ported, trainer binary on demand |
| Wan 2.x | blocked — needs `inference_flame::wan22_dit` lifted |

## Build

```bash
cargo build --release
```

Per-binary:

```bash
cargo build --release --bin train_qwenimage
```

## Run

Each trainer is a CLI binary. Example (Qwen-Image LoRA on 24 GB):

```bash
target/release/prepare_qwenimage \
  --input-dir <image+caption dir> \
  --output-dir cache/<run> \
  --vae-ckpt <qwen_image_vae.safetensors> \
  --text-encoder <qwen-image-2512/text_encoder/> \
  --tokenizer-path <qwen-image-2512/tokenizer/tokenizer.json> \
  --resolution 512

target/release/train_qwenimage \
  --model <qwen-image-2512/transformer/> \
  --cache-dir cache/<run> \
  --steps 3000 --rank 16 --lr 3e-4 --warmup-steps 200 \
  --save-every 500 --sample-every 500 \
  --sample-prompt "..." --sample-vae <vae> \
  --sample-text-encoder <text_encoder/> --sample-tokenizer <tokenizer.json> \
  --output-dir output/<run>
```

`prepare_*` and `sample_*` arguments mirror the trainer.

## Architecture

- `crates/eridiffusion-core/` — model forwards, encoders, samplers, training infra
  - `models/` — per-model DiT/UNet
  - `encoders/` — VAE encoders/decoders, text encoders (Qwen3, Qwen2.5-VL, Mistral-3B, Gemma3, T5-XXL, CLIP-L/G)
  - `sampler/` — flow-matching schedules + Euler/CFG denoise
  - `training/` — `BlockOffloader`, activation offload pool, checkpoint save/resume, EMA, schedule, logging
  - `data/` — bucketed latent dataset
  - `lora/` — LoRA wrapper
- `crates/eridiffusion-cli/` — `prepare_*`, `train_*`, `sample_*` binaries

## License

MIT
