## SDXL Conditioning Overview

This module builds the classic 2816-dimensional SDXL conditioning vector:

```
cond_2816 = concat([pooled_text_1280], [time_embed_1536])  // [N, 2816]
```

Where:

- `pooled_text_1280` comes from the CLIP-G pooled output (before the checkpoint `label_emb` MLP).
- `time_embed_1536` is a Diffusers-style sinusoidal embedding of timesteps (no learned weights).

In addition, the crate provides helpers to generate SDXL `time_ids` vectors and preprocess raw timesteps.

### Modules

- `timestep_helpers.rs`
  - `cast_to_f32` – convert integer timesteps to F32 on device.
  - `clamp_f32` – restrict timesteps to a safe range.
  - `preprocess_timesteps` – cast + clamp in one call.
- `timestep_embedding.rs`
  - `timestep_embedding` – build `[N, dim]` sinusoidal embeddings (dim even).
  - `sigma_to_timestep_vec` – convenience when working with sigma schedules.
- `time_ids.rs`
  - `build_time_ids` – construct `[N,6] = [orig_h, orig_w, crop_y, crop_x, target_h, target_w]` vectors.
- `mod.rs`
  - `sdxl_cond_2816_from_raw_ts` – recommended entry point; accepts pooled CLIP embeddings + raw timesteps (ints/floats), clamps timesteps, and produces `[N,2816]`.
  - `sdxl_cond_2816_from_sin` – legacy helper for callers that already maintain clamped F32 timesteps.

### Minimal Usage

```rust
use eridiffusion_training::conditioning::{
    sdxl_cond_2816_from_raw_ts,
    time_ids::build_time_ids,
};

let pooled_text = pooled_clip_g; // [N,1280]

let raw_ts = Tensor::from_vec_dtype(vec![0i32, 500, 1000], Shape::from_dims(&[3]), device.cuda_device_arc(), DType::I32)?;
let cond2816 = sdxl_cond_2816_from_raw_ts(&pooled_text, &raw_ts, 0.0, 1000.0, 10_000.0)?; // [N,2816]

// Apply the checkpoint label embedding MLP to obtain the 1280-d adapter input
let cond1280 = label_emb.forward_from_2816(&cond2816)?; // [N,1280]

let time_ids = build_time_ids(3, 1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0, &training_device)?; // [N,6]
```

Keep tensor storage in BF16 where possible, and only cast to F32 for math-heavy operations (LayerNorm, matmul, softmax). Save/restore conditioning tensors once per batch to avoid repeated host→device transfers.

## Feature toggle

Cargo features let you flip between the sinusoidal and time-IDs MLP paths without touching code:

```
[features]
default = ["cond_sinusoidal"]
cond_sinusoidal = []
cond_time_ids_mlp = []
```

Use the unified API:

```rust
use eridiffusion_training::conditioning::feature_toggle::{make_conditioning, CondArgs};

let cond = make_conditioning(CondArgs::sinusoidal(&pooled_1280, &raw_ts, 0.0, 1000.0, 10_000.0))?;
// or for the time-IDs path:
// let cond = make_conditioning(CondArgs::time_ids(
//     &pooled_1280,
//     &pooled_1280,
//     &time_ids,
//     &|ids| time_mlp.forward(ids),
//     0.0,
//     1000.0,
//     10_000.0,
//     Some(&raw_ts),
// ))?;
```

Build commands:
- Sinusoidal (default):
  `cargo run -p eridiffusion-training --features cond_sinusoidal --bin <trainer>`
- Time-IDs MLP:
  `cargo run -p eridiffusion-training --no-default-features --features cond_time_ids_mlp --bin <trainer>`

If both features are enabled, `cond_time_ids_mlp` wins by default; adjust the precedence in `conditioning/feature_toggle.rs` if you prefer the opposite.
