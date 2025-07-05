# diffusers-rs

An implementation of diffusion models in Rust based on tch.

## SD 3.5 LoKr Training Support

Working SD 3.5 LoKr trainer with fixed inf/nan loss issues. Runs at 6.6+ it/s on RTX 3090 Ti.

```bash
cd eridiffusion
cargo build --release --bin trainer
./target/release/trainer ../config/eri1024.yaml
```

## Getting the weights

You can fetch the necessary weight files from huggingface's hub (or any other locations supported by the hf-hub crate). On most cases, the weight files are automatically downloaded when running an example so there is no need for any manual downloads.

E.g. in order to use the `stable-diffusion-v2-1` model, you can just run the following example command and the weight files will be downloaded automatically.

```bash
cargo run --example stable-diffusion --features clap -- --sd-version v2-1 --prompt "A very rusty robot holding a fire torch."
```

In some cases, e.g. because the license has to be accepted or the files are not public, manual steps may be needed. E.g. for [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) use an access token and do the following from the root of this repository:

```bash
hf-hub download --token YOUR_TOKEN --repo stabilityai/stable-diffusion-xl-base-1.0 --local-dir .
rm unet/diffusion_pytorch_model.safetensors.index.json text_encoder_2/model.safetensors.index.json
```

## Running the examples

### Stable Diffusion XL

```bash
cargo run --example stable-diffusion-xl --features clap -- --prompt "A rusty robot holding a fire torch."
```

### Stable Diffusion v1.5, v2.1

```bash
cargo run --example stable-diffusion --features clap -- --prompt "A rusty robot holding a fire torch."
```

The default version is 1.5, but you can specify it explicitly and also use 2.1 with `--sd-version` flag.

```bash
cargo run --example stable-diffusion --features clap -- --sd-version v1-5 --prompt "A rusty robot holding a fire torch."
cargo run --example stable-diffusion --features clap -- --sd-version v2-1 --prompt "A rusty robot holding a fire torch."
```

### Image to Image

```
cargo run --example stable-diffusion-img2img --features clap -- --prompt "A fantasy landscape, trending on artstation" --input-image media/in_img2img.jpg
```

The default strength is 0.8, but it can be specified with `--strength` flag.

### Inpainting

<table>
<thead>
<tr>
<th>Input Image</th>
<th>Inpaint Mask</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td><img src="media/in.jpg" alt="drawing" width="128"/></td>
<td><img src="media/mask.png" alt="drawing" width="128"/></td>
<td><img src="media/out.jpg" alt="drawing" width="128"/></td>
</tr>
</tbody>
</table>

```
cargo run --example stable-diffusion-inpaint --features clap -- --prompt "Face of a yellow cat, high resolution, sitting on a park bench" --input-image media/in.jpg --mask-image media/mask.png
```

### ControlNet

<table>
<thead>
<tr>
<th>Input Image</th>
<th>Control Image</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td><img src="media/hf-logo.png" alt="drawing" width="128"/></td>
<td><img src="media/canny_hf-logo.png" alt="drawing" width="128"/></td>
<td><img src="media/controlnet-out.jpg" alt="drawing" width="128"/></td>
</tr>
</tbody>
</table>

```
cargo run --example controlnet --features clap,image,imageproc -- --prompt "a rusty robot, lit by a fire torch" --input-image media/hf-logo.png
```