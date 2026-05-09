//! Read image / video files into BF16 tensors `[1, 3, F, H, W]` in `[-1, 1]`.
//!
//! ## Why ffmpeg out-of-process
//! libavcodec FFI bindings (`ffmpeg-next`, `av-codec-rs`) drag a heavy C
//! dep tree and are awkward to keep building across distros. ffmpeg as a
//! subprocess is universal, well-tested, and isolates frame-decode from
//! the trainer process (any codec quirks fail in ffmpeg, not in our
//! tape-recording code path).
//!
//! ## Strategy
//! - Image inputs (`.jpg`/`.png`/`.webp`/`.bmp`): use the `image` crate.
//! - Video inputs (`.mp4`/`.webm`/`.mov`): shell `ffmpeg` to extract the
//!   first `--num-frames` frames at a target FPS as `rgb24` raw bytes
//!   over stdout, then reshape to `[3, F, H, W]` and normalize to
//!   `[-1, 1]`.
//!
//! ## Frame-count contract for Wan VAEs
//! Wan VAEs require `(F_src - 1) % 4 == 0` for the temporal compression
//! to land cleanly. The caller picks `num_frames` accordingly (e.g.
//! 1, 5, 9, 13, ...). This module does NOT enforce that — it returns
//! whatever frame count the caller asked for; the trainer's encode step
//! is what eventually fails on a bad count.

use flame_core::{DType, Result, Shape, Tensor};
use image::imageops::FilterType;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

/// Output of [`decode_to_tensor`]: a video tensor + the frame count we
/// actually got (callers may ask for 9 frames from a 7-frame source and
/// get 7 back).
pub struct DecodedClip {
    /// `[1, 3, F, H, W]` BF16 in `[-1, 1]`, where F is `num_frames` for
    /// images (always 1) or the actual frames decoded for video.
    pub video: Tensor,
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
}

/// Decode an image OR video file into a BF16 video tensor.
///
/// - `path`: file extension determines the path (`.jpg`/`.png`/`.webp`/
///   `.bmp` → image branch; `.mp4`/`.webm`/`.mov`/`.mkv` → ffmpeg branch).
/// - `target_size`: square pixel size both H and W are resized to.
/// - `num_frames_request`: for video, max frames to extract. For image,
///   ignored (always F=1).
/// - `target_fps`: optional. When `Some`, ffmpeg `-r` re-samples to this
///   rate before extraction; when `None`, native rate is used.
///
/// Returns a [`DecodedClip`].
pub fn decode_to_tensor(
    path: &Path,
    target_size: usize,
    num_frames_request: usize,
    target_fps: Option<f32>,
    device: Arc<cudarc::driver::CudaDevice>,
) -> Result<DecodedClip> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "jpg" | "jpeg" | "png" | "webp" | "bmp" => {
            decode_image(path, target_size, device)
        }
        "mp4" | "webm" | "mov" | "mkv" | "avi" => {
            decode_video_via_ffmpeg(path, target_size, num_frames_request, target_fps, device)
        }
        other => Err(flame_core::Error::InvalidInput(format!(
            "decode_to_tensor: unsupported extension '{other}' for {}",
            path.display()
        ))),
    }
}

/// Image-only branch: open via `image` crate, resize, normalize, lift to
/// `[1, 3, 1, H, W]` BF16.
fn decode_image(
    path: &Path,
    size: usize,
    device: Arc<cudarc::driver::CudaDevice>,
) -> Result<DecodedClip> {
    let img = image::open(path)
        .map_err(|e| flame_core::Error::InvalidInput(format!("image::open({}): {e}", path.display())))?
        .resize_exact(size as u32, size as u32, FilterType::Lanczos3)
        .to_rgb32f();
    let (w, h) = img.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    // HWC → CHW in [-1, 1].
    let mut pixels = vec![0f32; 3 * hu * wu];
    for (x, y, p) in img.enumerate_pixels() {
        let (xu, yu) = (x as usize, y as usize);
        for c in 0..3 {
            pixels[c * hu * wu + yu * wu + xu] = p.0[c] * 2.0 - 1.0;
        }
    }
    let video = Tensor::from_vec(
        pixels,
        Shape::from_dims(&[1, 3, 1, hu, wu]),
        device,
    )?
    .to_dtype(DType::BF16)?;
    Ok(DecodedClip {
        video,
        num_frames: 1,
        height: hu,
        width: wu,
    })
}

/// Video branch: shell `ffmpeg`, capture `rgb24` raw pixels.
fn decode_video_via_ffmpeg(
    path: &Path,
    size: usize,
    num_frames_request: usize,
    target_fps: Option<f32>,
    device: Arc<cudarc::driver::CudaDevice>,
) -> Result<DecodedClip> {
    if num_frames_request == 0 {
        return Err(flame_core::Error::InvalidInput(
            "decode_video_via_ffmpeg: num_frames_request must be > 0".into(),
        ));
    }

    // ffmpeg -loglevel error -i <path> [-r FPS] -vf scale=SxS -frames:v N
    //        -f rawvideo -pix_fmt rgb24 -
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-loglevel", "error"])
        .arg("-i")
        .arg(path);
    if let Some(fps) = target_fps {
        cmd.args(["-r", &format!("{fps}")]);
    }
    cmd.args([
        "-vf",
        &format!("scale={size}:{size}"),
        "-frames:v",
        &num_frames_request.to_string(),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]);
    let output = cmd
        .output()
        .map_err(|e| flame_core::Error::InvalidInput(format!("ffmpeg spawn: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(flame_core::Error::InvalidInput(format!(
            "ffmpeg failed for {} (status {:?}): {}",
            path.display(),
            output.status.code(),
            stderr
        )));
    }
    let raw = output.stdout;

    // Each frame is `size * size * 3` bytes (rgb24 packed HWC).
    let bytes_per_frame = size * size * 3;
    if raw.is_empty() {
        return Err(flame_core::Error::InvalidInput(format!(
            "ffmpeg returned 0 bytes for {} (no decodable frames at scale={size}:{size})",
            path.display()
        )));
    }
    if raw.len() % bytes_per_frame != 0 {
        return Err(flame_core::Error::InvalidInput(format!(
            "ffmpeg output {} bytes for {}, not a multiple of frame size {} (size={size})",
            raw.len(),
            path.display(),
            bytes_per_frame
        )));
    }
    let actual_frames = raw.len() / bytes_per_frame;

    // Reshape rgb24 HWC into CHW per frame → final layout [3, F, H, W] in [-1, 1].
    // numel = 3 * F * H * W
    let numel = 3 * actual_frames * size * size;
    let mut chw = vec![0f32; numel];
    for fi in 0..actual_frames {
        let frame_in = &raw[fi * bytes_per_frame..(fi + 1) * bytes_per_frame];
        for y in 0..size {
            for x in 0..size {
                let src_off = (y * size + x) * 3;
                let r = frame_in[src_off] as f32 / 255.0 * 2.0 - 1.0;
                let g = frame_in[src_off + 1] as f32 / 255.0 * 2.0 - 1.0;
                let b = frame_in[src_off + 2] as f32 / 255.0 * 2.0 - 1.0;
                // Layout: [3, F, H, W]
                chw[0 * actual_frames * size * size + fi * size * size + y * size + x] = r;
                chw[1 * actual_frames * size * size + fi * size * size + y * size + x] = g;
                chw[2 * actual_frames * size * size + fi * size * size + y * size + x] = b;
            }
        }
    }

    let video = Tensor::from_vec(
        chw,
        Shape::from_dims(&[1, 3, actual_frames, size, size]),
        device,
    )?
    .to_dtype(DType::BF16)?;

    Ok(DecodedClip {
        video,
        num_frames: actual_frames,
        height: size,
        width: size,
    })
}

/// Extension classifier — useful for prepare binaries to walk a dataset
/// directory and route per-file.
pub fn extension_kind(path: &Path) -> ExtensionKind {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg" | "png" | "webp" | "bmp") => ExtensionKind::Image,
        Some("mp4" | "webm" | "mov" | "mkv" | "avi") => ExtensionKind::Video,
        Some("txt") => ExtensionKind::Caption,
        _ => ExtensionKind::Other,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionKind {
    Image,
    Video,
    Caption,
    Other,
}
