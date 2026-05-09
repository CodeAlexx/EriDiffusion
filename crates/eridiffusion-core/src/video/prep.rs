//! High-level prep helpers — image+video walk + tensor construction in
//! one place so each `prepare_*.rs` binary can stay minimal.
//!
//! Typical use (Wan-style):
//! ```ignore
//! use eridiffusion_core::video::prep::walk_dataset;
//! let pairs = walk_dataset(&args.input_dir)?;
//! for pair in pairs {
//!     let clip = decode_to_tensor(&pair.media, args.size, args.num_frames, None, device.clone())?;
//!     let caption = read_caption(&pair.caption_or_none()).unwrap_or_default();
//!     // ... vae.encode(clip.video) + umt5.encode(caption) + save ...
//! }
//! ```

use crate::video::decode::{extension_kind, ExtensionKind};
use std::fs;
use std::path::{Path, PathBuf};

/// One entry in a dataset walk: a media file + optional `<stem>.txt`
/// caption sibling.
pub struct DatasetPair {
    pub media: PathBuf,
    pub caption: Option<PathBuf>,
    pub kind: ExtensionKind,
}

impl DatasetPair {
    /// Read the paired caption, or empty string when absent.
    pub fn read_caption(&self) -> String {
        self.caption
            .as_ref()
            .and_then(|p| fs::read_to_string(p).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    }
}

/// Walk a directory and return one `DatasetPair` per image/video file.
/// Files are sorted by path for reproducibility.
///
/// `<stem>.txt` is auto-paired when present in the same directory.
pub fn walk_dataset(input_dir: &Path) -> std::io::Result<Vec<DatasetPair>> {
    let mut out: Vec<DatasetPair> = Vec::new();
    for entry in fs::read_dir(input_dir)? {
        let path = entry?.path();
        let kind = extension_kind(&path);
        match kind {
            ExtensionKind::Image | ExtensionKind::Video => {
                let txt = path.with_extension("txt");
                let caption = if txt.exists() { Some(txt) } else { None };
                out.push(DatasetPair {
                    media: path,
                    caption,
                    kind,
                });
            }
            _ => {}
        }
    }
    out.sort_by(|a, b| a.media.cmp(&b.media));
    Ok(out)
}

/// Sanity-check a target frame count against Wan VAE's temporal-compression
/// constraint: `(F_src - 1) % 4 == 0` so the latent frame count is integral.
/// Returns the largest valid count `<= num_frames_request`.
pub fn snap_to_wan_frame_count(num_frames_request: usize) -> usize {
    if num_frames_request <= 1 {
        return 1;
    }
    let n = num_frames_request - 1;
    let snapped = (n / 4) * 4 + 1;
    snapped.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_wan_frames() {
        assert_eq!(snap_to_wan_frame_count(0), 1);
        assert_eq!(snap_to_wan_frame_count(1), 1);
        assert_eq!(snap_to_wan_frame_count(2), 1);
        assert_eq!(snap_to_wan_frame_count(5), 5);
        assert_eq!(snap_to_wan_frame_count(8), 5); // 8 → 7 fails (-1)%4 != 0; snap to 5
        assert_eq!(snap_to_wan_frame_count(9), 9);
        assert_eq!(snap_to_wan_frame_count(13), 13);
        assert_eq!(snap_to_wan_frame_count(15), 13); // 15 → snap to 13
    }
}
