//! O_DIRECT safetensors save — bypasses the page cache so checkpoint writes
//! don't evict the training dataset cache.
//!
//! Implementation ported from serenity-safetensors (`write_direct_streaming`).
//! 4 KB alignment, 4 MB chunked writes. Falls back to a buffered write if
//! O_DIRECT is unsupported (tmpfs, some network FS, non-Linux).
//!
//! Output format matches flame_core::serialization::save_tensors_safetensors:
//! tensors are serialized as F32 (regardless of original dtype) per the
//! existing convention. LoRA saves are F32 already so no precision loss.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use flame_core::Tensor;
use serde_json::{json, Value};

use crate::{EriDiffusionError, Result};

const ALIGN: usize = 4096;
const CHUNK: usize = 4 * 1024 * 1024;

/// Save a HashMap of tensors + extra metadata via O_DIRECT. F32 only.
/// Behaves identically to `flame_core::serialization::save_tensors_with_metadata`
/// from the reader's perspective.
pub fn save_tensors_with_metadata_direct(
    tensors: &HashMap<String, Tensor>,
    extra_metadata: &HashMap<String, String>,
    path: &Path,
) -> Result<()> {
    // Sort keys for deterministic file layout.
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    // Pull all tensor data to host as F32 (matches flame-core's save format).
    let mut datas: Vec<Vec<f32>> = Vec::with_capacity(names.len());
    let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(names.len());
    for name in &names {
        let t = tensors
            .get(*name)
            .ok_or_else(|| EriDiffusionError::Training(format!("save_direct: missing {name}")))?;
        let v = t
            .to_vec()
            .map_err(|e| EriDiffusionError::Training(format!("save_direct {name}: {e}")))?;
        shapes.push(t.shape().dims().to_vec());
        datas.push(v);
    }

    // Build safetensors header.
    let mut header_obj = serde_json::Map::new();
    if !extra_metadata.is_empty() {
        let mut m = serde_json::Map::new();
        for (k, v) in extra_metadata {
            m.insert(k.clone(), Value::String(v.clone()));
        }
        header_obj.insert("__metadata__".into(), Value::Object(m));
    }
    let mut offset: u64 = 0;
    for (i, name) in names.iter().enumerate() {
        let nbytes = (datas[i].len() * 4) as u64;
        header_obj.insert(
            (*name).clone(),
            json!({
                "dtype": "F32",
                "shape": shapes[i],
                "data_offsets": [offset, offset + nbytes],
            }),
        );
        offset += nbytes;
    }
    let header_json = serde_json::to_string(&Value::Object(header_obj))
        .map_err(|e| EriDiffusionError::Training(format!("save_direct header json: {e}")))?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let total_tensor_bytes: usize = datas.iter().map(|d| d.len() * 4).sum();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    #[cfg(target_os = "linux")]
    {
        match write_direct(path, header_len, header_bytes, &datas, total_tensor_bytes) {
            Ok(()) => return Ok(()),
            Err(e) => {
                log::debug!("save_direct: O_DIRECT failed ({e}); falling back to buffered write");
            }
        }
    }

    write_buffered(path, header_len, header_bytes, &datas)
}

#[cfg(target_os = "linux")]
fn write_direct(
    path: &Path,
    header_len: u64,
    header_bytes: &[u8],
    datas: &[Vec<f32>],
    total_tensor_bytes: usize,
) -> std::result::Result<(), String> {
    let cpath = std::ffi::CString::new(path.to_str().unwrap_or(""))
        .map_err(|e| format!("path: {e}"))?;
    let fd = unsafe {
        libc::open(
            cpath.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC | libc::O_DIRECT,
            0o644,
        )
    };
    if fd < 0 {
        return Err(format!("open O_DIRECT: {}", std::io::Error::last_os_error()));
    }

    let alloc_size = (CHUNK + ALIGN - 1) & !(ALIGN - 1);
    let layout = match std::alloc::Layout::from_size_align(alloc_size, ALIGN) {
        Ok(l) => l,
        Err(e) => {
            unsafe { libc::close(fd) };
            return Err(format!("layout: {e}"));
        }
    };

    let result: std::result::Result<(), String> = unsafe {
        let buf = std::alloc::alloc_zeroed(layout);
        if buf.is_null() {
            libc::close(fd);
            return Err("alloc aligned buffer failed".into());
        }
        let mut buffered = 0usize;
        let mut exact_len = 0usize;

        let mut flush = |buffered: &mut usize| -> std::result::Result<(), String> {
            if *buffered == 0 {
                return Ok(());
            }
            let write_len = (*buffered + ALIGN - 1) & !(ALIGN - 1);
            if write_len > *buffered {
                std::ptr::write_bytes(buf.add(*buffered), 0, write_len - *buffered);
            }
            let written = libc::write(fd, buf as *const libc::c_void, write_len);
            if written < 0 {
                return Err(format!("write: {}", std::io::Error::last_os_error()));
            }
            *buffered = 0;
            Ok(())
        };

        let mut copy_segment = |segment: &[u8],
                                buffered: &mut usize,
                                exact_len: &mut usize|
         -> std::result::Result<(), String> {
            let mut off = 0usize;
            *exact_len += segment.len();
            while off < segment.len() {
                let space = CHUNK - *buffered;
                let take = (segment.len() - off).min(space);
                std::ptr::copy_nonoverlapping(
                    segment.as_ptr().add(off),
                    buf.add(*buffered),
                    take,
                );
                *buffered += take;
                off += take;
                if *buffered == CHUNK {
                    flush(buffered)?;
                }
            }
            Ok(())
        };

        let result: std::result::Result<(), String> = (|| {
            copy_segment(&header_len.to_le_bytes(), &mut buffered, &mut exact_len)?;
            copy_segment(header_bytes, &mut buffered, &mut exact_len)?;
            for d in datas {
                let bytes: &[u8] = bytemuck_slice_f32(d);
                copy_segment(bytes, &mut buffered, &mut exact_len)?;
            }
            flush(&mut buffered)?;
            let expected = 8 + header_bytes.len() + total_tensor_bytes;
            if exact_len != expected {
                return Err(format!(
                    "save_direct length mismatch: expected {expected}, wrote {exact_len}"
                ));
            }
            libc::ftruncate(fd, expected as libc::off_t);
            Ok(())
        })();

        std::alloc::dealloc(buf, layout);
        result
    };

    let close_rc = unsafe { libc::close(fd) };
    if let Err(e) = result {
        return Err(e);
    }
    if close_rc < 0 {
        return Err(format!("close: {}", std::io::Error::last_os_error()));
    }
    Ok(())
}

fn write_buffered(
    path: &Path,
    header_len: u64,
    header_bytes: &[u8],
    datas: &[Vec<f32>],
) -> Result<()> {
    let file = File::create(path)
        .map_err(|e| EriDiffusionError::Training(format!("save_direct fallback create: {e}")))?;
    let mut w = BufWriter::new(file);
    w.write_all(&header_len.to_le_bytes())
        .map_err(|e| EriDiffusionError::Training(format!("save_direct header_len: {e}")))?;
    w.write_all(header_bytes)
        .map_err(|e| EriDiffusionError::Training(format!("save_direct header: {e}")))?;
    for d in datas {
        let bytes = bytemuck_slice_f32(d);
        w.write_all(bytes)
            .map_err(|e| EriDiffusionError::Training(format!("save_direct data: {e}")))?;
    }
    w.flush()
        .map_err(|e| EriDiffusionError::Training(format!("save_direct flush: {e}")))?;
    Ok(())
}

fn bytemuck_slice_f32(v: &[f32]) -> &[u8] {
    // Safety: f32 is plain-old-data, alignment of f32 (4) is a multiple of u8 (1).
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<f32>())
    }
}
