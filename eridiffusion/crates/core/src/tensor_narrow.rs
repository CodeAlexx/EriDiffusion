use crate::{Error, Result};
use flame_core::{DType, Shape, Tensor};
use crate::cuda::dtype_tag::dtype_to_tag;
use std::os::raw::{c_int, c_void};
use flame_core::device::CudaStreamRawPtrExt;

extern "C" {
    fn eri_narrow_strided_launch(
        src: *const c_void,
        dst: *mut c_void,
        rank: c_int,
        out_shape_host: *const i64,
        src_strides_host: *const i64,
        out_strides_host: *const i64,
        dim: c_int,
        start: i64,
        elem_size: i64,
        n_elements: i64,
        stream: *mut c_void,
    ) -> i32;

    fn eri_narrow_backward_scatter_add_launch(
        grad_out: *const c_void,
        grad_in: *mut c_void,
        rank: c_int,
        out_shape_host: *const i64,
        in_strides_host: *const i64,
        out_strides_host: *const i64,
        dim: c_int,
        start: i64,
        elem_size: i64,
        dtype_tag: c_int,
        n_elements: i64,
        stream: *mut c_void,
    ) -> i32;
}

pub trait TensorNarrowExt {
    fn narrow_general_cuda(&self, dim: usize, start: usize, length: usize) -> Result<Tensor>;
}

impl TensorNarrowExt for Tensor {
    fn narrow_general_cuda(&self, dim: usize, start: usize, length: usize) -> Result<Tensor> {
        let in_shape = self.shape();
        let rank = in_shape.rank();
        if dim >= rank {
            return Err(Error::TensorOp(format!(
                "narrow: dim {} out of range for rank {}",
                dim, rank
            )));
        }
        let dims = in_shape.dims();
        if start + length > dims[dim] {
            return Err(Error::TensorOp(format!(
                "narrow: range [{}..{}) exceeds dim {} (size {})",
                start,
                start + length,
                dim,
                dims[dim]
            )));
        }

        // Output shape
        let mut out_dims = dims.to_vec();
        out_dims[dim] = length;

        // Strides in elements (assume contiguous layout for input)
        let src_strides_elems: Vec<i64> = in_shape
            .strides()
            .into_iter()
            .map(|x| x as i64)
            .collect();
        // Row-major strides for output
        let mut out_strides_elems = vec![0i64; rank];
        if rank > 0 {
            out_strides_elems[rank - 1] = 1;
            for i in (0..rank - 1).rev() {
                out_strides_elems[i] = out_strides_elems[i + 1] * (out_dims[i + 1] as i64);
            }
        }

        let n_elements: i64 = out_dims.iter().fold(1i64, |acc, &d| acc * d as i64);
        let dtype: DType = self.dtype();
        let elem_size = dtype.size_in_bytes() as i64;

        // Allocate destination
        let dev = self.device().clone();
        let mut dst = Tensor::zeros_dtype(Shape::from_dims(&out_dims), dtype, dev.clone())
            .map_err(|e| Error::Flame(e))?;

        // Raw pointers
        let src_ptr = self.cuda_ptr() as *const c_void;
        let dst_ptr = dst.cuda_ptr_mut() as *mut c_void;

        let out_shape_i64: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();

        // Use device stream (default/null if not configured)
        let stream: *mut c_void = self.device().cuda_stream_raw_ptr();

        let code = unsafe {
            eri_narrow_strided_launch(
                src_ptr,
                dst_ptr,
                rank as c_int,
                out_shape_i64.as_ptr(),
                src_strides_elems.as_ptr(),
                out_strides_elems.as_ptr(),
                dim as c_int,
                start as i64,
                elem_size,
                n_elements,
                stream,
            )
        };
        if code != 0 {
            return Err(Error::TensorOp(format!(
                "narrow_strided_launch failed with code {}",
                code
            )));
        }

        Ok(dst)
    }
}

/// Scatter-add backward helper for narrow: accumulates grad_out into grad_in
/// at the slice [start, start+length) along `dim`.
pub fn narrow_backward_scatter_add_cuda(
    grad_out: &Tensor,
    grad_in: &mut Tensor,
    dim: usize,
    start: usize,
    length: usize,
) -> Result<()> {
    let in_shape = grad_in.shape();
    let out_shape = grad_out.shape();
    let rank = in_shape.rank();
    if rank != out_shape.rank() {
        return Err(Error::TensorOp("narrow backward: rank mismatch".into()));
    }
    if dim >= rank {
        return Err(Error::TensorOp(format!(
            "narrow backward: dim {} out of range",
            dim
        )));
    }
    // Validate shapes
    let in_dims = in_shape.dims();
    let out_dims = out_shape.dims();
    for i in 0..rank {
        if i == dim {
            if out_dims[i] != length {
                return Err(Error::TensorOp(format!(
                    "narrow backward: out length {} != {} at axis {}",
                    out_dims[i], length, i
                )));
            }
            if start + length > in_dims[i] {
                return Err(Error::TensorOp(format!(
                    "narrow backward: range [{}..{}) exceeds input dim {} (size {})",
                    start,
                    start + length,
                    i,
                    in_dims[i]
                )));
            }
        } else if out_dims[i] != in_dims[i] {
            return Err(Error::TensorOp(format!(
                "narrow backward: shape mismatch at axis {} (out {} vs in {})",
                i, out_dims[i], in_dims[i]
            )));
        }
    }
    // DType check
    if grad_out.dtype() != grad_in.dtype() {
        return Err(Error::TensorOp("narrow backward: dtype mismatch".into()));
    }

    // Build strides: input strides from Shape (contiguous assumption), output row-major
    let in_strides_elems: Vec<i64> = in_shape
        .strides()
        .into_iter()
        .map(|x| x as i64)
        .collect();
    let mut out_strides_elems = vec![0i64; rank];
    if rank > 0 {
        out_strides_elems[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            out_strides_elems[i] = out_strides_elems[i + 1] * (out_dims[i + 1] as i64);
        }
    }
    let out_shape_i64: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let n_elements: i64 = out_shape_i64.iter().product();
    let elem_size = grad_out.dtype().size_in_bytes() as i64;

    // Raw pointers
    let go_ptr = grad_out.cuda_ptr() as *const c_void;
    let gi_ptr = grad_in.cuda_ptr_mut() as *mut c_void;

    // Device stream (default/null if not configured)
    let stream: *mut c_void = grad_in.device().cuda_stream_raw_ptr();

    // Map dtype to tag expected by CUDA: F32=0, F16=1, BF16=2, I32=3
    let dtype_tag: c_int = dtype_to_tag(grad_in.dtype().into()) as c_int;

    let code = unsafe {
        eri_narrow_backward_scatter_add_launch(
            go_ptr,
            gi_ptr,
            rank as c_int,
            out_shape_i64.as_ptr(),
            in_strides_elems.as_ptr(),
            out_strides_elems.as_ptr(),
            dim as c_int,
            start as i64,
            elem_size,
            dtype_tag,
            n_elements,
            stream,
        )
    };
    if code != 0 {
        return Err(Error::TensorOp(format!(
            "narrow_backward_scatter_add_launch failed with code {}",
            code
        )));
    }
    Ok(())
}
