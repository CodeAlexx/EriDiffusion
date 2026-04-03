use std::collections::HashSet;

use eridiffusion_core::{device::shared_cuda_device, tensor_narrow::{narrow_backward_scatter_add_cuda, TensorNarrowExt}, DType};
use flame_core::{Shape, Tensor};

fn to_f32_tensor(t: &Tensor) -> Tensor {
    if t.dtype() == DType::F32 {
        t.clone_result().expect("clone")
    } else {
        t.to_dtype(DType::F32).expect("cast")
    }
}

fn tensor_from_data(dtype: DType, shape: &[usize], data: &[f32]) -> Tensor {
    let dev = shared_cuda_device().expect("cuda");
    let shape = Shape::from_dims(shape);
    match dtype {
        DType::F32 => Tensor::from_vec(data.to_vec(), shape, dev).expect("tensor"),
        _ => Tensor::from_slice_dtype(data, shape, dev, dtype).expect("tensor"),
    }
}

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

fn tolerance(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => 1e-5,
        DType::F16 => 5e-3,
        DType::BF16 => 1e-2,
        DType::I32 => 1e-5,
        _ => 1e-5,
    }
}

fn host_check_forward(
    input: &Tensor,
    output: &Tensor,
    in_dims: &[usize],
    dim: usize,
    start: usize,
    len: usize,
    dtype: DType,
) {
    let input_f32 = to_f32_tensor(input);
    let output_f32 = to_f32_tensor(output);
    let a = input_f32.to_vec().expect("input host");
    let b = output_f32.to_vec().expect("output host");
    let rank = in_dims.len();
    let mut out_dims = in_dims.to_vec();
    out_dims[dim] = len;

    let out_strides = compute_strides(&out_dims);
    let in_strides = compute_strides(in_dims);
    let tol = tolerance(dtype);

    for lin_out in 0..b.len() {
        // unravel to multi-index
        let mut rem = lin_out;
        let mut idx = vec![0usize; rank];
        for i in 0..rank {
            let stride = if rank == 0 { 1 } else { out_strides[i] };
            let coord = if stride == 0 { 0 } else { rem / stride };
            if stride != 0 {
                rem -= coord * stride;
            }
            idx[i] = coord;
        }
        let mut lin_in = 0usize;
        for i in 0..rank {
            let coord = if i == dim { idx[i] + start } else { idx[i] };
            lin_in += coord * in_strides.get(i).copied().unwrap_or(1);
        }
        let diff = (b[lin_out] - a[lin_in]).abs();
        assert!(
            diff <= tol,
            "forward mismatch dtype {:?} dim {} start {} len {} at out idx {}: {} vs {} (tol {})",
            dtype,
            dim,
            start,
            len,
            lin_out,
            b[lin_out],
            a[lin_in],
            tol
        );
    }
}

fn forward_case(dtype: DType, dims: &[usize], dim: usize, start: usize, len: usize) {
    let elem_count: usize = dims.iter().product();
    let data: Vec<f32> = (0..elem_count).map(|i| i as f32 * 0.01 - 1.0).collect();
    let input = tensor_from_data(dtype, dims, &data);
    let output = input
        .narrow_general_cuda(dim, start, len)
        .expect("narrow forward");
    host_check_forward(&input, &output, dims, dim, start, len, dtype);
}

fn build_cases(dims: &[usize], dim: usize) -> Vec<(usize, usize)> {
    let size = dims[dim];
    let mut lengths = HashSet::new();
    lengths.insert(1usize);
    lengths.insert(size);
    if size > 2 {
        lengths.insert(size.saturating_sub(1));
    }
    if size > 3 {
        lengths.insert(size / 2);
    }

    let mut cases = Vec::new();
    for len in lengths.into_iter().filter(|&l| l > 0 && l <= size) {
        let mut starts = HashSet::new();
        starts.insert(0usize);
        if size > len {
            starts.insert(size - len);
        }
        if size > len + 1 {
            starts.insert((size - len) / 2);
        }
        for start in starts { cases.push((start, len)); }
    }
    cases.sort();
    cases
}

fn run_forward_suite(dtype: DType) {
    let _ = shared_cuda_device();
    let dims = [2usize, 3usize, 4usize, 5usize];
    for dim in 0..dims.len() {
        for (start, len) in build_cases(&dims, dim) {
            forward_case(dtype, &dims, dim, start, len);
        }
    }
    // Rank-1 sanity case
    forward_case(dtype, &[9usize], 0, 3, 3);
}

fn to_f32_vec(t: &Tensor) -> Vec<f32> {
    to_f32_tensor(t).to_vec().expect("to_vec")
}

fn backward_case(dtype: DType, dims: &[usize], dim: usize, start: usize, len: usize, repeats: usize) {
    let elem_in: usize = dims.iter().product();
    let mut out_dims = dims.to_vec();
    out_dims[dim] = len;
    let elem_out: usize = out_dims.iter().product();

    let go_data: Vec<f32> = (0..elem_out).map(|i| i as f32 * 0.02 + 0.5).collect();
    let grad_out = tensor_from_data(dtype, &out_dims, &go_data);
    let dev = shared_cuda_device().expect("cuda");
    let mut grad_in = Tensor::zeros_dtype(Shape::from_dims(dims), dtype, dev.clone()).expect("zeros");

    for _ in 0..repeats {
        narrow_backward_scatter_add_cuda(&grad_out, &mut grad_in, dim, start, len)
            .expect("narrow backward");
    }

    let gi = to_f32_vec(&grad_in);
    let in_strides = compute_strides(dims);
    let out_strides = compute_strides(&out_dims);
    let tol = tolerance(dtype) * repeats as f32;

    for lin_in in 0..elem_in {
        // Unravel input index
        let mut rem = lin_in;
        let mut idx = vec![0usize; dims.len()];
        for i in 0..dims.len() {
            let stride = if dims.is_empty() { 1 } else { in_strides[i] };
            let coord = if stride == 0 { 0 } else { rem / stride };
            if stride != 0 {
                rem -= coord * stride;
            }
            idx[i] = coord;
        }
        let val = gi[lin_in];
        if idx[dim] >= start && idx[dim] < start + len {
            let mut lin_out = 0usize;
            for i in 0..dims.len() {
                let coord = if i == dim { idx[i] - start } else { idx[i] };
                lin_out += coord * out_strides.get(i).copied().unwrap_or(1);
            }
            let expected = go_data[lin_out] * repeats as f32;
            let diff = (val - expected).abs();
            assert!(
                diff <= tol,
                "backward mismatch dtype {:?} dim {} start {} len {} repeats {} at {}: {} vs {} (tol {})",
                dtype,
                dim,
                start,
                len,
                repeats,
                lin_in,
                val,
                expected,
                tol
            );
        } else {
            assert!(val.abs() <= tol, "backward outside slice nonzero: {} -> {}", lin_in, val);
        }
    }
}

fn run_backward_suite(dtype: DType, repeats: usize) {
    let _ = shared_cuda_device();
    let dims = [2usize, 3usize, 4usize, 5usize];
    for dim in 0..dims.len() {
        for (start, len) in build_cases(&dims, dim) {
            backward_case(dtype, &dims, dim, start, len, repeats);
        }
    }
    backward_case(dtype, &[11usize], 0, 5, 3, repeats);
}

#[test]
fn narrow_forward_cuda_fp32() {
    run_forward_suite(DType::F32);
}

#[test]
fn narrow_forward_cuda_bf16() {
    run_forward_suite(DType::BF16);
}

#[test]
fn narrow_forward_cuda_f16() {
    run_forward_suite(DType::F16);
}

#[test]
fn narrow_forward_cuda_i32() {
    run_forward_suite(DType::I32);
}

#[test]
fn narrow_backward_cuda_fp32_single() {
    run_backward_suite(DType::F32, 1);
}

#[test]
fn narrow_backward_cuda_fp32_accumulates() {
    run_backward_suite(DType::F32, 2);
}

#[test]
fn narrow_backward_cuda_bf16_single() {
    run_backward_suite(DType::BF16, 1);
}

#[test]
fn narrow_backward_cuda_bf16_accumulates() {
    run_backward_suite(DType::BF16, 2);
}

#[test]
fn narrow_backward_cuda_f16_single() {
    run_backward_suite(DType::F16, 1);
}

#[test]
fn narrow_backward_cuda_f16_accumulates() {
    run_backward_suite(DType::F16, 2);
}

#[test]
fn narrow_backward_cuda_i32_single() {
    run_backward_suite(DType::I32, 1);
}

#[test]
fn narrow_backward_cuda_i32_accumulates() {
    run_backward_suite(DType::I32, 2);
}
