//! Raw FFI bindings to cuDNN library
//! 
//! This provides low-level access to NVIDIA's cuDNN library for optimized
//! deep learning primitives, particularly convolution operations.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_int, c_void, c_float, c_double, c_ulonglong};

// Opaque types for cuDNN handles
#[repr(C)]
pub struct cudnnContext;
pub type cudnnHandle_t = *mut cudnnContext;

#[repr(C)]
pub struct cudnnTensorStruct;
pub type cudnnTensorDescriptor_t = *mut cudnnTensorStruct;

#[repr(C)]
pub struct cudnnFilterStruct;
pub type cudnnFilterDescriptor_t = *mut cudnnFilterStruct;

#[repr(C)]
pub struct cudnnConvolutionStruct;
pub type cudnnConvolutionDescriptor_t = *mut cudnnConvolutionStruct;

#[repr(C)]
pub struct cudnnActivationStruct;
pub type cudnnActivationDescriptor_t = *mut cudnnActivationStruct;

// Status codes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum cudnnStatus_t {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
}

// Data types
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum cudnnDataType_t {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
    CUDNN_DATA_BFLOAT16 = 9,
    CUDNN_DATA_INT64 = 10,
}

// Tensor format
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2,
}

// Convolution mode
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum cudnnConvolutionMode_t {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
}

// Forward convolution algorithms
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum cudnnConvolutionFwdAlgo_t {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8,
}

// Algorithm performance
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudnnConvolutionFwdAlgoPerf_t {
    pub algo: cudnnConvolutionFwdAlgo_t,
    pub status: cudnnStatus_t,
    pub time: c_float,
    pub memory: usize,
    pub determinism: c_int,
    pub mathType: c_int,
    pub reserved: [c_int; 3],
}

impl Default for cudnnConvolutionFwdAlgoPerf_t {
    fn default() -> Self {
        Self {
            algo: cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            status: cudnnStatus_t::CUDNN_STATUS_SUCCESS,
            time: 0.0,
            memory: 0,
            determinism: 0,
            mathType: 0,
            reserved: [0; 3],
        }
    }
}

// Math type
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum cudnnMathType_t {
    CUDNN_DEFAULT_MATH = 0,
    CUDNN_TENSOR_OP_MATH = 1,
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
    CUDNN_FMA_MATH = 3,
}

// External C functions
#[link(name = "cudnn")]
extern "C" {
    // Library management
    pub fn cudnnGetVersion() -> usize;
    pub fn cudnnGetCudartVersion() -> usize;
    pub fn cudnnGetErrorString(status: cudnnStatus_t) -> *const i8;
    
    // Handle management
    pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnSetStream(handle: cudnnHandle_t, stream: *mut c_void) -> cudnnStatus_t;
    pub fn cudnnGetStream(handle: cudnnHandle_t, stream: *mut *mut c_void) -> cudnnStatus_t;
    
    // Tensor descriptor management
    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetTensor4dDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        format: cudnnTensorFormat_t,
        dataType: cudnnDataType_t,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
    ) -> cudnnStatus_t;
    pub fn cudnnGetTensor4dDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        dataType: *mut cudnnDataType_t,
        n: *mut c_int,
        c: *mut c_int,
        h: *mut c_int,
        w: *mut c_int,
        nStride: *mut c_int,
        cStride: *mut c_int,
        hStride: *mut c_int,
        wStride: *mut c_int,
    ) -> cudnnStatus_t;
    
    // Filter descriptor management
    pub fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetFilter4dDescriptor(
        filterDesc: cudnnFilterDescriptor_t,
        dataType: cudnnDataType_t,
        format: cudnnTensorFormat_t,
        k: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
    ) -> cudnnStatus_t;
    
    // Convolution descriptor management
    pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetConvolution2dDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
        pad_h: c_int,
        pad_w: c_int,
        u: c_int,  // vertical stride
        v: c_int,  // horizontal stride
        dilation_h: c_int,
        dilation_w: c_int,
        mode: cudnnConvolutionMode_t,
        computeType: cudnnDataType_t,
    ) -> cudnnStatus_t;
    pub fn cudnnSetConvolutionMathType(
        convDesc: cudnnConvolutionDescriptor_t,
        mathType: cudnnMathType_t,
    ) -> cudnnStatus_t;
    
    // Get output dimensions
    pub fn cudnnGetConvolution2dForwardOutputDim(
        convDesc: cudnnConvolutionDescriptor_t,
        inputTensorDesc: cudnnTensorDescriptor_t,
        filterDesc: cudnnFilterDescriptor_t,
        n: *mut c_int,
        c: *mut c_int,
        h: *mut c_int,
        w: *mut c_int,
    ) -> cudnnStatus_t;
    
    // Algorithm selection
    pub fn cudnnGetConvolutionForwardAlgorithm_v7(
        handle: cudnnHandle_t,
        srcDesc: cudnnTensorDescriptor_t,
        filterDesc: cudnnFilterDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        destDesc: cudnnTensorDescriptor_t,
        requestedAlgoCount: c_int,
        returnedAlgoCount: *mut c_int,
        perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
    ) -> cudnnStatus_t;
    
    // Get workspace size
    pub fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: cudnnHandle_t,
        xDesc: cudnnTensorDescriptor_t,
        wDesc: cudnnFilterDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        yDesc: cudnnTensorDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        sizeInBytes: *mut usize,
    ) -> cudnnStatus_t;
    
    // Forward convolution
    pub fn cudnnConvolutionForward(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        wDesc: cudnnFilterDescriptor_t,
        w: *const c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: usize,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t;
    
    // Bias addition
    pub fn cudnnAddTensor(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        aDesc: cudnnTensorDescriptor_t,
        A: *const c_void,
        beta: *const c_void,
        cDesc: cudnnTensorDescriptor_t,
        C: *mut c_void,
    ) -> cudnnStatus_t;
    
    // Activation functions
    pub fn cudnnCreateActivationDescriptor(activationDesc: *mut cudnnActivationDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyActivationDescriptor(activationDesc: cudnnActivationDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnActivationForward(
        handle: cudnnHandle_t,
        activationDesc: cudnnActivationDescriptor_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t;
}

// Helper function to check cuDNN status
pub fn check_cudnn_status(status: cudnnStatus_t) -> Result<(), String> {
    if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        unsafe {
            let error_str = cudnnGetErrorString(status);
            let c_str = std::ffi::CStr::from_ptr(error_str);
            Err(format!("cuDNN error: {:?}", c_str.to_str().unwrap_or("Unknown error")))
        }
    } else {
        Ok(())
    }
}

// Macro for checking cuDNN calls
#[macro_export]
macro_rules! check_cudnn {
    ($call:expr) => {{
        let status = unsafe { $call };
        $crate::check_cudnn_status(status)?;
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cudnn_version() {
        unsafe {
            let version = cudnnGetVersion();
            println!("cuDNN version: {}", version);
            assert!(version > 0);
        }
    }
    
    #[test]
    fn test_create_destroy_handle() {
        unsafe {
            let mut handle: cudnnHandle_t = std::ptr::null_mut();
            let status = cudnnCreate(&mut handle);
            assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
            assert!(!handle.is_null());
            
            let status = cudnnDestroy(handle);
            assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        }
    }
}