// Optional CUDA pinned host memory FFI (scaffold)

extern "C" {
    pub fn cuda_alloc_pinned_host(size: usize) -> *mut core::ffi::c_void;
    pub fn cuda_free_pinned_host(ptr: *mut core::ffi::c_void);
    /// kind: 1=H2D, 2=D2H, 3=D2D
    pub fn cuda_memcpy_async(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        size: usize,
        kind: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

