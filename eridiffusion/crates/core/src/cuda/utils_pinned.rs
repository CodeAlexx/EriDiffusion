use core::ffi::c_void;

pub use flame_core::{PinnedAllocFlags, PinnedHostBuffer, PinnedPool};
use flame_core::{memcpy_async_device_to_host, memcpy_async_host_to_device};

#[deprecated(note = "use flame_core pinned API")]
pub struct Pinned {
    buffer: Option<PinnedHostBuffer<u8>>,
    bytes: usize,
}

#[allow(deprecated)]
impl Pinned {
    pub fn alloc(bytes: usize) -> Option<Self> {
        let buffer = PinnedHostBuffer::<u8>::with_capacity_elems(bytes, PinnedAllocFlags::WRITE_COMBINED).ok()?;
        Some(Self {
            buffer: Some(buffer),
            bytes,
        })
    }

    #[inline]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    #[inline]
    pub fn ptr(&self) -> *mut c_void {
        self.buffer
            .as_ref()
            .map(|b| b.as_ptr() as *mut c_void)
            .unwrap_or(core::ptr::null_mut())
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.buffer
            .as_mut()
            .map(|b| b.as_mut_ptr() as *mut c_void)
            .unwrap_or(core::ptr::null_mut())
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        self.buffer.as_mut().map(|b| b.as_mut_bytes())
    }

    /// Finalize and return the underlying pinned buffer.
    pub fn into_buffer(mut self) -> PinnedHostBuffer<u8> {
        self.buffer.take().expect("Pinned buffer already taken")
    }

    /// # Safety
    /// Equivalent to dropping the underlying buffer. After calling this, the
    /// raw pointer must not be used.
    pub unsafe fn free(mut self) {
        self.buffer.take();
    }
}

pub fn memcpy_h2d_async(dst_dev: *mut c_void, src_host: *const c_void, bytes: usize, stream: *mut c_void) -> i32 {
    match memcpy_async_host_to_device(dst_dev, src_host, bytes, stream) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("[pinned] memcpy_h2d_async failed: {err}");
            -1
        }
    }
}

pub fn memcpy_d2h_async(dst_host: *mut c_void, src_dev: *const c_void, bytes: usize, stream: *mut c_void) -> i32 {
    match memcpy_async_device_to_host(dst_host, src_dev, bytes, stream) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("[pinned] memcpy_d2h_async failed: {err}");
            -1
        }
    }
}
