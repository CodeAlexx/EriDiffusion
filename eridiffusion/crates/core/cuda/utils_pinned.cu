#include <cuda_runtime.h>
#include <stdint.h>

extern "C" void* cuda_alloc_pinned_host(size_t size) {
    void* p = nullptr;
    cudaError_t e = cudaHostAlloc(&p, size, cudaHostAllocPortable);
    return (e == cudaSuccess) ? p : nullptr;
}
extern "C" int cuda_free_pinned_host(void* p) {
    return (int)cudaFreeHost(p);
}
extern "C" int cuda_memcpy_async(void* dst, const void* src, size_t bytes, int kind, void* stream_void) {
    cudaMemcpyKind k = cudaMemcpyDefault;
    if (kind == 1) k = cudaMemcpyHostToDevice;
    else if (kind == 2) k = cudaMemcpyDeviceToHost;
    else if (kind == 3) k = cudaMemcpyDeviceToDevice;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    cudaError_t e = cudaMemcpyAsync(dst, src, bytes, k, stream);
    return (int)e;
}
