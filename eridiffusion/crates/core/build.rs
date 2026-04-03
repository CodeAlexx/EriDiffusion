fn main() {
    // Compile CUDA narrow kernels into a static library
    let mut build = cc::Build::new();
    build.cuda(true);
    // Target architecture; adjust or make configurable if needed
    build.flag("-arch=sm_80");
    // Enable fast math optionally
    // build.flag("-use_fast_math");

    build.file("cuda/narrow_strided.cu");
    build.file("cuda/narrow_strided_backward.cu");
    build.file("cuda/utils_pinned.cu");

    // Compile
    build.compile("eridiffusion_core_cuda");

    // Re-run if sources change
    println!("cargo:rerun-if-changed=cuda/narrow_strided.cu");
    println!("cargo:rerun-if-changed=cuda/narrow_strided_backward.cu");
    println!("cargo:rerun-if-changed=cuda/utils_pinned.cu");
}
