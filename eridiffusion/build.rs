use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/rms_norm.cu");
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Only compile CUDA kernels if cuda feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda_kernels(&out_dir);
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels(out_dir: &PathBuf) {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    let nvcc_path = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    
    // Compile CUDA kernel to PTX
    let status = std::process::Command::new(&nvcc_path)
        .args(&[
            "-ptx",
            "-o",
            out_dir.join("rms_norm.ptx").to_str().unwrap(),
            "src/kernels/rms_norm.cu",
            "-arch=sm_70", // Minimum for Tensor Cores
            "--use_fast_math",
            "-O3",
        ])
        .status()
        .expect("Failed to execute nvcc");
    
    if !status.success() {
        panic!("CUDA compilation failed");
    }
    
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
}