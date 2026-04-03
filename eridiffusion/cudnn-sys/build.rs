use std::env;
use std::path::PathBuf;

fn main() {
    // Find CUDA installation
    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    // Add CUDA library paths
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    println!("cargo:rustc-link-search=native={}/lib", cuda_home);
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    
    // Add cuDNN paths from pip installation
    let cudnn_lib_path = "/home/alex/SimpleTuner/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib";
    let cudnn_include_path = "/home/alex/SimpleTuner/.venv/lib/python3.11/site-packages/nvidia/cudnn/include";
    
    println!("cargo:rustc-link-search=native={}", cudnn_lib_path);
    println!("cargo:include={}", cudnn_include_path);
    
    // Link cuDNN libraries - use versioned names
    println!("cargo:rustc-link-lib=cudnn");
    println!("cargo:rustc-link-lib=cudart");
    
    // Alternative: link specific cuDNN libraries if main one fails
    // println!("cargo:rustc-link-lib=cudnn_ops");
    // println!("cargo:rustc-link-lib=cudnn_cnn");
    // println!("cargo:rustc-link-lib=cudnn_adv");
    
    // Set up include paths
    let cuda_include = PathBuf::from(&cuda_home).join("include");
    println!("cargo:include={}", cuda_include.display());
    
    // Rerun if environment changes
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDNN_PATH");
    
    // Check for cuDNN availability via environment variable
    if let Ok(cudnn_path) = env::var("CUDNN_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cudnn_path);
        println!("cargo:rustc-link-search=native={}/lib", cudnn_path);
    }
}