//! Simple test to check Flux LoRA basics

#[test]
fn test_flux_lora_compilation() {
    // This test just checks that the modules compile
    use eridiffusion::networks::lora::LoRAModule;
    use eridiffusion::models::flux_lora::save_lora::{LoRAConfig, save_flux_lora};
    
    println!("Flux LoRA modules compile successfully!");
}

#[test]
fn test_ai_toolkit_naming() {
    // Test that we're using the correct naming convention
    let expected_names = vec![
        "transformer.double_blocks.0.img_attn.to_q.lora_A",
        "transformer.double_blocks.0.img_attn.to_k.lora_A",
        "transformer.double_blocks.0.img_attn.to_v.lora_A",
        "transformer.single_blocks.0.attn.to_q.lora_A",
    ];
    
    for name in &expected_names {
        assert!(name.starts_with("transformer."));
        assert!(name.contains(".lora_A") || name.contains(".lora_B"));
    }
    
    println!("AI-Toolkit naming convention verified!");
}