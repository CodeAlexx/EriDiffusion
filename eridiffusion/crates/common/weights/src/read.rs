use anyhow::Result;
use half::{f16, bf16};

pub fn to_vec_f32_from_f16(raw: &[u8]) -> Result<Vec<f32>> {
    let u: &[u16] = bytemuck::try_cast_slice::<u8, u16>(raw)
        .map_err(|e| anyhow::anyhow!("bytemuck cast error: {}", e))?;
    Ok(u.iter().map(|&bits| f16::from_bits(bits).to_f32()).collect())
}

pub fn to_vec_f32_from_bf16(raw: &[u8]) -> Result<Vec<f32>> {
    let u: &[u16] = bytemuck::try_cast_slice::<u8, u16>(raw)
        .map_err(|e| anyhow::anyhow!("bytemuck cast error: {}", e))?;
    Ok(u.iter().map(|&bits| bf16::from_bits(bits).to_f32()).collect())
}

pub fn to_vec_i32(raw: &[u8]) -> Result<Vec<i32>> {
    Ok(bytemuck::try_cast_slice::<u8, i32>(raw)
        .map_err(|e| anyhow::anyhow!("bytemuck cast error: {}", e))?
        .to_vec())
}

/// Bool is stored as U8 in safetensors (0 or 1)
pub fn to_vec_bool_u8(raw: &[u8]) -> Result<Vec<u8>> { // 0 or 1
    Ok(raw.to_vec())
}
