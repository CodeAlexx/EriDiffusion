use anyhow::{Result, bail};

pub fn assert_not_text_encoder(keys: &[String]) -> Result<()> {
    let total = keys.len().min(2000);
    if total == 0 { return Ok(()); }
    let mut enc = 0usize;
    for k in keys.iter().take(2000) {
        let prefix = k.split('.').next().unwrap_or("");
        if prefix == "encoder" { enc += 1; }
    }
    if enc * 2 > total { // >50%
        bail!("Refusing to load text-encoder checkpoint: encoder.* = {} of {}", enc, total);
    }
    Ok(())
}

