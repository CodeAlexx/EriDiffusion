use anyhow::{Result, bail};
use flame_core::DType;
use eridiffusion_core::Device;
use eridiffusion_common_weights as cw;
use hashbrown::HashMap;
use std::sync::Arc;
use crate::adapter::AdapterSet;
use crate::kinds::{lora::LoRA, locon2d::LoCon2d};
#[cfg(feature = "experimental_lyco")]
use crate::kinds::{loha::LoHa, lokr::LoKr, dora::DoRA, ia3::IA3};
use super::BaseShapes;

fn load_one_file(path: &std::path::Path, _device: &Device, _dtype: DType, base: &BaseShapes, set: &mut AdapterSet) -> Result<()> {
    let mut ld = cw::SafeLoader::open(path.to_str().ok_or_else(|| anyhow::anyhow!("non-utf8 path"))?)?;
    let keys = ld.list_keys()?;
    let mut used = HashMap::new();
    // Group by target
    let mut groups: HashMap<String, HashMap<String, String>> = HashMap::new();
    for k in keys {
        if !(k.ends_with(".weight") || k.ends_with(".alpha")) { continue; }
        let parts: Vec<&str> = k.split('.').collect();
        if parts.len() < 3 { continue; }
        let target = parts[..parts.len()-2].join(".");
        let leaf = parts[parts.len()-2..].join(".");
        groups.entry(target).or_default().insert(leaf, k);
    }
    for (target, g) in groups {
        // Down/Up pairs
        let is_lora = g.contains_key("lora_down.weight") && g.contains_key("lora_up.weight");
        let is_locon = g.contains_key("conv_down.weight") && g.contains_key("conv_up.weight");
        #[cfg(feature = "experimental_lyco")]
        let is_loha = g.contains_key("hada_t1") && g.contains_key("hada_t2") && g.contains_key("hada_w1") && g.contains_key("hada_w2");
        #[cfg(feature = "experimental_lyco")]
        let is_lokr = g.keys().any(|k| k.contains("lokr_"));
        #[cfg(feature = "experimental_lyco")]
        let is_dora = is_lora && g.contains_key("dora_scale");
        #[cfg(feature = "experimental_lyco")]
        let is_ia3 = g.contains_key("ia3_in") || g.contains_key("ia3_out");
        if !is_lora && !is_locon {
            #[cfg(feature = "experimental_lyco")]
            if !(is_loha || is_lokr || is_dora || is_ia3) { continue; }
            #[cfg(not(feature = "experimental_lyco"))]
            { continue; }
        }
        let (wshape, is_conv) = base.by_target.get(&target).cloned().unwrap_or((vec![], false));
        if is_lora {
            if is_conv { bail!("target '{}' expects conv, got LoRA", target); }
            // base [IN,OUT]
            if wshape.len()!=2 { bail!("base shape for {} invalid: {:?}", target, wshape); }
            let a_k = g.get("lora_down.weight").unwrap();
            let b_k = g.get("lora_up.weight").unwrap();
            let a = ld.get_bf16(a_k)?; // [IN, R]
            let b = ld.get_bf16(b_k)?; // [R, OUT]
            let r = a.shape().dims()[1] as usize;
            let alpha = if let Some(a_k) = g.get("alpha") {
                ld.get_as(a_k, DType::F32).ok()
                    .and_then(|t| t.to_vec().ok())
                    .and_then(|v: Vec<f32>| v.first().copied())
                    .unwrap_or(r as f32)
            } else { r as f32 };
            let name = format!("{}:lora", target);
            let l = LoRA::new(name, a, b, alpha, r);
            set.insert(target.clone(), Arc::new(l));
            used.insert(a_k.clone(), true); used.insert(b_k.clone(), true);
        } else if is_locon {
            if !is_conv { bail!("target '{}' expects linear, got LoCon2d", target); }
            // base [KH,KW,IC,OC]
            if wshape.len()!=4 { bail!("base shape for {} invalid: {:?}", target, wshape); }
            let (kh,kw,ic,oc) = (wshape[0] as usize, wshape[1] as usize, wshape[2] as usize, wshape[3] as usize);
            let a_k = g.get("conv_down.weight").unwrap();
            let b_k = g.get("conv_up.weight").unwrap();
            let a = ld.get_bf16(a_k)?; // [OC, R]
            let b = ld.get_bf16(b_k)?; // [R, IC*KH*KW]
            let r = a.shape().dims()[1] as usize;
            let alpha = if let Some(a_k) = g.get("alpha") {
                ld.get_as(a_k, DType::F32).ok()
                    .and_then(|t| t.to_vec().ok())
                    .and_then(|v: Vec<f32>| v.first().copied())
                    .unwrap_or(r as f32)
            } else { r as f32 };
            let name = format!("{}:locon2d", target);
            let l = LoCon2d::new(name, a, b, alpha, r, kh, kw, ic, oc);
            set.insert(target.clone(), Arc::new(l));
            used.insert(a_k.clone(), true); used.insert(b_k.clone(), true);
        }
        #[cfg(feature = "experimental_lyco")]
        if is_loha {
            let a = ld.get_bf16(g.get("hada_t1").unwrap())?;
            let b = ld.get_bf16(g.get("hada_t2").unwrap())?;
            let c = ld.get_bf16(g.get("hada_w1").unwrap())?;
            let d = ld.get_bf16(g.get("hada_w2").unwrap())?;
            let r1 = a.shape().dims()[1] as usize; let r2 = c.shape().dims()[1] as usize;
            let alpha = r1.max(r2) as f32;
            let name = format!("{}:loha", target);
            set.insert(target.clone(), Arc::new(LoHa::new(name, a, b, c, d, alpha, r1, r2)));
        }
        #[cfg(feature = "experimental_lyco")]
        if is_lokr {
            if let (Some(pk), Some(qk), Some(rk)) = (g.get("lokr_p"), g.get("lokr_q"), g.get("lokr_r")) {
                let p = ld.get_bf16(pk)?; let q = ld.get_bf16(qk)?; let r = ld.get_bf16(rk)?;
                let name = format!("{}:lokr", target);
            set.insert(target.clone(), Arc::new(LoKr::new(name, p, q, r, 1.0)));
            }
        }
        #[cfg(feature = "experimental_lyco")]
        if is_dora {
            let a = ld.get_bf16(g.get("lora_down.weight").unwrap())?;
            let b = ld.get_bf16(g.get("lora_up.weight").unwrap())?;
            let r = a.shape().dims()[1] as usize;
            let alpha = ld.get_f32(g.get("alpha").unwrap_or(&"".into())).unwrap_or(r as f32);
            let gain = ld.get_bf16(g.get("dora_scale").unwrap()).ok();
            let name = format!("{}:dora", target);
            set.insert(target.clone(), Arc::new(DoRA::new(name, a, b, alpha, r, gain)));
        }
        #[cfg(feature = "experimental_lyco")]
        if is_ia3 {
            let v_in = g.get("ia3_in").and_then(|k| ld.get_bf16(k).ok());
            let v_out= g.get("ia3_out").and_then(|k| ld.get_bf16(k).ok());
            let name = format!("{}:ia3", target);
            set.insert(target.clone(), Arc::new(IA3::new(name, v_in, v_out)));
        }
    }
    // Strict: error on unknown/missing/unused adapter keys
    // Mark recognized keys as used above; now iterate all file keys and ensure each belongs to a recognized group/leaf
    let all_keys = ld.list_keys()?;
    let mut leftovers = Vec::new();
    for k in all_keys {
        if k.ends_with(".weight") || k.ends_with(".alpha") {
            if !used.contains_key(&k) {
                leftovers.push(k);
            }
        }
    }
    if !leftovers.is_empty() {
        bail!("LyCORIS strict loader: unused/unknown keys (first 10): {:?}", &leftovers[..leftovers.len().min(10)]);
    }
    Ok(())
}

pub fn load_lycoris_dir(path:&std::path::Path, device:&Device, dtype:DType, base_shapes:&BaseShapes) -> Result<AdapterSet> {
    let mut set = AdapterSet::new();
    if path.is_file() {
        load_one_file(path, device, dtype, base_shapes, &mut set)?;
    } else {
        for entry in std::fs::read_dir(path)? {
            let p = entry?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                load_one_file(&p, device, dtype, base_shapes, &mut set)?;
            }
        }
    }
    Ok(set)
}
