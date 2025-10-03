use regex::Regex;
use adapters::adapter::AdapterSet;
use flame_core::{Parameter, Result as FlameResult};
use adapters::loader::BaseShapes;
use std::collections::HashMap;
use eridiffusion_core::Tensor;

pub fn compile_globs(globs: &[String]) -> Vec<Regex> {
    globs.iter().filter_map(|g| glob_to_regex(g).ok()).collect()
}

fn glob_to_regex(glob: &str) -> Result<Regex, regex::Error> {
    let mut out = String::from("^");
    for ch in glob.chars() {
        match ch {
            '*' => out.push_str(".*"),
            '.' => out.push_str("\\."),
            '(' | ')' | '|' => out.push(ch),
            _ => out.push(ch),
        }
    }
    out.push('$');
    Regex::new(&out)
}

pub fn filter_adapters(set: &AdapterSet, allow: &[String], deny: &[String]) -> AdapterSet {
    let allow_re = if allow.is_empty() { Vec::<Regex>::new() } else { compile_globs(allow) };
    let deny_re = if deny.is_empty() { Vec::<Regex>::new() } else { compile_globs(deny) };
    let mut out = AdapterSet::new();
    for (k, a) in &set.by_target {
        let allowed = if allow_re.is_empty() { true } else { allow_re.iter().any(|re| re.is_match(k)) };
        let denied = deny_re.iter().any(|re| re.is_match(k));
        if allowed && !denied {
            out.insert(k.clone(), a.clone());
        }
    }
    out
}

pub fn to_parameters(set: &AdapterSet) -> FlameResult<Vec<Parameter>> {
    let mut out = Vec::new();
    for t in set.params() {
        let p = Parameter::new(t.requires_grad_(true));
        out.push(p);
    }
    Ok(out)
}

pub fn base_shapes_from_weights(weights: &HashMap<String, Tensor>) -> BaseShapes {
    let mut bs = BaseShapes::default();
    for (k, t) in weights {
        let dims = t.shape().dims().iter().map(|&d| d as i64).collect::<Vec<_>>();
        let is_conv = t.shape().dims().len()==4;
        bs.by_target.insert(k.clone(), (dims, is_conv));
    }
    bs
}
