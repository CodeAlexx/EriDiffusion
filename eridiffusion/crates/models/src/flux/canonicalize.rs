//! Helpers for mapping assorted Flux checkpoint layouts onto the canonical
//! `blocks.{:02}.…` scheme.  Legacy exports from SimpleTuner used prefixes such
//! as `double_blocks.{i}` or `single_blocks.{i}`; we normalise those so the
//! model loader can reason about block ordering without bespoke match arms.

/// Convert a raw safetensor key into the canonical "blocks.{:02}.…" form.
pub fn canonicalize_key(raw: &str) -> String {
    if let Some(rest) = raw.strip_prefix("double_blocks.") {
        normalise_block_prefix(rest)
    } else if let Some(rest) = raw.strip_prefix("single_blocks.") {
        normalise_block_prefix(rest)
    } else if let Some(rest) = raw.strip_prefix("blocks.") {
        normalise_block_prefix(rest)
    } else {
        raw.to_string()
    }
}

/// Map every key through [`canonicalize_key`].
pub fn canonicalize_list<I>(keys: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    keys.into_iter().map(|k| canonicalize_key(&k)).collect()
}

/// Best-effort block index extraction used by diagnostics and loaders.
pub fn block_index(raw: &str) -> Option<usize> {
    let canonical = canonicalize_key(raw);
    canonical
        .strip_prefix("blocks.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(idx, _)| idx.parse::<usize>().ok())
}

fn normalise_block_prefix(rest: &str) -> String {
    if let Some((idx, suffix)) = rest.split_once('.') {
        if let Ok(num) = idx.parse::<usize>() {
            return format!("blocks.{:02}.{}", num, suffix);
        }
    }
    rest.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalises_double_blocks() {
        assert_eq!(canonicalize_key("double_blocks.3.img_attn.qkv.weight"), "blocks.03.img_attn.qkv.weight");
    }

    #[test]
    fn canonicalises_single_blocks() {
        assert_eq!(canonicalize_key("single_blocks.12.img_mlp.0.weight"), "blocks.12.img_mlp.0.weight");
    }

    #[test]
    fn leaves_unknown_prefixes() {
        assert_eq!(canonicalize_key("other.stuff"), "other.stuff");
    }

    #[test]
    fn extracts_block_index() {
        assert_eq!(block_index("double_blocks.7.foo"), Some(7));
        assert_eq!(block_index("blocks.02.bar"), Some(2));
        assert_eq!(block_index("weird"), None);
    }
}
