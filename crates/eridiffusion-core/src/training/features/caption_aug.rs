//! Caption-augmentation helpers for the kohya-style tag-list workflow.
//!
//! Phase 6 — wired (not skeleton). This module ships:
//!   1. `shuffle_tags` — comma-separated tag shuffling with a `keep_tags`
//!      prefix so e.g. trigger words stay pinned.
//!   2. `load_filter_list` / `caption_passes` — substring blocklist used at
//!      dataset-prep time to drop captions matching any line in a filter file.
//!
//! Both helpers operate on caption STRINGS. The runtime training loop reads
//! pre-encoded `text_embedding` tensors from the cache, so per-step shuffling
//! requires either re-encoding (Phase 7+) or pre-encoding K alternative
//! variants at prep time. Phase 6 ships the infrastructure only; trainers
//! plumb the `--caption-tag-shuffle` flag for forward-compat but do not
//! consume it. See `FEATURES.md` for the deferred-consumer plan.
//!
//! Filter-list semantics: pure substring (case-sensitive) matching, one
//! pattern per line. Empty lines and lines starting with `#` are skipped.
//! `regex` is intentionally NOT a workspace dep — substring is 80% of the
//! kohya-style use case (drop captions with `nsfw`, `bad_anatomy`, …) and
//! avoids pulling a new transitive crate for one prep-time feature.

use crate::Result;
use rand::seq::SliceRandom;
use rand::Rng;
use std::path::Path;

/// Shuffle the tags of a comma-separated caption.
///
/// `keep_tags` (≥0) leaves the first N tags in place; the remaining tags are
/// permuted via `rng.shuffle`. Whitespace around each tag is trimmed; empty
/// tags after trimming are dropped. Output uses ", " as the separator.
///
/// When `keep_tags` ≥ tag_count or there are <2 shuffleable tags, returns the
/// caption normalized (trimmed tags, ", " separator) but unshuffled.
pub fn shuffle_tags<R: Rng + ?Sized>(caption: &str, keep_tags: usize, rng: &mut R) -> String {
    let tags: Vec<String> = caption
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if tags.len() <= keep_tags + 1 {
        return tags.join(", ");
    }
    let (head, tail) = tags.split_at(keep_tags);
    let mut tail_vec: Vec<String> = tail.to_vec();
    tail_vec.shuffle(rng);
    let mut out: Vec<String> = head.to_vec();
    out.extend(tail_vec);
    out.join(", ")
}

/// Load a caption-filter file. Returns one substring pattern per non-empty,
/// non-comment line. Lines are trimmed; lines beginning with `#` are skipped.
///
/// Returns `Ok(empty vec)` for an empty file (caller can treat that as "no
/// filtering"); errors only on I/O failure.
pub fn load_filter_list(path: &Path) -> Result<Vec<String>> {
    let body = std::fs::read_to_string(path).map_err(|e| {
        crate::EriDiffusionError::Data(format!(
            "caption-filter-list {}: {e}",
            path.display()
        ))
    })?;
    let patterns: Vec<String> = body
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect();
    Ok(patterns)
}

/// `true` if the caption passes all filters (none of `filters` is a substring
/// of `caption`). Empty `filters` always passes.
pub fn caption_passes(caption: &str, filters: &[String]) -> bool {
    !filters.iter().any(|f| caption.contains(f.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::io::Write;

    #[test]
    fn shuffle_keeps_prefix_pinned() {
        let mut rng = StdRng::seed_from_u64(42);
        let cap = "trigger_word, hat, blue_eyes, smile, looking_at_viewer";
        // keep_tags=1 → "trigger_word" must always be first.
        for _ in 0..50 {
            let out = shuffle_tags(cap, 1, &mut rng);
            assert!(out.starts_with("trigger_word, "), "got {out}");
        }
    }

    #[test]
    fn shuffle_actually_shuffles() {
        let mut rng = StdRng::seed_from_u64(7);
        let cap = "a, b, c, d, e, f, g, h";
        // With seed 7 and keep_tags=0, output should differ from input at
        // least once across 10 calls (probability of all-identity is 1/8!^10).
        let mut differed = false;
        for _ in 0..10 {
            let out = shuffle_tags(cap, 0, &mut rng);
            if out != cap {
                differed = true;
                break;
            }
        }
        assert!(differed);
    }

    #[test]
    fn shuffle_normalizes_whitespace_and_empties() {
        let mut rng = StdRng::seed_from_u64(0);
        let cap = "  a  ,, b ,c,,  ";
        let out = shuffle_tags(cap, 3, &mut rng);
        // Only "a", "b", "c" remain after trim+drop-empty; keep_tags=3 → identity.
        assert_eq!(out, "a, b, c");
    }

    #[test]
    fn shuffle_keep_geq_count_is_identity() {
        let mut rng = StdRng::seed_from_u64(0);
        let cap = "one, two";
        // With ≤ keep_tags+1 tags we just return the trimmed identity.
        assert_eq!(shuffle_tags(cap, 5, &mut rng), "one, two");
    }

    #[test]
    fn empty_caption_returns_empty() {
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(shuffle_tags("", 0, &mut rng), "");
        assert_eq!(shuffle_tags("   ,  , ", 0, &mut rng), "");
    }

    #[test]
    fn filter_list_loads_and_skips_comments() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp.as_file(), "# this is a comment").unwrap();
        writeln!(tmp.as_file(), "nsfw").unwrap();
        writeln!(tmp.as_file(), "  ").unwrap();
        writeln!(tmp.as_file(), "bad_anatomy").unwrap();
        let pats = load_filter_list(tmp.path()).unwrap();
        assert_eq!(pats, vec!["nsfw".to_string(), "bad_anatomy".to_string()]);
    }

    #[test]
    fn filter_list_missing_file_errs() {
        let r = load_filter_list(Path::new("/no/such/file/zzz.txt"));
        assert!(r.is_err());
    }

    #[test]
    fn caption_passes_basic() {
        let filters = vec!["nsfw".to_string(), "bad_hand".to_string()];
        assert!(caption_passes("a girl, smile", &filters));
        assert!(!caption_passes("a girl, nsfw, smile", &filters));
        assert!(!caption_passes("bad_handshake", &filters)); // substring match
    }

    #[test]
    fn caption_passes_empty_filters_always_true() {
        let filters: Vec<String> = vec![];
        assert!(caption_passes("anything goes", &filters));
        assert!(caption_passes("", &filters));
    }
}
