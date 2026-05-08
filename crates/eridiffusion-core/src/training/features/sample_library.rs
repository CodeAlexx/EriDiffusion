//! Validation prompt library — JSON file describing a list of prompts to
//! render at each periodic-sample step. Replaces the single
//! `--sample-prompt` / `--sample-seed` pair with an N-prompt × M-seed sweep.
//!
//! Phase target: 2 (skeleton wired into Klein only initially; other trainers
//! pick this up in later phases).
//!
//! Config flag: `--validation-prompts-file <PATH>` → `SampleLibrary`.
//! When unset, the existing single-prompt path is unchanged → byte-identical.
//!
//! JSON formats accepted (either):
//!
//! ```json
//! { "prompts": [ { "prompt": "...", "size": 1024, "seeds": [42, 43] } ] }
//! ```
//!
//! ```json
//! [ { "prompt": "...", "size": 1024 } ]
//! ```

use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// One validation prompt. `negative` defaults to empty string. `size` falls
/// back to the trainer's `--sample-size`. `seeds` falls back to a single
/// `[--sample-seed]` when empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplePrompt {
    pub prompt: String,
    #[serde(default)]
    pub negative: String,
    #[serde(default)]
    pub size: Option<usize>,
    #[serde(default)]
    pub seeds: Vec<u64>,
}

/// JSON wrapper. We accept either `{ "prompts": [...] }` or a bare `[...]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleLibrary {
    pub prompts: Vec<SamplePrompt>,
}

impl SampleLibrary {
    /// Load a prompt library from a JSON file. Tolerates either the wrapped
    /// `{ "prompts": [...] }` form or a bare `[...]` array.
    pub fn from_file(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| {
            crate::EriDiffusionError::Data(format!(
                "validation prompts {}: {e}",
                path.display()
            ))
        })?;
        if let Ok(wrapped) = serde_json::from_str::<SampleLibrary>(&s) {
            return Ok(wrapped);
        }
        let bare: Vec<SamplePrompt> = serde_json::from_str(&s)?;
        Ok(Self { prompts: bare })
    }

    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_wrapped_form() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("v.json");
        std::fs::write(
            &p,
            r#"{"prompts":[{"prompt":"a cat","negative":"","size":1024,"seeds":[42,43]}]}"#,
        )
        .unwrap();
        let lib = SampleLibrary::from_file(&p).unwrap();
        assert_eq!(lib.len(), 1);
        assert_eq!(lib.prompts[0].prompt, "a cat");
        assert_eq!(lib.prompts[0].size, Some(1024));
        assert_eq!(lib.prompts[0].seeds, vec![42, 43]);
    }

    #[test]
    fn parses_bare_array() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("v.json");
        std::fs::write(
            &p,
            r#"[{"prompt":"a dog"},{"prompt":"a fish","size":512}]"#,
        )
        .unwrap();
        let lib = SampleLibrary::from_file(&p).unwrap();
        assert_eq!(lib.len(), 2);
        assert_eq!(lib.prompts[0].prompt, "a dog");
        assert_eq!(lib.prompts[0].size, None);
        assert!(lib.prompts[0].seeds.is_empty());
        assert_eq!(lib.prompts[1].size, Some(512));
    }

    #[test]
    fn missing_file_errs() {
        let r = SampleLibrary::from_file(Path::new("/no/such/zzz.json"));
        assert!(r.is_err());
    }

    #[test]
    fn malformed_errs() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.json");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(b"not json at all").unwrap();
        let r = SampleLibrary::from_file(&p);
        assert!(r.is_err());
    }

    #[test]
    fn missing_prompt_field_errs() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("v.json");
        std::fs::write(&p, r#"[{"size":512}]"#).unwrap();
        let r = SampleLibrary::from_file(&p);
        assert!(r.is_err());
    }
}
