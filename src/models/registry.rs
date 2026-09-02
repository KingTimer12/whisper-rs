//! Map a user-supplied model name to either a local directory or an HF repo.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    Local(PathBuf),
    Hub {
        repo: String,
        /// Repo-relative directory the model's files live under, when a repo
        /// hosts more than one model side by side (e.g. `altunenes/parakeet-rs`,
        /// a bucket of unrelated ONNX exports one subdirectory each). `None`
        /// for every Whisper alias and for any bare `owner/repo` name a
        /// caller types directly -- those files sit at the repo root, exactly
        /// as before this field existed.
        subfolder: Option<String>,
    },
}

/// Alias to Hugging Face repo, plus an optional repo-relative subfolder for
/// aliases whose files don't live at the repo root.
const ALIASES: &[(&str, &str, Option<&str>)] = &[
    ("tiny", "Systran/faster-whisper-tiny", None),
    ("tiny.en", "Systran/faster-whisper-tiny.en", None),
    ("base", "Systran/faster-whisper-base", None),
    ("base.en", "Systran/faster-whisper-base.en", None),
    ("small", "Systran/faster-whisper-small", None),
    ("small.en", "Systran/faster-whisper-small.en", None),
    ("medium", "Systran/faster-whisper-medium", None),
    ("medium.en", "Systran/faster-whisper-medium.en", None),
    ("large-v1", "Systran/faster-whisper-large-v1", None),
    ("large-v2", "Systran/faster-whisper-large-v2", None),
    ("large-v3", "Systran/faster-whisper-large-v3", None),
    ("large", "Systran/faster-whisper-large-v3", None),
    ("distil-large-v3", "distil-whisper/distil-large-v3-ct2", None),
    // Not NVIDIA's own repo: NVIDIA publishes only NeMo/safetensors/GGUF
    // artifacts (confirmed via the HF API's file listing), not the ONNX
    // encoder/decoder split `parakeet-rs` requires. `altunenes/parakeet-rs`
    // is the `parakeet-rs` author's own export bucket, one subdirectory per
    // model (see that repo's README, `Setup` section); this is the
    // Multilingual 3.5 variant's subdirectory within it.
    (
        "nemotron",
        "altunenes/parakeet-rs",
        Some("nemotron-3.5-asr-streaming-0.6b-onnx"),
    ),
];

pub fn resolve(name: &str) -> ModelRef {
    let as_path = Path::new(name);
    if as_path.is_dir() {
        return ModelRef::Local(as_path.to_path_buf());
    }

    if let Some((_, repo, subfolder)) = ALIASES.iter().find(|(alias, _, _)| *alias == name) {
        return ModelRef::Hub {
            repo: (*repo).to_string(),
            subfolder: subfolder.map(str::to_string),
        };
    }

    ModelRef::Hub {
        repo: name.to_string(),
        subfolder: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_aliases_map_to_systran_repos() {
        assert_eq!(
            resolve("large-v3"),
            ModelRef::Hub { repo: "Systran/faster-whisper-large-v3".into(), subfolder: None }
        );
        assert_eq!(
            resolve("tiny"),
            ModelRef::Hub { repo: "Systran/faster-whisper-tiny".into(), subfolder: None }
        );
        assert_eq!(
            resolve("medium"),
            ModelRef::Hub { repo: "Systran/faster-whisper-medium".into(), subfolder: None }
        );
    }

    #[test]
    fn distil_alias_maps_to_the_distil_whisper_org() {
        assert_eq!(
            resolve("distil-large-v3"),
            ModelRef::Hub { repo: "distil-whisper/distil-large-v3-ct2".into(), subfolder: None }
        );
    }

    #[test]
    fn nemotron_alias_maps_to_the_parakeet_rs_export_bucket_subfolder() {
        assert_eq!(
            resolve("nemotron"),
            ModelRef::Hub {
                repo: "altunenes/parakeet-rs".into(),
                subfolder: Some("nemotron-3.5-asr-streaming-0.6b-onnx".into()),
            }
        );
    }

    #[test]
    fn a_name_with_a_slash_is_an_explicit_repo() {
        assert_eq!(
            resolve("someone/my-whisper-ct2"),
            ModelRef::Hub { repo: "someone/my-whisper-ct2".into(), subfolder: None }
        );
    }

    #[test]
    fn an_existing_directory_wins_over_everything() {
        let dir = std::env::temp_dir().join("whisper_rs_t7_tiny");
        std::fs::create_dir_all(&dir).unwrap();

        let resolved = resolve(dir.to_str().unwrap());

        assert_eq!(resolved, ModelRef::Local(dir));
    }

    #[test]
    fn an_unknown_name_without_a_slash_is_still_treated_as_a_repo() {
        // Better a clear 404 from the hub than a silent wrong alias.
        assert_eq!(
            resolve("not-a-real-model"),
            ModelRef::Hub { repo: "not-a-real-model".into(), subfolder: None }
        );
    }
}
