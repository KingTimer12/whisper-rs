//! Map a user-supplied model name to either a local directory or an HF repo.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    Local(PathBuf),
    Hub { repo: String },
}

/// Alias to Hugging Face repo. Resolution order is local path, alias, then repo.
const ALIASES: &[(&str, &str)] = &[
    ("tiny", "Systran/faster-whisper-tiny"),
    ("tiny.en", "Systran/faster-whisper-tiny.en"),
    ("base", "Systran/faster-whisper-base"),
    ("base.en", "Systran/faster-whisper-base.en"),
    ("small", "Systran/faster-whisper-small"),
    ("small.en", "Systran/faster-whisper-small.en"),
    ("medium", "Systran/faster-whisper-medium"),
    ("medium.en", "Systran/faster-whisper-medium.en"),
    ("large-v1", "Systran/faster-whisper-large-v1"),
    ("large-v2", "Systran/faster-whisper-large-v2"),
    ("large-v3", "Systran/faster-whisper-large-v3"),
    ("large", "Systran/faster-whisper-large-v3"),
    ("distil-large-v3", "distil-whisper/distil-large-v3-ct2"),
    ("nemotron", "nvidia/nemotron-3.5-asr-streaming-0.6b"),
];

pub fn resolve(name: &str) -> ModelRef {
    let as_path = Path::new(name);
    if as_path.is_dir() {
        return ModelRef::Local(as_path.to_path_buf());
    }

    if let Some((_, repo)) = ALIASES.iter().find(|(alias, _)| *alias == name) {
        return ModelRef::Hub {
            repo: (*repo).to_string(),
        };
    }

    ModelRef::Hub {
        repo: name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_aliases_map_to_systran_repos() {
        assert_eq!(
            resolve("large-v3"),
            ModelRef::Hub { repo: "Systran/faster-whisper-large-v3".into() }
        );
        assert_eq!(
            resolve("tiny"),
            ModelRef::Hub { repo: "Systran/faster-whisper-tiny".into() }
        );
        assert_eq!(
            resolve("medium"),
            ModelRef::Hub { repo: "Systran/faster-whisper-medium".into() }
        );
    }

    #[test]
    fn distil_alias_maps_to_the_distil_whisper_org() {
        assert_eq!(
            resolve("distil-large-v3"),
            ModelRef::Hub { repo: "distil-whisper/distil-large-v3-ct2".into() }
        );
    }

    #[test]
    fn nemotron_alias_maps_to_the_nvidia_repo() {
        assert_eq!(
            resolve("nemotron"),
            ModelRef::Hub { repo: "nvidia/nemotron-3.5-asr-streaming-0.6b".into() }
        );
    }

    #[test]
    fn a_name_with_a_slash_is_an_explicit_repo() {
        assert_eq!(
            resolve("someone/my-whisper-ct2"),
            ModelRef::Hub { repo: "someone/my-whisper-ct2".into() }
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
            ModelRef::Hub { repo: "not-a-real-model".into() }
        );
    }
}
