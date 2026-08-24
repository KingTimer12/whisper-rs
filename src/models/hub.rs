//! Fetch CTranslate2 Whisper models from Hugging Face, or use a local directory.

use crate::error::{Error, Result};
use crate::models::registry::{resolve, ModelRef};
use hf_hub::{split_id, HFClient, HFClientSync, HFError};
use std::path::{Path, PathBuf};

/// Files CTranslate2 needs. `model.bin` and `config.json` are mandatory; the
/// rest are fetched when present, since repos differ in tokenizer layout.
const REQUIRED: &[&str] = &["model.bin", "config.json"];
/// A repo must ship at least one of these -- `ct2rs`'s HF tokenizer loader
/// needs a `tokenizer.json`, or a `vocabulary.json`/`vocabulary.txt` it can
/// build one from. `preprocessor_config.json` is not in this list: it is
/// truly optional, since `ensure_preprocessor_config` synthesizes it when
/// absent (see that function's docs).
const TOKENIZER_FILES: &[&str] = &["tokenizer.json", "vocabulary.json", "vocabulary.txt"];
const OPTIONAL: &[&str] = &[
    "tokenizer.json",
    "vocabulary.json",
    "vocabulary.txt",
    "preprocessor_config.json",
];

const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";

/// Mel bin counts used by `ct2rs`'s built-in mel-filterbank generator
/// (`mel_spec::mel::mel`, called by `PreprocessorConfig::read` whenever
/// `mel_filters` is absent from the JSON). Whisper large-v3 and its distils
/// switched from 80 to 128 mel bins; every earlier size (tiny/base/small/
/// medium/large-v1/large-v2) uses 80.
const MEL_BINS_LARGE_V3: usize = 128;
const MEL_BINS_DEFAULT: usize = 80;

#[derive(Debug, Clone, Default)]
pub struct FetchOptions {
    pub download_root: Option<PathBuf>,
    pub local_files_only: bool,
}

/// Resolve `name` to a local directory containing a usable CTranslate2 model,
/// downloading from the hub when needed.
pub fn ensure_model(name: &str, opts: &FetchOptions) -> Result<PathBuf> {
    match resolve(name) {
        ModelRef::Local(dir) => {
            validate_dir(name, &dir)?;
            // No repo id here, just whatever the caller passed (path or alias);
            // that is still the best signal we have for mel-bin count.
            ensure_preprocessor_config(&dir, name)?;
            Ok(dir)
        }
        ModelRef::Hub { repo } => {
            if opts.local_files_only {
                return Err(Error::ModelNotFound {
                    name: name.to_string(),
                    path: opts
                        .download_root
                        .clone()
                        .unwrap_or_else(|| PathBuf::from("<hf cache>")),
                    message: "local_files_only is set, so nothing was downloaded".into(),
                });
            }
            let dir = download(name, &repo, opts)?;
            validate_dir(name, &dir)?;
            ensure_preprocessor_config(&dir, &repo)?;
            Ok(dir)
        }
    }
}

/// Write a synthetic `preprocessor_config.json` when the model directory
/// doesn't have one.
///
/// `ct2rs::Whisper::new` (ct2rs 0.10) unconditionally opens
/// `<model_dir>/preprocessor_config.json` and fails with a raw "No such file
/// or directory" I/O error if it is missing -- there is no fallback in that
/// crate. The standard `Systran/faster-whisper-*` Hugging Face repos this
/// crate downloads from do not ship that file at all (confirmed via HF hub's
/// own "confirmed absent" cache marker), because upstream `faster-whisper`
/// works around the gap by vendoring its own copy as a package asset instead
/// of fetching it from the hub. We have no such bundled asset, so we
/// synthesize the file here instead. This runs for both the download path
/// and the local-directory path, since a user's own converted directory can
/// be missing it too.
///
/// Every field but `feature_size` (mel bin count) is a fixed Whisper
/// feature-extractor constant. `mel_filters` is deliberately omitted (written
/// as `null`): `ct2rs`'s reader treats an absent/null `mel_filters` as "compute
/// it yourself" and calls the same `mel_spec::mel::mel` filterbank generator
/// CTranslate2's own converter would have used, keyed only off
/// `sampling_rate`, `n_fft`, and `feature_size` -- so we don't need to ship a
/// filterbank matrix at all, just the three scalars that determine it.
///
/// Only `feature_size` needs a real decision: it is not recoverable from the
/// CT2 `config.json` (checked against a real downloaded `tiny` model -- that
/// file carries `alignment_heads`/`lang_ids` but nothing about mel bins), so
/// it is inferred from the resolved model identity (repo id for hub models,
/// the caller-supplied name/path for local ones): anything naming
/// `large-v3` (which also matches `distil-large-v3`) uses 128 mel bins;
/// everything else (tiny/base/small/medium/large-v1/large-v2) uses 80, which
/// is also `ct2rs`'s own struct-level default via `mel_spec`.
fn ensure_preprocessor_config(dir: &Path, identity: &str) -> Result<()> {
    let path = dir.join(PREPROCESSOR_CONFIG_FILE);
    if path.is_file() {
        // The repo shipped one, or a previous run already synthesized one:
        // never overwrite -- theirs (or the earlier synthesis) is authoritative.
        return Ok(());
    }

    let feature_size = if identity.to_ascii_lowercase().contains("large-v3") {
        MEL_BINS_LARGE_V3
    } else {
        MEL_BINS_DEFAULT
    };

    // This writes into the model directory, which for `ModelRef::Local` is a
    // directory the *user* owns (not our download cache), and the choice of
    // `feature_size` is a heuristic guess from a name, not a certainty. If
    // that guess is wrong -- e.g. a locally-converted 128-bin (large-v3)
    // model living in a directory whose path never mentions "large-v3" --
    // the wrong value gets persisted once and, because synthesis never
    // overwrites an existing file, silently stays wrong on every future load
    // until the user deletes it by hand. Warning loudly at the point of
    // writing, naming both the chosen value and the exact path, is the only
    // way that mistake is ever surfaced.
    tracing::warn!(
        "synthesizing {path} (feature_size={feature_size}) for model {identity:?}: ct2rs \
         requires this file and it was missing; feature_size is guessed from the model name/path \
         (\"large-v3\" => 128, otherwise 80) and, once written, will not be regenerated -- delete \
         {path} by hand and re-run if this guess is wrong for your model",
        path = path.display(),
    );

    let json = format!(
        r#"{{
  "chunk_length": 30,
  "feature_extractor_type": "WhisperFeatureExtractor",
  "feature_size": {feature_size},
  "hop_length": 160,
  "n_fft": 400,
  "n_samples": 480000,
  "nb_max_frames": 3000,
  "padding_side": "right",
  "padding_value": 0.0,
  "processor_class": "WhisperProcessor",
  "return_attention_mask": false,
  "sampling_rate": 16000,
  "mel_filters": null
}}
"#
    );

    std::fs::write(&path, json).map_err(|e| Error::ModelNotFound {
        name: identity.to_string(),
        path: dir.to_path_buf(),
        message: format!(
            "failed to write synthesized {PREPROCESSOR_CONFIG_FILE}: {e}"
        ),
    })
}

fn download(name: &str, repo: &str, opts: &FetchOptions) -> Result<PathBuf> {
    let mut builder = HFClient::builder();
    if let Some(root) = &opts.download_root {
        builder = builder.cache_dir(root.clone());
    }
    let client = builder.build().map_err(|e| Error::Download {
        name: name.to_string(),
        message: e.to_string(),
    })?;
    let client = HFClientSync::from_inner(client).map_err(|e| Error::Download {
        name: name.to_string(),
        message: e.to_string(),
    })?;

    let (owner, repo_name) = split_id(repo);
    let api_repo = client.model(owner, repo_name);

    let mut dir: Option<PathBuf> = None;

    for file in REQUIRED {
        let path = api_repo
            .download_file()
            .filename(*file)
            .send()
            .map_err(|e| Error::Download {
                name: name.to_string(),
                message: format!("{file}: {e}"),
            })?;
        if dir.is_none() {
            dir = path.parent().map(Path::to_path_buf);
        }
    }

    for file in OPTIONAL {
        // Absent optional files are normal; repos differ in tokenizer layout.
        if let Err(e) = api_repo.download_file().filename(*file).send()
            && !matches!(e, HFError::EntryNotFound { .. })
        {
            tracing::debug!("optional file {file} not fetched for {repo}: {e}");
        }
    }

    dir.ok_or_else(|| Error::Download {
        name: name.to_string(),
        message: "downloaded files have no parent directory".into(),
    })
}

/// Fail early, and clearly, rather than letting CTranslate2 abort in C++.
pub fn validate_dir(name: &str, dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        return Err(Error::ModelNotFound {
            name: name.to_string(),
            path: dir.to_path_buf(),
            message: "not a directory".into(),
        });
    }

    let missing: Vec<&str> = REQUIRED
        .iter()
        .copied()
        .filter(|f| !dir.join(f).is_file())
        .collect();

    if !missing.is_empty() {
        return Err(Error::ModelNotFound {
            name: name.to_string(),
            path: dir.to_path_buf(),
            message: format!("missing required file(s): {}", missing.join(", ")),
        });
    }

    // A repo shipping none of these passes every other check here and then
    // dies inside CTranslate2's C++ tokenizer loader with exactly the opaque
    // native error this function exists to prevent -- so require at least
    // one up front instead.
    if !TOKENIZER_FILES.iter().any(|f| dir.join(f).is_file()) {
        return Err(Error::ModelNotFound {
            name: name.to_string(),
            path: dir.to_path_buf(),
            message: format!(
                "missing a tokenizer file: need at least one of {}",
                TOKENIZER_FILES.join(", ")
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("whisper_rs_t8_{tag}"));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn validate_accepts_a_directory_with_the_required_files() {
        let d = temp_dir("ok");
        std::fs::write(d.join("model.bin"), b"x").unwrap();
        std::fs::write(d.join("config.json"), b"{}").unwrap();
        std::fs::write(d.join("tokenizer.json"), b"{}").unwrap();

        validate_dir("tiny", &d).unwrap();
    }

    #[test]
    fn validate_rejects_a_directory_without_any_tokenizer_file() {
        let d = temp_dir("no_tokenizer");
        std::fs::write(d.join("model.bin"), b"x").unwrap();
        std::fs::write(d.join("config.json"), b"{}").unwrap();
        // Deliberately no tokenizer.json / vocabulary.json / vocabulary.txt.

        let err = validate_dir("tiny", &d).unwrap_err();

        match err {
            Error::ModelNotFound { message, .. } => assert!(
                message.contains("tokenizer"),
                "the message must explain the missing tokenizer file, got: {message}"
            ),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_any_single_tokenizer_variant() {
        for variant in TOKENIZER_FILES {
            let d = temp_dir(&format!("tokenizer_variant_{variant}"));
            std::fs::write(d.join("model.bin"), b"x").unwrap();
            std::fs::write(d.join("config.json"), b"{}").unwrap();
            std::fs::write(d.join(variant), b"x").unwrap();

            validate_dir("tiny", &d).unwrap_or_else(|e| {
                panic!("{variant} alone should satisfy the tokenizer requirement, got: {e}")
            });
        }
    }

    #[test]
    fn validate_rejects_a_directory_without_model_bin() {
        let d = temp_dir("no_bin");
        std::fs::write(d.join("config.json"), b"{}").unwrap();

        let err = validate_dir("tiny", &d).unwrap_err();

        match err {
            Error::ModelNotFound { name, path, message } => {
                assert_eq!(name, "tiny");
                assert_eq!(path, d);
                assert!(
                    message.contains("model.bin"),
                    "the message must name the missing file, got: {message}"
                );
            }
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn a_valid_local_directory_is_returned_as_is() {
        let d = temp_dir("local");
        std::fs::write(d.join("model.bin"), b"x").unwrap();
        std::fs::write(d.join("config.json"), b"{}").unwrap();
        std::fs::write(d.join("tokenizer.json"), b"{}").unwrap();

        let got = ensure_model(d.to_str().unwrap(), &FetchOptions::default()).unwrap();

        assert_eq!(got, d);
    }

    #[test]
    fn synthesize_writes_the_expected_json_when_absent() {
        let d = temp_dir("synth_default");

        ensure_preprocessor_config(&d, "Systran/faster-whisper-tiny").unwrap();

        let written = std::fs::read_to_string(d.join("preprocessor_config.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(parsed["feature_size"], 80);
        assert_eq!(parsed["sampling_rate"], 16000);
        assert_eq!(parsed["hop_length"], 160);
        assert_eq!(parsed["n_fft"], 400);
        assert_eq!(parsed["chunk_length"], 30);
        assert_eq!(parsed["n_samples"], 480000);
        assert_eq!(parsed["nb_max_frames"], 3000);
        assert!(parsed["mel_filters"].is_null());
    }

    #[test]
    fn synthesize_uses_128_mel_bins_for_large_v3_identities() {
        let d = temp_dir("synth_large_v3");
        ensure_preprocessor_config(&d, "Systran/faster-whisper-large-v3").unwrap();
        let written = std::fs::read_to_string(d.join("preprocessor_config.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(parsed["feature_size"], 128);

        let d2 = temp_dir("synth_distil_large_v3");
        ensure_preprocessor_config(&d2, "distil-whisper/distil-large-v3-ct2").unwrap();
        let written2 = std::fs::read_to_string(d2.join("preprocessor_config.json")).unwrap();
        let parsed2: serde_json::Value = serde_json::from_str(&written2).unwrap();
        assert_eq!(parsed2["feature_size"], 128);
    }

    #[test]
    fn synthesize_never_overwrites_an_existing_file() {
        let d = temp_dir("synth_no_overwrite");
        std::fs::write(d.join("preprocessor_config.json"), b"{\"custom\": true}").unwrap();

        ensure_preprocessor_config(&d, "Systran/faster-whisper-large-v3").unwrap();

        let written = std::fs::read_to_string(d.join("preprocessor_config.json")).unwrap();
        assert_eq!(written, "{\"custom\": true}");
    }

    #[test]
    fn local_files_only_refuses_to_download() {
        let opts = FetchOptions {
            local_files_only: true,
            ..Default::default()
        };

        let err = ensure_model("tiny", &opts).unwrap_err();

        match err {
            Error::ModelNotFound { message, .. } => assert!(
                message.contains("local_files_only"),
                "the message must explain why nothing was downloaded, got: {message}"
            ),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }
}
