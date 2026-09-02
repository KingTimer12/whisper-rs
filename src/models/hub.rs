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
        // Every Whisper alias resolves with `subfolder: None`; this path
        // never needed a subfolder, so it is simply ignored here.
        ModelRef::Hub { repo, subfolder: _ } => {
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

/// Build a synchronous HF Hub client honouring `opts.download_root`. Shared
/// by every fetcher (CT2 Whisper, and -- behind the `nemotron` feature --
/// Nemotron) since the client construction itself has nothing
/// backend-specific about it.
fn build_client(name: &str, opts: &FetchOptions) -> Result<HFClientSync> {
    let mut builder = HFClient::builder();
    if let Some(root) = &opts.download_root {
        builder = builder.cache_dir(root.clone());
    }
    let client = builder.build().map_err(|e| Error::Download {
        name: name.to_string(),
        message: e.to_string(),
    })?;
    HFClientSync::from_inner(client).map_err(|e| Error::Download {
        name: name.to_string(),
        message: e.to_string(),
    })
}

/// Download `required` (hard failure if any is missing) and `optional`
/// (best-effort, missing is normal) files from `repo` into the HF cache, and
/// return the directory they landed in. Shared plumbing for both the CT2
/// Whisper fetcher and, behind the `nemotron` feature, the Nemotron fetcher
/// -- the two backends need different files but the same download loop.
fn download_files(
    name: &str,
    repo: &str,
    opts: &FetchOptions,
    required: &[&str],
    optional: &[&str],
) -> Result<PathBuf> {
    let client = build_client(name, opts)?;

    let (owner, repo_name) = split_id(repo);
    let api_repo = client.model(owner, repo_name);

    let mut dir: Option<PathBuf> = None;

    for file in required {
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

    for file in optional {
        // Absent optional files are normal; repos differ in layout.
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

fn download(name: &str, repo: &str, opts: &FetchOptions) -> Result<PathBuf> {
    download_files(name, repo, opts, REQUIRED, OPTIONAL)
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

/// Files `parakeet-rs` requires to load a Nemotron model: a SentencePiece
/// tokenizer (`nemotron.rs::from_pretrained` -> `SentencePieceVocab::from_file`)
/// and the two ONNX graphs (`model_nemotron.rs::NemotronModel::from_pretrained`,
/// which explicitly checks both paths and returns `Error::Config` naming
/// whichever is absent -- an opaque error from deep inside `parakeet-rs`/`ort`
/// unless we catch it first, exactly like `validate_dir` does for CTranslate2).
/// This is an entirely different, ONNX-shaped layout from CT2 Whisper's
/// `REQUIRED`/`TOKENIZER_FILES`, so it is a separate list rather than a
/// variant of those.
#[cfg(feature = "nemotron")]
const NEMOTRON_REQUIRED: &[&str] = &["tokenizer.model", "encoder.onnx", "decoder_joint.onnx"];

/// `encoder.onnx` can reference its weights out-of-line as
/// `encoder.onnx.data` (the standard ONNX "external data" convention, used
/// when a graph's weights exceed the 2 GiB protobuf limit). `parakeet-rs`'s
/// own README (`Setup` section) lists it as required for the full-precision
/// exports it links (both the English-only and Multilingual 3.5 Nemotron
/// entries, including the `nemotron` alias's own
/// `altunenes/parakeet-rs/nemotron-3.5-asr-streaming-0.6b-onnx`), and the
/// `nemotron` alias fetches it accordingly. It is still not *hard-required*
/// in `validate_nemotron_dir`, though: nothing in `parakeet-rs`'s Rust code
/// opens this path by name (`ort`'s session loader resolves it internally,
/// transparently, only if the `.onnx` protobuf actually references external
/// data), and the README's own quantized mirrors (int8/int4) embed their
/// weights and ship no `.data` file at all -- a user pointing
/// `NemotronModel` at one of those locally must not be rejected for lacking
/// a file that checkpoint never needed. So: downloaded whenever the repo has
/// it (alongside `REQUIRED`/`OPTIONAL`'s existing "absent is normal"
/// handling for the CT2 path), never checked in `validate_nemotron_dir`. A
/// full-precision export missing it is `ort`'s own load error to raise, not
/// ours to pre-empt.
#[cfg(feature = "nemotron")]
const NEMOTRON_OPTIONAL: &[&str] = &["encoder.onnx.data"];

/// Resolve `name` to a local directory containing a usable Nemotron (ONNX)
/// model, downloading from the hub when needed.
///
/// Deliberately not a variant of [`ensure_model`]: that function's
/// `REQUIRED`/`TOKENIZER_FILES` lists and its synthesized
/// `preprocessor_config.json` (see [`ensure_preprocessor_config`]) are
/// CTranslate2-specific concerns that do not apply to Nemotron's ONNX
/// layout, and `WhisperModel` has shipped on `ensure_model`'s exact
/// behaviour and error messages for two versions -- parameterizing it would
/// risk changing those for a backend that isn't the one being added. This
/// function reuses only the plumbing that is genuinely shared: HF client
/// construction and the download loop ([`download_files`]), `split_id`, and
/// the `Error::Download`/`Error::ModelNotFound` shapes.
#[cfg(feature = "nemotron")]
pub fn ensure_nemotron_model(name: &str, opts: &FetchOptions) -> Result<PathBuf> {
    match resolve(name) {
        ModelRef::Local(dir) => {
            validate_nemotron_dir(name, &dir)?;
            Ok(dir)
        }
        ModelRef::Hub { repo, subfolder } => {
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
            // Some Hub repos (the `nemotron` alias's `altunenes/parakeet-rs`
            // bucket) host several unrelated models side by side, one
            // subdirectory each. `hf-hub`'s `download_file().filename(...)`
            // takes any repo-relative path, so prefixing each filename with
            // the subfolder downloads straight into it; `download_files`
            // then derives the returned directory from the downloaded
            // file's parent, which naturally becomes that subfolder --
            // exactly the flat directory `NemotronModel::from_pretrained`
            // expects, with no extra "which subdir did we land in" logic.
            let prefixed = |file: &str| match &subfolder {
                Some(sub) => format!("{sub}/{file}"),
                None => file.to_string(),
            };
            let required: Vec<String> = NEMOTRON_REQUIRED.iter().map(|f| prefixed(f)).collect();
            let optional: Vec<String> = NEMOTRON_OPTIONAL.iter().map(|f| prefixed(f)).collect();
            let required_refs: Vec<&str> = required.iter().map(String::as_str).collect();
            let optional_refs: Vec<&str> = optional.iter().map(String::as_str).collect();

            let dir = download_files(name, &repo, opts, &required_refs, &optional_refs)?;
            validate_nemotron_dir(name, &dir)?;
            Ok(dir)
        }
    }
}

/// Fail early, and clearly, rather than letting `parakeet-rs`/`ort` abort
/// with an opaque native error deep in ONNX session construction.
#[cfg(feature = "nemotron")]
pub fn validate_nemotron_dir(name: &str, dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        return Err(Error::ModelNotFound {
            name: name.to_string(),
            path: dir.to_path_buf(),
            message: "not a directory".into(),
        });
    }

    let missing: Vec<&str> = NEMOTRON_REQUIRED
        .iter()
        .copied()
        .filter(|f| !dir.join(f).is_file())
        .collect();

    if !missing.is_empty() {
        return Err(Error::ModelNotFound {
            name: name.to_string(),
            path: dir.to_path_buf(),
            message: format!(
                "missing required file(s): {} -- a Nemotron model directory needs a \
                 SentencePiece tokenizer (tokenizer.model) and both ONNX graphs \
                 (encoder.onnx, decoder_joint.onnx); without them parakeet-rs fails \
                 to load the model",
                missing.join(", ")
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

    #[cfg(feature = "nemotron")]
    mod nemotron {
        use super::*;

        fn write_all_nemotron_files(d: &Path) {
            std::fs::write(d.join("tokenizer.model"), b"x").unwrap();
            std::fs::write(d.join("encoder.onnx"), b"x").unwrap();
            std::fs::write(d.join("decoder_joint.onnx"), b"x").unwrap();
        }

        #[test]
        fn validate_accepts_a_directory_with_all_three_files() {
            let d = temp_dir("nemotron_ok");
            write_all_nemotron_files(&d);

            validate_nemotron_dir("nemotron", &d).unwrap();
        }

        #[test]
        fn validate_accepts_the_three_required_files_without_encoder_onnx_data() {
            // A quantized (int8/int4) export embeds its weights and ships no
            // encoder.onnx.data at all -- validate_nemotron_dir must not
            // reject it for that, since it is optional, not required.
            let d = temp_dir("nemotron_no_data_file");
            write_all_nemotron_files(&d);
            assert!(!d.join("encoder.onnx.data").exists());

            validate_nemotron_dir("nemotron", &d).unwrap();
        }

        #[test]
        fn validate_accepts_a_directory_that_also_has_encoder_onnx_data() {
            let d = temp_dir("nemotron_with_data_file");
            write_all_nemotron_files(&d);
            std::fs::write(d.join("encoder.onnx.data"), b"x").unwrap();

            validate_nemotron_dir("nemotron", &d).unwrap();
        }

        #[test]
        fn validate_rejects_a_directory_missing_tokenizer_model() {
            let d = temp_dir("nemotron_no_tokenizer");
            std::fs::write(d.join("encoder.onnx"), b"x").unwrap();
            std::fs::write(d.join("decoder_joint.onnx"), b"x").unwrap();

            let err = validate_nemotron_dir("nemotron", &d).unwrap_err();

            match err {
                Error::ModelNotFound { message, .. } => assert!(
                    message.contains("tokenizer.model"),
                    "the message must name the missing file, got: {message}"
                ),
                other => panic!("expected ModelNotFound, got {other:?}"),
            }
        }

        #[test]
        fn validate_rejects_a_directory_missing_encoder_onnx() {
            let d = temp_dir("nemotron_no_encoder");
            std::fs::write(d.join("tokenizer.model"), b"x").unwrap();
            std::fs::write(d.join("decoder_joint.onnx"), b"x").unwrap();

            let err = validate_nemotron_dir("nemotron", &d).unwrap_err();

            match err {
                Error::ModelNotFound { message, .. } => assert!(
                    message.starts_with("missing required file(s): encoder.onnx "),
                    "the message must name the missing file first, got: {message}"
                ),
                other => panic!("expected ModelNotFound, got {other:?}"),
            }
        }

        #[test]
        fn validate_rejects_a_directory_missing_decoder_joint_onnx() {
            let d = temp_dir("nemotron_no_decoder");
            std::fs::write(d.join("tokenizer.model"), b"x").unwrap();
            std::fs::write(d.join("encoder.onnx"), b"x").unwrap();

            let err = validate_nemotron_dir("nemotron", &d).unwrap_err();

            match err {
                Error::ModelNotFound { name, path, message } => {
                    assert_eq!(name, "nemotron");
                    assert_eq!(path, d);
                    assert!(
                        message.contains("decoder_joint.onnx"),
                        "the message must name the missing file, got: {message}"
                    );
                }
                other => panic!("expected ModelNotFound, got {other:?}"),
            }
        }

        #[test]
        fn validate_rejects_something_that_is_not_a_directory() {
            let d = temp_dir("nemotron_not_a_dir");
            let file = d.join("some_file.txt");
            std::fs::write(&file, b"x").unwrap();

            let err = validate_nemotron_dir("nemotron", &file).unwrap_err();

            match err {
                Error::ModelNotFound { message, .. } => assert!(
                    message.contains("not a directory"),
                    "got: {message}"
                ),
                other => panic!("expected ModelNotFound, got {other:?}"),
            }
        }

        #[test]
        fn a_valid_local_nemotron_directory_is_returned_as_is_and_no_preprocessor_config_written() {
            let d = temp_dir("nemotron_local");
            write_all_nemotron_files(&d);

            let got = ensure_nemotron_model(d.to_str().unwrap(), &FetchOptions::default()).unwrap();

            assert_eq!(got, d);
            // Nemotron is ONNX; the CT2-only synthesized preprocessor_config.json
            // must never be written for it.
            assert!(!d.join("preprocessor_config.json").exists());
        }

        #[test]
        fn nemotron_local_files_only_refuses_to_download() {
            let opts = FetchOptions {
                local_files_only: true,
                ..Default::default()
            };

            // "nemotron" resolves to a hub repo, so this must be rejected
            // before any network access happens.
            let err = ensure_nemotron_model("nemotron", &opts).unwrap_err();

            match err {
                Error::ModelNotFound { message, .. } => assert!(
                    message.contains("local_files_only"),
                    "the message must explain why nothing was downloaded, got: {message}"
                ),
                other => panic!("expected ModelNotFound, got {other:?}"),
            }
        }
    }
}
