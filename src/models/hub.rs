//! Fetch CTranslate2 Whisper models from Hugging Face, or use a local directory.

use crate::error::{Error, Result};
use crate::models::registry::{resolve, ModelRef};
use hf_hub::{split_id, HFClient, HFClientSync, HFError};
use std::path::{Path, PathBuf};

/// Files CTranslate2 needs. `model.bin` and `config.json` are mandatory; the
/// rest are fetched when present, since repos differ in tokenizer layout.
const REQUIRED: &[&str] = &["model.bin", "config.json"];
const OPTIONAL: &[&str] = &[
    "tokenizer.json",
    "vocabulary.json",
    "vocabulary.txt",
    "preprocessor_config.json",
];

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
            Ok(dir)
        }
    }
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

        validate_dir("tiny", &d).unwrap();
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

        let got = ensure_model(d.to_str().unwrap(), &FetchOptions::default()).unwrap();

        assert_eq!(got, d);
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
