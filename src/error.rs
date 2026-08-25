//! Error type for the whole crate. Mapped to Python exceptions in python/mod.rs.
//!
//! Nothing constructs these variants yet: later tasks (audio, VAD, ct2rs
//! wrapper) return them. Allow dead_code until then so clippy stays clean.
#![allow(dead_code)]

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot read audio file {}: {message}", path.display())]
    AudioRead { path: PathBuf, message: String },

    #[error("unsupported audio format in {}: {message}", path.display())]
    AudioFormat { path: PathBuf, message: String },

    #[error("audio file {} contains no audio track", path.display())]
    AudioEmpty { path: PathBuf },

    #[error("resampling failed: {0}")]
    Resample(String),

    #[error("model {name} not found: {message} (looked in {})", path.display())]
    ModelNotFound {
        name: String,
        path: PathBuf,
        message: String,
    },

    #[error("failed to download model {name}: {message}")]
    Download { name: String, message: String },

    #[error("voice activity detection failed: {0}")]
    Vad(String),

    #[error("CTranslate2 failed while transcribing: {0}")]
    Ct2(String),

    #[error("diarization failed: {0}")]
    Diarize(String),

    #[error("{message}")]
    OnnxRuntimeMissing { message: String },
}
