mod asr;
mod audio;
mod chunk;
#[doc(hidden)]
pub mod diarize;
mod error;
mod models;
mod pipeline;
mod python;
mod stitch;
mod types;
mod vad;

use pyo3::prelude::*;

/// Route the crate's `tracing` events to stderr the first time the extension
/// module is imported.
///
/// Without this every `tracing::warn!` the crate emits -- inert VAD
/// thresholds, a synthesized `preprocessor_config.json` -- is dropped on the
/// floor, because a plain Python process installs no subscriber of its own.
/// `try_init` is deliberate: if the embedding application already set a global
/// subscriber (a host app, or `tracing-subscriber` wired up by another
/// extension), it wins and this is a no-op.
///
/// `WHISPER_RS_LOG` takes the usual `EnvFilter` syntax
/// (`WHISPER_RS_LOG=whisper_rs=debug`); the default shows warnings and errors.
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_env("WHISPER_RS_LOG")
        .unwrap_or_else(|_| EnvFilter::new("whisper_rs=warn"));

    let _ = fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .without_time()
        .try_init();
}

#[pymodule]
fn whisper_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_tracing();
    m.add_class::<python::model::WhisperModel>()?;
    m.add_class::<python::iter::SegmentIterator>()?;
    m.add_class::<python::segment::Segment>()?;
    m.add_class::<python::segment::Word>()?;
    m.add_class::<python::segment::TranscriptionInfo>()?;
    Ok(())
}
