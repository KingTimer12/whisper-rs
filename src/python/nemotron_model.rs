//! `NemotronModel`: the Nemotron backend exposed to Python.
//!
//! Structurally parallel to `WhisperModel` (`model.rs`) — same eager
//! prepare/VAD/language-detect/diarize split — but with none of Whisper's
//! beam-search decoding knobs, which do not apply to Nemotron's fixed
//! greedy decoder.

use crate::asr::nemotron::{NemotronAsr, NemotronConfig};
use crate::asr::Asr;
use crate::models::hub::{ensure_nemotron_model, FetchOptions};
use crate::python::iter::SegmentIterator;
use crate::python::model::{diarize_all, distinct_speakers, vad_params_from_dict, DEFAULT_MAX_SPEAKERS};
use crate::python::segment::TranscriptionInfo;
use crate::python::to_pyerr;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A loaded Nemotron model, ready to transcribe audio files.
///
/// # Example
/// ```python
/// import whisper_rs
///
/// model = whisper_rs.NemotronModel("nemotron")
/// segments, info = model.transcribe("audio.wav")
/// for segment in segments:
///     print(segment.start, segment.end, segment.text)
/// ```
#[pyclass]
pub struct NemotronModel {
    asr: Arc<NemotronAsr>,
    model_dir: PathBuf,
    /// The language configured at construction time, if any. Kept (rather
    /// than being consumed once by `NemotronAsr::new` and discarded) so that
    /// `transcribe` calls that don't pass `language=` explicitly still honour
    /// it, instead of silently falling through to auto-detection -- see
    /// `resolve_language`.
    target_lang: Option<String>,
}

#[pymethods]
impl NemotronModel {
    #[new]
    #[pyo3(signature = (
        model,
        *,
        download_root = None,
        local_files_only = false,
        target_lang = None,
    ))]
    fn new(
        py: Python<'_>,
        model: &str,
        download_root: Option<PathBuf>,
        local_files_only: bool,
        target_lang: Option<String>,
    ) -> PyResult<Self> {
        let opts = FetchOptions {
            download_root,
            local_files_only,
        };
        let config = NemotronConfig {
            target_lang: target_lang.clone(),
        };

        let name = model.to_string();
        let (asr, model_dir) = py
            .detach(move || -> crate::error::Result<(NemotronAsr, PathBuf)> {
                let dir = ensure_nemotron_model(&name, &opts)?;
                let asr = NemotronAsr::new(&dir, config)?;
                Ok((asr, dir))
            })
            .map_err(to_pyerr)?;

        Ok(Self {
            asr: Arc::new(asr),
            model_dir,
            target_lang,
        })
    }

    /// Directory the model was loaded from. Useful when debugging cache issues.
    #[getter]
    fn model_path(&self) -> String {
        self.model_dir.display().to_string()
    }

    /// Transcribe an audio file.
    ///
    /// Returns `(segments, info)`: `segments` is a lazily-decoded iterator
    /// (nothing is decoded until it is iterated) and `info` is a
    /// [`TranscriptionInfo`] that is already fully populated by the time
    /// this call returns.
    ///
    /// See `WhisperModel::transcribe`'s docs for the shared eager
    /// prepare/VAD/language-detect/diarize behaviour and the
    /// `word_timestamps`/`diarize` tri-state rules -- they apply identically
    /// here. Nemotron has no beam-search decoding knobs (`beam_size`,
    /// `temperature`, `patience`, `length_penalty`, `repetition_penalty`):
    /// its decoder is fixed and greedy, so this method takes none of them.
    ///
    /// # Language resolution
    ///
    /// Precedence: **explicit per-call `language` > constructor
    /// `target_lang` > auto-detect.**
    ///
    /// | `language` (per-call) | `target_lang` (constructor) | Result |
    /// |---|---|---|
    /// | `Some(code)` | either | `code`; no detection pass runs |
    /// | `None` | `Some(code)` | `code`; no detection pass runs |
    /// | `None` | `None` | auto-detected from the first window |
    ///
    /// A `target_lang` configured on the model used to be silently discarded
    /// by every `transcribe()` call that omitted `language=`: the resolved
    /// value (whatever auto-detect guessed) always clobbered it, because the
    /// only place `target_lang` was consulted was once, inside
    /// `NemotronAsr::new`, before any audio existed. Skipping the detection
    /// pass entirely when `target_lang` applies also avoids the wasted
    /// encoder pass detection costs.
    ///
    /// `info.language` reports whatever was actually used (or `"unknown"` if
    /// detection found no language tag at all); `info.language_probability`
    /// is only populated when a detection pass actually ran, since there is
    /// no probability to report for a language pinned by the caller or by
    /// `target_lang`.
    ///
    /// Every `word.probability` in the returned segments is always `1.0`:
    /// the underlying timestamped-token API carries no per-token
    /// confidence, and computing a real value would require an expensive
    /// second full decode pass, so it is not attempted.
    #[pyo3(signature = (
        audio,
        *,
        language = None,
        word_timestamps = None,
        vad_filter = true,
        vad_parameters = None,
        diarize = false,
        max_speakers = DEFAULT_MAX_SPEAKERS,
        num_speakers = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn transcribe(
        &self,
        py: Python<'_>,
        audio: PathBuf,
        language: Option<String>,
        word_timestamps: Option<bool>,
        vad_filter: bool,
        vad_parameters: Option<Bound<'_, PyDict>>,
        diarize: bool,
        max_speakers: usize,
        num_speakers: Option<usize>,
    ) -> PyResult<(SegmentIterator, TranscriptionInfo)> {
        // See `WhisperModel::transcribe` for why this must error rather than
        // silently accept-and-ignore.
        if !diarize {
            if max_speakers != DEFAULT_MAX_SPEAKERS {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "max_speakers={max_speakers} has no effect without diarize=True. \
                     Pass diarize=True, or leave max_speakers unset."
                )));
            }
            if let Some(k) = num_speakers {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "num_speakers={k} has no effect without diarize=True. \
                     Pass diarize=True, or leave num_speakers unset."
                )));
            }
        }

        let word_timestamps = match (word_timestamps, diarize) {
            (Some(false), true) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "word_timestamps=False cannot be combined with diarize=True: \
                     speakers are assigned per word, so word timestamps are required. \
                     Pass word_timestamps=True, or leave it unset to have it enabled \
                     automatically.",
                ))
            }
            (Some(explicit), _) => explicit,
            (None, diarize) => diarize,
        };

        let params = vad_params_from_dict(vad_parameters.as_ref())?;

        // Explicit per-call `language` beats the constructor's `target_lang`,
        // which beats auto-detection. `Some` here means "skip detection
        // entirely, this call already knows the language".
        let resolved_language = resolve_language(language.as_deref(), self.target_lang.as_deref());

        let asr = Arc::clone(&self.asr);
        let asr_for_prep = Arc::clone(&asr);
        let path: PathBuf = audio;
        let (windows, info, turns) = py
            .detach(move || -> crate::error::Result<_> {
                let prepared = crate::pipeline::prepare(Path::new(&path), vad_filter, &params)?;
                let mut info = prepared.info;

                match resolved_language {
                    Some(code) => info.language = code,
                    None => match prepared.windows.first() {
                        Some(w) => {
                            let (code, probability) = asr_for_prep.detect_language(&w.samples)?;
                            info.language = code;
                            info.language_probability = Some(probability);
                        }
                        None => info.language = "unknown".to_string(),
                    },
                }

                let turns = if diarize {
                    diarize_all(&prepared.samples, max_speakers, num_speakers)?
                } else {
                    Vec::new()
                };

                info.num_speakers = distinct_speakers(diarize, &turns);

                Ok((prepared.windows, info, turns))
            })
            .map_err(to_pyerr)?;

        // `info.language` is what's reported to the caller and may
        // legitimately be `"unknown"` (detection found nothing). Never hand
        // that literal to the decoder though: `NemotronAsr::transcribe`
        // forwards this string straight into `set_target_lang`, which
        // rejects anything outside its known tag set, so an "unknown" here
        // would fail every window with a confusing `RuntimeError` from deep
        // in the decode loop. `"auto"` is `parakeet-rs`'s documented
        // language-agnostic-decoding sentinel -- the correct way to say
        // "don't force a language" -- so normalize to it right at this
        // boundary, once, rather than downstream in `NemotronAsr` or
        // `SegmentIterator` where the reason would be far from the code.
        let decode_language = decode_language(&info.language).to_string();
        Ok((
            SegmentIterator::new(asr, windows, decode_language, word_timestamps, turns),
            info.into(),
        ))
    }
}

/// Which language code a `transcribe()` call should use, and whether an
/// auto-detection pass is needed at all.
///
/// Precedence: explicit per-call `language` beats the constructor's
/// `target_lang`, which beats auto-detection. Returns `Some(code)` when the
/// caller (via either source) already pinned the language -- detection must
/// be skipped in that case -- or `None` when nothing was pinned and
/// detection has to run.
///
/// Pure and free of the pyclass so it is unit-testable without model
/// weights; the surrounding `transcribe` glue (detection, decoding) is not.
fn resolve_language(explicit: Option<&str>, configured: Option<&str>) -> Option<String> {
    explicit.or(configured).map(str::to_string)
}

/// Normalize a language code for the decoder: `NemotronAsr::transcribe`
/// passes this straight into `parakeet-rs`'s `set_target_lang`, which
/// recognizes real tags and the `"auto"` sentinel but rejects anything else
/// -- including the literal `"unknown"` this crate's own `detect_language`
/// can return. Every other code (a real tag, or already `"auto"`) passes
/// through unchanged.
fn decode_language(code: &str) -> &str {
    if code == "unknown" {
        "auto"
    } else {
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_language_wins_over_configured_target_lang() {
        assert_eq!(
            resolve_language(Some("en"), Some("fr")),
            Some("en".to_string())
        );
    }

    #[test]
    fn explicit_language_wins_when_nothing_is_configured() {
        assert_eq!(resolve_language(Some("en"), None), Some("en".to_string()));
    }

    #[test]
    fn configured_target_lang_is_used_when_no_explicit_language_is_given() {
        assert_eq!(resolve_language(None, Some("fr")), Some("fr".to_string()));
    }

    #[test]
    fn neither_given_means_detection_must_run() {
        assert_eq!(resolve_language(None, None), None);
    }

    #[test]
    fn unknown_is_normalized_to_auto_for_the_decoder() {
        assert_eq!(decode_language("unknown"), "auto");
    }

    #[test]
    fn a_real_language_code_passes_through_unchanged() {
        assert_eq!(decode_language("en"), "en");
        assert_eq!(decode_language("fr-FR"), "fr-FR");
    }

    #[test]
    fn already_auto_passes_through_unchanged() {
        assert_eq!(decode_language("auto"), "auto");
    }
}
