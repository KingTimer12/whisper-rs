//! `NemotronModel`: the Nemotron backend exposed to Python.
//!
//! Structurally parallel to `WhisperModel` (`model.rs`) — same eager
//! prepare/VAD/language-detect/diarize split — but with none of Whisper's
//! beam-search decoding knobs, which do not apply to Nemotron's fixed
//! greedy decoder.

use crate::asr::nemotron::{NemotronAsr, NemotronConfig};
use crate::asr::Asr;
use crate::models::hub::{ensure_model, FetchOptions};
use crate::python::iter::SegmentIterator;
use crate::python::model::{diarize_all, distinct_speakers, vad_params_from_dict};
use crate::python::segment::TranscriptionInfo;
use crate::python::to_pyerr;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Default upper bound on the speaker count for `diarize=True`.
///
/// Named because the signature default and the "you set this without
/// diarize=True" check must not be able to drift apart.
const DEFAULT_MAX_SPEAKERS: usize = 8;

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
        let config = NemotronConfig { target_lang };

        let name = model.to_string();
        let (asr, model_dir) = py
            .detach(move || -> crate::error::Result<(NemotronAsr, PathBuf)> {
                let dir = ensure_model(&name, &opts)?;
                let asr = NemotronAsr::new(&dir, config)?;
                Ok((asr, dir))
            })
            .map_err(to_pyerr)?;

        Ok(Self {
            asr: Arc::new(asr),
            model_dir,
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

        let asr = Arc::clone(&self.asr);
        let asr_for_prep = Arc::clone(&asr);
        let path: PathBuf = audio;
        let (windows, info, turns) = py
            .detach(move || -> crate::error::Result<_> {
                let prepared = crate::pipeline::prepare(Path::new(&path), vad_filter, &params)?;
                let mut info = prepared.info;

                match language {
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

        let language = info.language.clone();
        Ok((
            SegmentIterator::new(asr, windows, language, word_timestamps, turns),
            info.into(),
        ))
    }
}
