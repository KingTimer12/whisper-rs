use crate::asr::ct2::{Ct2Asr, Ct2Config};
use crate::asr::Asr;
use crate::models::hub::{ensure_model, FetchOptions};
use crate::python::iter::SegmentIterator;
use crate::python::segment::TranscriptionInfo;
use crate::python::to_pyerr;
use crate::vad::VadParams;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A loaded Whisper model, ready to transcribe audio files.
///
/// Loads (downloading if needed) a CTranslate2-converted Whisper checkpoint
/// and exposes [`transcribe`][WhisperModel::transcribe]. `model` may be a
/// short alias (`"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v3"`, ...),
/// an explicit `org/repo` on the Hugging Face Hub, or a local directory
/// already in CTranslate2 format (see `whisper_rs.convert.convert_model` to
/// produce one from an arbitrary Hugging Face Whisper checkpoint).
///
/// # Example
/// ```python
/// import whisper_rs
///
/// model = whisper_rs.WhisperModel("tiny")
/// segments, info = model.transcribe("audio.wav")
/// print(info.duration, info.duration_after_vad)
/// for segment in segments:
///     print(segment.start, segment.end, segment.text)
/// ```
/// Default upper bound on the speaker count for `diarize=True`.
///
/// Named because the signature default and the "you set this without
/// diarize=True" check must not be able to drift apart.
const DEFAULT_MAX_SPEAKERS: usize = 8;

#[pyclass]
pub struct WhisperModel {
    asr: Arc<Ct2Asr>,
    config: Ct2Config,
    model_dir: PathBuf,
}

#[pymethods]
impl WhisperModel {
    #[new]
    #[pyo3(signature = (
        model,
        *,
        device = "cpu",
        device_index = 0,
        compute_type = "default",
        cpu_threads = 0,
        num_workers = 1,
        download_root = None,
        local_files_only = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: &str,
        device: &str,
        device_index: i32,
        compute_type: &str,
        cpu_threads: usize,
        num_workers: usize,
        download_root: Option<PathBuf>,
        local_files_only: bool,
    ) -> PyResult<Self> {
        let opts = FetchOptions {
            download_root,
            local_files_only,
        };

        let config = Ct2Config {
            device: device.to_string(),
            device_index,
            compute_type: compute_type.to_string(),
            cpu_threads,
            num_workers,
            ..Ct2Config::default()
        };

        let name = model.to_string();
        let cfg = config.clone();
        // Downloading and loading the model both take seconds to minutes.
        let (asr, model_dir) = py
            .detach(move || -> crate::error::Result<(Ct2Asr, PathBuf)> {
                let dir = ensure_model(&name, &opts)?;
                let asr = Ct2Asr::new(&dir, cfg)?;
                Ok((asr, dir))
            })
            .map_err(to_pyerr)?;

        Ok(Self {
            asr: Arc::new(asr),
            config,
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
    /// # Whole-file, whole-language-detection eagerness
    ///
    /// Loading, VAD, and windowing the *entire* audio file happen eagerly,
    /// synchronously, inside this call (the whole decoded file is held in
    /// memory as `f32` samples -- there is no streaming/chunked-from-disk
    /// path in v1). If `language` is left as `None`, this call *also*
    /// auto-detects the language from the first window up front, before any
    /// segment has been produced. Detection is a single encoder pass (far
    /// cheaper than the full decode this used to cost), but it loads a
    /// second, transient copy of the model weights, because `ct2rs` keeps the
    /// ones this class holds private. Passing `language=` explicitly (e.g.
    /// `"en"`) skips detection, and that load, entirely.
    ///
    /// `diarize=True` joins this same eager section: diarization needs the
    /// whole file's samples before the first speaker can be assigned, so it
    /// runs synchronously inside this call, alongside VAD, windowing, and
    /// language detection. Only ASR decoding stays lazy, deferred until
    /// `segments` is iterated.
    ///
    /// # `word_timestamps`
    ///
    /// Tri-state, resolved against `diarize`:
    ///
    /// | `word_timestamps` | `diarize` | Result |
    /// |---|---|---|
    /// | `None` (default) | `False` | word timestamps off |
    /// | `None` (default) | `True`  | word timestamps on |
    /// | `True`           | either  | word timestamps on |
    /// | `False`          | `False` | word timestamps off |
    /// | `False`          | `True`  | `ValueError` |
    ///
    /// Speakers are assigned per word, so diarization requires word
    /// timestamps; an explicit `False` alongside `diarize=True` is a
    /// contradiction rather than something silently overridden, and raises
    /// `ValueError`. Leaving `word_timestamps` unset lets it follow
    /// `diarize` automatically.
    ///
    /// # `vad_parameters["threshold"]` / `["neg_threshold"]`
    ///
    /// These two keys are only meaningful when the crate is compiled with
    /// the `silero-vad` feature. The default VAD backend (WebRTC) emits only
    /// 0.0/1.0 speech probabilities, so the hysteresis band between
    /// `threshold` and `neg_threshold` degenerates to a no-op on the default
    /// build; a warning is logged (via `tracing`) when either is set in that
    /// configuration. They are still accepted (not rejected) because they
    /// are real, effective parameters under `silero-vad`.
    ///
    /// # `info.language_probability`
    ///
    /// The detector's own probability for the detected language, or `None`
    /// when `language=` was pinned (nothing was detected, so there is no
    /// score to report) or when the audio held no speech at all.
    #[pyo3(signature = (
        audio,
        *,
        language = None,
        task = "transcribe",
        beam_size = 5,
        patience = 1.0,
        length_penalty = 1.0,
        temperature = 0.0,
        repetition_penalty = 1.0,
        no_repeat_ngram_size = 0,
        max_initial_timestamp = 1.0,
        suppress_blank = true,
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
        task: &str,
        beam_size: usize,
        patience: f32,
        length_penalty: f32,
        temperature: f32,
        repetition_penalty: f32,
        no_repeat_ngram_size: usize,
        max_initial_timestamp: f32,
        suppress_blank: bool,
        word_timestamps: Option<bool>,
        vad_filter: bool,
        vad_parameters: Option<Bound<'_, PyDict>>,
        diarize: bool,
        max_speakers: usize,
        num_speakers: Option<usize>,
    ) -> PyResult<(SegmentIterator, TranscriptionInfo)> {
        if task != "transcribe" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "task {task:?} is not supported in v1, only \"transcribe\""
            )));
        }

        // The same rule the `word_timestamps` check below enforces, applied to
        // the other two knobs: a parameter that cannot be honoured must error
        // rather than be quietly discarded. Without this, `max_speakers=999`
        // (impossible under any configuration -- the backend takes a u8) and
        // `num_speakers=99` alongside `diarize=False` are both accepted and
        // thrown away, and the caller never learns their request did nothing.
        //
        // Checked up here, before `prepare`, so the error arrives immediately
        // instead of after a full decode, VAD and language-detection pass.
        if !diarize {
            if max_speakers != DEFAULT_MAX_SPEAKERS {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "max_speakers={max_speakers} has no effect without diarize=True.                      Pass diarize=True, or leave max_speakers unset."
                )));
            }
            if let Some(k) = num_speakers {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "num_speakers={k} has no effect without diarize=True.                      Pass diarize=True, or leave num_speakers unset."
                )));
            }
        }

        // Per-word speaker assignment requires word timestamps. Silently
        // switching on a parameter the caller passed as `False` is the
        // accept-and-ignore behaviour this crate forbids, so an explicit
        // `False` alongside `diarize=True` is a contradiction and errors.
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

        // These four are per-call, so the backend is rebuilt when they differ
        // from the loaded configuration.
        let needs_reload = beam_size != self.config.beam_size
            || (patience - self.config.patience).abs() > f32::EPSILON
            || (length_penalty - self.config.length_penalty).abs() > f32::EPSILON
            || (temperature - self.config.temperature).abs() > f32::EPSILON
            || (repetition_penalty - self.config.repetition_penalty).abs() > f32::EPSILON
            || no_repeat_ngram_size != self.config.no_repeat_ngram_size
            || (max_initial_timestamp - self.config.max_initial_timestamp).abs() > f32::EPSILON
            || suppress_blank != self.config.suppress_blank;

        let asr = if needs_reload {
            let cfg = Ct2Config {
                beam_size,
                patience,
                length_penalty,
                temperature,
                repetition_penalty,
                no_repeat_ngram_size,
                max_initial_timestamp,
                suppress_blank,
                ..self.config.clone()
            };
            let dir = self.model_dir.clone();
            Arc::new(
                py.detach(move || Ct2Asr::new(&dir, cfg))
                    .map_err(to_pyerr)?,
            )
        } else {
            Arc::clone(&self.asr)
        };

        let path: PathBuf = audio;
        let asr_for_prep = Arc::clone(&asr);
        let (windows, info, turns) = py
            .detach(move || -> crate::error::Result<_> {
                let prepared = crate::pipeline::prepare(Path::new(&path), vad_filter, &params)?;
                let mut info = prepared.info;

                match language {
                    Some(code) => info.language = code,
                    None => match prepared.windows.first() {
                        // One encoder pass over the first window; the code is
                        // then reused for every window so the language cannot
                        // flip mid-file.
                        Some(w) => {
                            let (code, probability) =
                                asr_for_prep.detect_language(&w.samples)?;
                            info.language = code;
                            info.language_probability = Some(probability);
                        }
                        // No speech at all: there is nothing to detect from,
                        // and no probability to report.
                        None => info.language = "unknown".to_string(),
                    },
                }

                // Diarization consumes the WHOLE file rather than the VAD
                // windows the ASR decodes: clustering speaker embeddings
                // globally is what lifts the speaker count off any per-window
                // limit, so it cannot be done window by window.
                let turns = if diarize {
                    diarize_all(&prepared.samples, max_speakers, num_speakers)?
                } else {
                    Vec::new()
                };

                // The distinct speakers actually present in the turns, not the
                // `max_speakers` bound the caller asked for: reporting the
                // bound would claim speakers that were never found.
                info.num_speakers = if diarize {
                    let mut ids: Vec<usize> = turns.iter().map(|t| t.speaker).collect();
                    ids.sort_unstable();
                    ids.dedup();
                    Some(ids.len())
                } else {
                    None
                };

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

/// Diarize the whole signal, or explain why this build cannot.
///
/// Split out from `transcribe` so the `#[cfg]` pair lives in one place: an
/// `#[cfg]`-diverging expression inlined into a `let` is easy to get subtly
/// wrong, and the not-enabled arm must fail loudly. `diarize=True` on a build
/// without the feature is a request this binary cannot honour, so it errors
/// rather than silently returning no turns -- which would look exactly like
/// audio containing no speakers.
#[cfg_attr(not(feature = "diarization"), allow(unused_variables))]
fn diarize_all(
    samples: &[f32],
    max_speakers: usize,
    num_speakers: Option<usize>,
) -> crate::error::Result<Vec<crate::diarize::SpeakerTurn>> {
    #[cfg(feature = "diarization")]
    {
        use crate::diarize::Diarizer;
        let diarizer =
            crate::diarize::polyvoice::PolyvoiceDiarizer::new(max_speakers, num_speakers)?;
        diarizer.diarize(samples)
    }
    #[cfg(not(feature = "diarization"))]
    {
        Err(crate::error::Error::Diarize(
            "this build has no diarization support: reinstall with \
             `pip install whisper-rs[diarization]`, or build the crate with \
             --features diarization"
                .to_string(),
        ))
    }
}

fn vad_params_from_dict(dict: Option<&Bound<'_, PyDict>>) -> PyResult<VadParams> {
    let mut params = VadParams::default();
    let Some(dict) = dict else {
        return Ok(params);
    };

    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "threshold" => {
                params.threshold = value.extract()?;
                warn_if_inert_on_default_backend("threshold");
            }
            "neg_threshold" => {
                params.neg_threshold = value.extract()?;
                warn_if_inert_on_default_backend("neg_threshold");
            }
            "min_speech_duration_ms" => params.min_speech_ms = value.extract()?,
            "min_silence_duration_ms" => params.min_silence_ms = value.extract()?,
            "speech_pad_ms" => params.speech_pad_ms = value.extract()?,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown vad_parameters key {other:?}; supported: threshold, \
                     neg_threshold, min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms"
                )))
            }
        }
    }

    Ok(params)
}

/// Warn that `key` is accepted but has no effect on the compiled-in default
/// VAD backend.
///
/// The default backend is WebRTC (see `vad::WebRtcBackend`'s docs), which
/// emits only 0.0/1.0 probabilities -- so the two-threshold hysteresis band
/// `threshold`/`neg_threshold` select between degenerates: 1.0 always opens
/// a region and 0.0 always closes it regardless of where either threshold is
/// set. Both keys are provably inert unless the crate is built with the
/// `silero-vad` feature, which supplies a graded backend where they matter.
/// The spec forbids silently accepting and ignoring an argument, so this
/// warns (it does not error: the key is still meaningful under that
/// feature, and the default build must keep accepting it for API
/// compatibility with callers who build both ways).
#[cfg(not(feature = "silero-vad"))]
fn warn_if_inert_on_default_backend(key: &str) {
    tracing::warn!(
        "vad_parameters[{key:?}] has no effect: the default VAD backend (WebRTC) only emits \
         0.0/1.0 probabilities, so {key} degenerates to a no-op. Build with the \
         `silero-vad` feature for a graded backend where this parameter matters."
    );
}

#[cfg(feature = "silero-vad")]
fn warn_if_inert_on_default_backend(_key: &str) {}

#[cfg(all(test, not(feature = "diarization")))]
mod tests {
    #[test]
    fn diarize_without_the_feature_errors_instead_of_returning_no_turns() {
        // Returning an empty Vec here would be indistinguishable from audio
        // with no detectable speakers, so a build that cannot diarize must say
        // so, and must say how to get one that can.
        let err = match super::diarize_all(&[0.0; 16_000], 8, None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a build without the feature cannot diarize"),
        };
        assert!(err.contains("whisper-rs[diarization]"), "got: {err}");
        assert!(err.contains("--features diarization"), "got: {err}");
    }
}
