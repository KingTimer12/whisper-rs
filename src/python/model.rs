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
        word_timestamps = false,
        vad_filter = true,
        vad_parameters = None,
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
        word_timestamps: bool,
        vad_filter: bool,
        vad_parameters: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(SegmentIterator, TranscriptionInfo)> {
        if task != "transcribe" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "task {task:?} is not supported in v1, only \"transcribe\""
            )));
        }

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
        let (windows, mut info) = py
            .detach(move || -> crate::error::Result<_> {
                let prepared = crate::pipeline::prepare(Path::new(&path), vad_filter, &params)?;
                let mut info = prepared.info;

                info.language = match language {
                    Some(code) => code,
                    None => match prepared.windows.first() {
                        // One extra 30 s decode, then the code is reused for
                        // every window so the language cannot flip mid-file.
                        Some(w) => asr_for_prep.detect_language(&w.samples)?,
                        None => "unknown".to_string(),
                    },
                };

                Ok((prepared.windows, info))
            })
            .map_err(to_pyerr)?;

        // Never fabricated: the API cannot produce it.
        info.language_probability = None;

        let language = info.language.clone();
        Ok((
            SegmentIterator::new(asr, windows, language, word_timestamps),
            info.into(),
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
            "threshold" => params.threshold = value.extract()?,
            "neg_threshold" => params.neg_threshold = value.extract()?,
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
