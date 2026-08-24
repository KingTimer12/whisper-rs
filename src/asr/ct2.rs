//! ct2rs (CTranslate2) Whisper backend.

use super::Asr;
use crate::error::{Error, Result};
use crate::types::{Seg, Word};
use std::path::Path;

/// Whisper mel frames per second used by ct2rs's frame-indexed options
/// (30 s of audio maps to 1500 frames, i.e. 50 frames/s). ct2rs 0.10 has no
/// seconds-based `max_initial_timestamp` option, only a frame index, so this
/// is used to convert `Ct2Config::max_initial_timestamp` (seconds) into the
/// `max_initial_timestamp_index` ct2rs actually takes.
const FRAMES_PER_SECOND: f32 = 50.0;

#[derive(Debug, Clone)]
pub struct Ct2Config {
    pub device: String,
    pub device_index: i32,
    pub compute_type: String,
    pub cpu_threads: usize,
    pub num_workers: usize,
    pub beam_size: usize,
    pub patience: f32,
    pub length_penalty: f32,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub max_initial_timestamp: f32,
    pub suppress_blank: bool,
    pub temperature: f32,
}

impl Default for Ct2Config {
    fn default() -> Self {
        Self {
            device: "cpu".into(),
            device_index: 0,
            compute_type: "default".into(),
            cpu_threads: 0,
            num_workers: 1,
            beam_size: 5,
            patience: 1.0,
            length_penalty: 1.0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            max_initial_timestamp: 1.0,
            suppress_blank: true,
            temperature: 0.0,
        }
    }
}

pub struct Ct2Asr {
    inner: ct2rs::Whisper,
    options: ct2rs::WhisperOptions,
}

impl Ct2Asr {
    pub fn new(model_dir: &Path, cfg: Ct2Config) -> Result<Self> {
        // NOTE: ct2rs 0.10's `Config` has no per-replica worker-count knob
        // independent of `device_indices` (see `num_replicas()`, which is
        // derived from the device/tensor-parallel setup, not settable
        // directly). `cfg.num_workers` therefore has no equivalent here and
        // is intentionally not read below.
        let config = ct2rs::Config {
            device: parse_device(&cfg.device)?,
            compute_type: parse_compute_type(&cfg.compute_type)?,
            device_indices: vec![cfg.device_index],
            num_threads_per_replica: cfg.cpu_threads,
            ..Default::default()
        };

        let inner =
            ct2rs::Whisper::new(model_dir, config).map_err(|e| Error::Ct2(e.to_string()))?;

        let options = ct2rs::WhisperOptions {
            beam_size: cfg.beam_size,
            patience: cfg.patience,
            length_penalty: cfg.length_penalty,
            repetition_penalty: cfg.repetition_penalty,
            no_repeat_ngram_size: cfg.no_repeat_ngram_size,
            max_initial_timestamp_index: (cfg.max_initial_timestamp * FRAMES_PER_SECOND).round()
                as usize,
            suppress_blank: cfg.suppress_blank,
            sampling_temperature: cfg.temperature,
            ..Default::default()
        };

        Ok(Self { inner, options })
    }

    /// Window size the model expects, in samples.
    pub fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }
}

fn parse_device(name: &str) -> Result<ct2rs::Device> {
    match name {
        "cpu" => Ok(ct2rs::Device::CPU),
        "cuda" => Ok(ct2rs::Device::CUDA),
        other => Err(Error::Ct2(format!(
            "unknown device {other:?}, expected \"cpu\" or \"cuda\""
        ))),
    }
}

fn parse_compute_type(name: &str) -> Result<ct2rs::ComputeType> {
    match name {
        "default" => Ok(ct2rs::ComputeType::DEFAULT),
        "auto" => Ok(ct2rs::ComputeType::AUTO),
        "float32" => Ok(ct2rs::ComputeType::FLOAT32),
        "float16" => Ok(ct2rs::ComputeType::FLOAT16),
        "bfloat16" => Ok(ct2rs::ComputeType::BFLOAT16),
        "int8" => Ok(ct2rs::ComputeType::INT8),
        "int8_float16" => Ok(ct2rs::ComputeType::INT8_FLOAT16),
        "int8_float32" => Ok(ct2rs::ComputeType::INT8_FLOAT32),
        "int8_bfloat16" => Ok(ct2rs::ComputeType::INT8_BFLOAT16),
        other => Err(Error::Ct2(format!("unknown compute_type {other:?}"))),
    }
}

impl Asr for Ct2Asr {
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        word_timestamps: bool,
    ) -> Result<Vec<Seg>> {
        let segments = self
            .inner
            .generate_segments(samples, language, &self.options)
            .map_err(|e| Error::Ct2(e.to_string()))?;

        Ok(segments
            .into_iter()
            .map(|s| Seg {
                // The real id is assigned by stitch().
                id: 0,
                start: s.start,
                end: s.end,
                text: s.text,
                words: if word_timestamps {
                    s.words.map(|words| {
                        words
                            .into_iter()
                            .map(|w| Word {
                                start: w.start,
                                end: w.end,
                                text: w.word,
                                probability: w.probability,
                            })
                            .collect()
                    })
                } else {
                    None
                },
            })
            .collect())
    }

    fn detect_language(&self, samples: &[f32]) -> Result<String> {
        // No detection API exists, so the code is read off the raw output tokens.
        let raw = self
            .inner
            .generate(samples, None, true, &self.options)
            .map_err(|e| Error::Ct2(e.to_string()))?;

        Ok(raw
            .iter()
            .find_map(|line| super::parse_language_token(line))
            .unwrap_or_else(|| "unknown".to_string()))
    }
}
