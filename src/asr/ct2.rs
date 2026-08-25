//! ct2rs (CTranslate2) Whisper backend.

use super::Asr;
use crate::error::{Error, Result};
use crate::types::{Seg, Word};
use std::path::Path;

/// Whisper mel frames per second used by ct2rs's frame-indexed options.
///
/// Whisper emits timestamp tokens at 0.02 s granularity, i.e. 50 per second
/// (30 s of audio maps to 1500 mel frames total). ct2rs 0.10 has no
/// seconds-based `max_initial_timestamp` option, only a frame index
/// (`max_initial_timestamp_index`, default 50), so this constant converts
/// `Ct2Config::max_initial_timestamp` (seconds) into that index. The default
/// case round-trips exactly (`1.0 s * 50.0 == 50`, matching ct2rs's own
/// default), which corroborates the ratio, but the mapping for other values
/// is inferred from CTranslate2's mel-frame timing rather than a documented
/// ct2rs guarantee.
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
    /// Kept so `detect_language` can load a transient detector from the same
    /// weights; `ct2rs::Whisper` does not expose the ones it holds.
    model_dir: std::path::PathBuf,
    config: Ct2Config,
}

/// Build the ct2rs `Config` from a `Ct2Config`.
///
/// ct2rs/CTranslate2 has no independent "worker count" knob: replicas are
/// determined by how many entries `device_indices` has (CTranslate2's
/// `ReplicaPool` loads one model replica per entry, see
/// `models/model.cc`'s `device_indices.size() * num_replicas_per_device`
/// reservation). Repeating the same device index `num_workers` times is the
/// documented way to run `num_workers` concurrent replicas on one device, so
/// `cfg.num_workers` maps onto the *length* of `device_indices` rather than
/// onto a dedicated field. `num_workers` is clamped to at least 1 so a
/// misconfigured `0` still produces a working single-replica config instead
/// of an empty (rejected) device list.
pub(crate) fn build_config(cfg: &Ct2Config) -> Result<ct2rs::Config> {
    Ok(ct2rs::Config {
        device: parse_device(&cfg.device)?,
        compute_type: parse_compute_type(&cfg.compute_type)?,
        device_indices: vec![cfg.device_index; cfg.num_workers.max(1)],
        num_threads_per_replica: cfg.cpu_threads,
        ..Default::default()
    })
}

impl Ct2Asr {
    pub fn new(model_dir: &Path, cfg: Ct2Config) -> Result<Self> {
        let config = build_config(&cfg)?;

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

        Ok(Self {
            inner,
            options,
            model_dir: model_dir.to_path_buf(),
            config: cfg,
        })
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
                                speaker: None,
                            })
                            .collect()
                    })
                } else {
                    None
                },
                speaker: None,
            })
            .collect())
    }

    /// Detect the language with a real encoder pass.
    ///
    /// The detector is built here and dropped when this returns: it loads a
    /// second copy of the weights (see [`super::detect::LanguageDetector`] for
    /// why it cannot share ours), so it must not outlive the call. Callers
    /// that want to avoid the load entirely should pin the language instead.
    fn detect_language(&self, samples: &[f32]) -> Result<(String, f32)> {
        let detector =
            super::detect::LanguageDetector::new(&self.model_dir, build_config(&self.config)?)?;
        detector.detect(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_workers_becomes_that_many_device_indices() {
        let cfg = Ct2Config {
            num_workers: 3,
            device_index: 0,
            ..Default::default()
        };
        let config = build_config(&cfg).unwrap();
        assert_eq!(config.device_indices, vec![0, 0, 0]);
    }

    #[test]
    fn zero_num_workers_still_yields_one_device_index() {
        let cfg = Ct2Config {
            num_workers: 0,
            device_index: 2,
            ..Default::default()
        };
        let config = build_config(&cfg).unwrap();
        assert_eq!(config.device_indices, vec![2]);
    }
}
