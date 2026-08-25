//! Encoder-only Whisper language detection.
//!
//! # Why this exists
//!
//! `ct2rs::Whisper` (the high-level API this crate transcribes with) detects
//! the language internally -- `generate`/`generate_segments` call a private
//! `detect_language_token` whenever `language` is `None` -- but it never
//! reports *what* it detected, and the language token is part of the decoder
//! *prompt*, so it is absent from the generated sequence the public API
//! returns. Recovering the code by scanning that output (the previous
//! approach) therefore always failed and silently degraded to the string
//! `"unknown"`, which was then handed back to the decoder as the literal
//! language token `<|unknown|>` -- producing confident garbage transcripts
//! whenever the caller did not pin `language=`.
//!
//! The real detector *is* reachable, one layer down:
//! [`ct2rs::sys::Whisper::detect_language`] takes an encoder input and
//! returns every language with its probability, best first. It needs a log-mel
//! spectrogram, which `ct2rs` builds with the public `mel_spec` crate from the
//! model's `preprocessor_config.json`; both are reproduced here.
//!
//! # Cost
//!
//! `ct2rs::Whisper` owns its `sys::Whisper` privately, so a detector cannot
//! borrow the already-loaded weights and must load its own copy. Callers are
//! expected to build a detector, use it, and drop it (see
//! `Ct2Asr::detect_language`) rather than hold one alongside the transcriber,
//! so the second copy is transient rather than doubling resident memory for
//! the life of the model. Detection itself is a single encoder pass -- much
//! cheaper than the full 30 s decode it replaces.

use crate::error::{Error, Result};
use mel_spec::mel::{log_mel_spectrogram, mel, norm_mel};
use mel_spec::stft::Spectrogram;
use ndarray::{s, stack, Array2, Array3, Axis};
use serde::Deserialize;
use std::path::Path;

/// The subset of `preprocessor_config.json` needed to build the mel input.
///
/// Field names and the `mel_filters` fallback mirror `ct2rs`'s own
/// `PreprocessorConfig` so a directory that loads in `ct2rs::Whisper::new`
/// also loads here.
#[derive(Deserialize)]
struct PreprocessorConfig {
    feature_size: usize,
    hop_length: usize,
    n_fft: usize,
    nb_max_frames: usize,
    sampling_rate: usize,
    mel_filters: Option<Vec<Vec<f64>>>,
}

pub struct LanguageDetector {
    whisper: ct2rs::sys::Whisper,
    feature_size: usize,
    hop_length: usize,
    n_fft: usize,
    nb_max_frames: usize,
    mel_filters: Array2<f64>,
}

impl LanguageDetector {
    /// Load a detector from the same directory the transcriber uses.
    ///
    /// `preprocessor_config.json` is guaranteed to exist by
    /// `models::hub::ensure_preprocessor_config`, which synthesizes one when
    /// the upstream repo (e.g. `Systran/faster-whisper-*`) omits it.
    pub fn new(model_dir: &Path, config: ct2rs::Config) -> Result<Self> {
        let whisper = ct2rs::sys::Whisper::new(model_dir, config)
            .map_err(|e| Error::Ct2(format!("loading the language detector failed: {e}")))?;

        let path = model_dir.join("preprocessor_config.json");
        let file = std::fs::File::open(&path).map_err(|e| {
            Error::Ct2(format!("reading {} failed: {e}", path.display()))
        })?;
        let cfg: PreprocessorConfig = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| Error::Ct2(format!("parsing {} failed: {e}", path.display())))?;

        let mel_filters = match cfg.mel_filters {
            Some(rows) => {
                let height = rows.len();
                let width = rows.first().map(Vec::len).unwrap_or_default();
                Array2::from_shape_vec(
                    (height, width),
                    rows.into_iter().flatten().collect::<Vec<f64>>(),
                )
                .map_err(|e| Error::Ct2(format!("mel_filters in {} is ragged: {e}", path.display())))?
            }
            // Same call ct2rs makes when the key is absent.
            None => mel(
                cfg.sampling_rate as f64,
                cfg.n_fft,
                cfg.feature_size,
                None,
                None,
                false,
                true,
            ),
        };

        Ok(Self {
            whisper,
            feature_size: cfg.feature_size,
            hop_length: cfg.hop_length,
            n_fft: cfg.n_fft,
            nb_max_frames: cfg.nb_max_frames,
            mel_filters,
        })
    }

    /// Detect the language of one window.
    ///
    /// Returns the bare code (`"en"`, not `<|en|>`) and its probability.
    /// `samples` must be one window's worth of 16 kHz mono audio in `[-1, 1]`;
    /// anything past the first `nb_max_frames` mel frames is ignored, matching
    /// the single-chunk encoder input Whisper expects.
    pub fn detect(&self, samples: &[f32]) -> Result<(String, f32)> {
        let mut features = self.mel_spectrogram(samples);
        let shape = features.shape().to_vec();

        let view = ct2rs::sys::StorageView::new(
            &shape,
            features
                .as_slice_mut()
                .expect("the mel array is built in standard layout"),
            Default::default(),
        )
        .map_err(|e| Error::Ct2(format!("building the detector input failed: {e}")))?;

        let detected = self
            .whisper
            .detect_language(&view)
            .map_err(|e| Error::Ct2(format!("language detection failed: {e}")))?;

        let best = detected
            .into_iter()
            .next()
            .and_then(|batch| batch.into_iter().next())
            .ok_or_else(|| Error::Ct2("language detection returned no candidates".into()))?;

        Ok((strip_token(&best.language), best.probability))
    }

    /// Build the single-chunk log-mel spectrogram the encoder expects.
    ///
    /// Mirrors `ct2rs`'s `generate_mel_spectrogram` for the one-chunk case:
    /// frames past `nb_max_frames` are dropped rather than starting a second
    /// chunk, because a detector only ever looks at the first window.
    fn mel_spectrogram(&self, samples: &[f32]) -> Array3<f32> {
        let mut stft = Spectrogram::new(self.n_fft, self.hop_length);
        let mut frames: Array2<f32> = Array2::zeros((self.feature_size, self.nb_max_frames));

        // The column index is the *input* frame index, not a count of
        // successful FFTs: `Spectrogram::add` returns `None` while its window
        // fills, and ct2rs leaves those columns zero rather than compacting
        // later frames into them. Detection must see the exact features
        // transcription would, so this indexing matches ct2rs's.
        for (column, chunk) in samples.chunks(self.hop_length).enumerate() {
            if column >= self.nb_max_frames {
                break;
            }
            if let Some(fft) = stft.add(chunk) {
                let value = norm_mel(&log_mel_spectrogram(&fft, &self.mel_filters))
                    .mapv(|v| v as f32);
                frames.slice_mut(s![.., column]).assign(&value.slice(s![.., 0]));
            }
        }

        let mut stacked = stack(Axis(0), &[frames.view()])
            .expect("stacking one array of a fixed shape cannot fail");
        if !stacked.is_standard_layout() {
            stacked = stacked.as_standard_layout().into_owned();
        }
        stacked
    }
}

/// `<|en|>` -> `en`. Anything not shaped like a token is returned unchanged.
fn strip_token(token: &str) -> String {
    token
        .strip_prefix("<|")
        .and_then(|rest| rest.strip_suffix("|>"))
        .unwrap_or(token)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_a_language_token() {
        assert_eq!(strip_token("<|en|>"), "en");
    }

    #[test]
    fn leaves_a_bare_code_alone() {
        assert_eq!(strip_token("en"), "en");
    }

    #[test]
    fn leaves_a_half_formed_token_alone() {
        assert_eq!(strip_token("<|en"), "<|en");
    }
}
