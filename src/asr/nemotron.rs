//! `parakeet-rs`'s Nemotron (0.6B) backend.
//!
//! Entirely behind the `nemotron` feature: without it, `parakeet-rs` (and
//! `ort`, unless `diarization` also pulls it in) is absent from the
//! dependency graph.

use super::Asr;
use crate::error::{Error, Result};
use crate::types::{Seg, Word};
use parakeet_rs::{Nemotron, NemotronMode, TimestampMode};
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, Clone, Default)]
pub struct NemotronConfig {
    /// Ignored for `NemotronMode::EnglishOnly`, which has no language
    /// conditioning at all.
    pub target_lang: Option<String>,
}

pub struct NemotronAsr {
    inner: Mutex<Nemotron>,
    mode: NemotronMode,
}

/// A SentencePiece language-tag piece, e.g. `<en-US>` or `<en>`: `<`, two or
/// five inner characters matching `xx` or `xx-XX`, `>`. Mirrors the shape
/// `parakeet-rs`'s own (private) `is_lang_tag` checks, reimplemented here
/// because that helper is not exported.
fn parse_lang_tag(piece: &str) -> Option<&str> {
    let bytes = piece.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'<' || bytes[bytes.len() - 1] != b'>' {
        return None;
    }
    let inner = &piece[1..piece.len() - 1];
    let inner_bytes = inner.as_bytes();
    let shape_ok = match inner_bytes.len() {
        2 => inner_bytes[0].is_ascii_lowercase() && inner_bytes[1].is_ascii_lowercase(),
        5 => {
            inner_bytes[0].is_ascii_lowercase()
                && inner_bytes[1].is_ascii_lowercase()
                && inner_bytes[2] == b'-'
                && inner_bytes[3].is_ascii_uppercase()
                && inner_bytes[4].is_ascii_uppercase()
        }
        _ => false,
    };
    shape_ok.then_some(inner)
}

impl NemotronAsr {
    pub fn new(model_dir: &Path, config: NemotronConfig) -> Result<Self> {
        crate::onnx::init_ort()?;

        let mut nemotron = Nemotron::from_pretrained(model_dir, None)
            .map_err(|e| Error::Nemotron(format!("failed to load Nemotron model: {e}")))?;
        let mode = nemotron.mode();

        #[allow(clippy::collapsible_if)] // two independent conditions read more clearly separate
        if mode == NemotronMode::Multilingual {
            if let Some(lang) = &config.target_lang {
                nemotron.set_target_lang(lang).map_err(|e| {
                    Error::Nemotron(format!("unsupported target_lang {lang:?}: {e}"))
                })?;
            }
        }

        Ok(Self {
            inner: Mutex::new(nemotron),
            mode,
        })
    }
}

impl Asr for NemotronAsr {
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        word_timestamps: bool,
    ) -> Result<Vec<Seg>> {
        let mut nemotron = self
            .inner
            .lock()
            .map_err(|_| Error::Nemotron("model lock poisoned".into()))?;

        #[allow(clippy::collapsible_if)] // two independent conditions read more clearly separate
        if self.mode == NemotronMode::Multilingual {
            if let Some(lang) = language {
                nemotron.set_target_lang(lang).map_err(|e| {
                    Error::Nemotron(format!("unsupported target_lang {lang:?}: {e}"))
                })?;
            }
        }

        let mode = if word_timestamps {
            TimestampMode::Words
        } else {
            TimestampMode::Tokens
        };
        let result = nemotron
            .transcribe_audio_with_timestamps(samples, Some(mode))
            .map_err(|e| Error::Nemotron(format!("transcription failed: {e}")))?;

        if result.tokens.is_empty() {
            return Ok(Vec::new());
        }

        let start = result.tokens.first().map(|t| t.start).unwrap_or(0.0);
        let end = result.tokens.last().map(|t| t.end).unwrap_or(start);

        let words = word_timestamps.then(|| {
            result
                .tokens
                .iter()
                .map(|t| Word {
                    start: t.start,
                    end: t.end,
                    text: t.text.clone(),
                    probability: 1.0,
                    speaker: None,
                })
                .collect::<Vec<_>>()
        });

        Ok(vec![Seg {
            id: 0,
            start,
            end,
            text: result.text,
            words,
            speaker: None,
        }])
    }

    fn detect_language(&self, samples: &[f32]) -> Result<(String, f32)> {
        if self.mode == NemotronMode::EnglishOnly {
            return Ok(("en".to_string(), 1.0));
        }

        let mut nemotron = self
            .inner
            .lock()
            .map_err(|_| Error::Nemotron("model lock poisoned".into()))?;

        // `transcribe_audio_with_timestamps` strips language-tag tokens (it
        // filters `lang_tag_ids` out before returning), so it cannot be used
        // to recover the detected tag. `transcribe_audio_with_tokens` returns
        // every emitted token, tags included, so use that instead.
        let tokens = nemotron
            .transcribe_audio_with_tokens(samples)
            .map_err(|e| Error::Nemotron(format!("language detection pass failed: {e}")))?;

        match tokens
            .iter()
            .find_map(|t| parse_lang_tag(&t.text).map(|lang| (lang, t.logprob)))
        {
            // `logprob` is the tag token's log-softmax over the *entire*
            // vocabulary (~13k pieces: every language tag and every ordinary
            // text piece), not a distribution restricted to language
            // alternatives. `.exp()` turns it into a well-formed value in
            // (0, 1], but it answers "how confident was the decoder in
            // emitting this exact token next, versus 13k unrelated
            // alternatives" — a proxy correlated with language confidence,
            // not a language-detection probability. It can be misleadingly
            // high or low independent of actual language ambiguity (e.g. a
            // tag can dominate the position-0 softmax just because few other
            // tokens are plausible there, regardless of how ambiguous the
            // language is). This is NOT comparable to the Whisper backend's
            // `detect_language` (`src/asr/detect.rs`), whose probability is
            // a genuine softmax over language candidates only. Still an
            // improvement over a hardcoded constant, since it does vary with
            // the model's actual output.
            Some((lang, logprob)) => Ok((lang.to_string(), logprob.exp())),
            None => Ok(("unknown".to_string(), 0.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_two_letter_tag_parses() {
        assert_eq!(parse_lang_tag("<en>"), Some("en"));
    }

    #[test]
    fn a_locale_tag_parses() {
        assert_eq!(parse_lang_tag("<en-US>"), Some("en-US"));
    }

    #[test]
    fn ordinary_text_is_not_a_tag() {
        assert_eq!(parse_lang_tag("hello"), None);
        assert_eq!(parse_lang_tag("<hi"), None);
        assert_eq!(parse_lang_tag("hi>"), None);
        assert_eq!(parse_lang_tag(""), None);
    }

    #[test]
    fn wrong_case_is_not_a_tag() {
        assert_eq!(parse_lang_tag("<EN>"), None);
        assert_eq!(parse_lang_tag("<en-us>"), None);
    }

    /// The Arc<NemotronAsr> design in the Python layer requires this. It is
    /// already guaranteed by `impl Asr for NemotronAsr` compiling (`Asr:
    /// Send + Sync`, see `src/asr/mod.rs`) — this records that fact rather
    /// than leaving it as an open question.
    #[test]
    fn nemotron_asr_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NemotronAsr>();
    }
}
