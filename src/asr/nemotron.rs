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
    /// For `NemotronMode::EnglishOnly`, which has no language conditioning
    /// at all, only `"auto"` or an English variant (`"en"`, `"en-US"`, ...)
    /// can be honoured; anything else is rejected with an error rather than
    /// silently ignored -- see `resolve_target_lang`.
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

/// Whether `lang` denotes English (case/region variants included): `"en"`,
/// `"EN"`, `"en-US"`, `"en-GB"`, etc. Used to decide whether an
/// `EnglishOnly` model can honour a requested language without erroring.
fn is_english(lang: &str) -> bool {
    let lang = lang.split(['-', '_']).next().unwrap_or(lang);
    lang.eq_ignore_ascii_case("en")
}

/// Decide whether a requested language can be applied to `mode`, and if so,
/// whether `set_target_lang` actually needs calling.
///
/// - `Multilingual`: always honoured; the caller should forward `lang` to
///   `set_target_lang`.
/// - `EnglishOnly`: `"auto"` (the "don't force a language" sentinel) and any
///   English variant are trivially satisfied without calling
///   `set_target_lang` (the model has no language conditioning at all).
///   Anything else cannot be honoured and must error -- accept-and-ignore is
///   forbidden in this crate.
fn resolve_target_lang(mode: NemotronMode, lang: &str) -> Result<Option<&str>> {
    match mode {
        NemotronMode::Multilingual => Ok(Some(lang)),
        NemotronMode::EnglishOnly => {
            if lang == "auto" || is_english(lang) {
                Ok(None)
            } else {
                Err(Error::Nemotron(format!(
                    "target_lang {lang:?} cannot be honoured: the loaded model is \
                     English-only and has no language conditioning, so only \"auto\" \
                     or an English code (\"en\", \"en-US\", ...) can be satisfied"
                )))
            }
        }
    }
}

impl NemotronAsr {
    pub fn new(model_dir: &Path, config: NemotronConfig) -> Result<Self> {
        crate::onnx::init_ort()?;

        let mut nemotron = Nemotron::from_pretrained(model_dir, None)
            .map_err(|e| Error::Nemotron(format!("failed to load Nemotron model: {e}")))?;
        let mode = nemotron.mode();

        if let Some(lang) = &config.target_lang
            && let Some(lang) = resolve_target_lang(mode, lang)?
        {
            nemotron
                .set_target_lang(lang)
                .map_err(|e| Error::Nemotron(format!("unsupported target_lang {lang:?}: {e}")))?;
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

        if let Some(lang) = language
            && let Some(lang) = resolve_target_lang(self.mode, lang)?
        {
            nemotron
                .set_target_lang(lang)
                .map_err(|e| Error::Nemotron(format!("unsupported target_lang {lang:?}: {e}")))?;
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
                    // Placeholder: `transcribe_audio_with_timestamps`'s
                    // `TimedToken` carries no per-token confidence (only
                    // `text`/`start`/`end`). A real value would require a
                    // second full decode pass via
                    // `transcribe_audio_with_tokens`'s `TokenInfo::logprob`,
                    // which is too expensive to run just for this. Same
                    // convention as `src/python/iter.rs:159`.
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

    #[test]
    fn english_only_accepts_auto() {
        assert_eq!(
            resolve_target_lang(NemotronMode::EnglishOnly, "auto").unwrap(),
            None
        );
    }

    #[test]
    fn english_only_accepts_en() {
        assert_eq!(
            resolve_target_lang(NemotronMode::EnglishOnly, "en").unwrap(),
            None
        );
    }

    #[test]
    fn english_only_accepts_uppercase_region_variant() {
        assert_eq!(
            resolve_target_lang(NemotronMode::EnglishOnly, "EN-US").unwrap(),
            None
        );
    }

    #[test]
    fn english_only_rejects_non_english() {
        let err = resolve_target_lang(NemotronMode::EnglishOnly, "pt").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("pt"), "message should name the code: {msg}");
        assert!(
            msg.contains("English-only"),
            "message should explain why: {msg}"
        );
    }

    #[test]
    fn multilingual_accepts_non_english() {
        assert_eq!(
            resolve_target_lang(NemotronMode::Multilingual, "pt").unwrap(),
            Some("pt")
        );
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
