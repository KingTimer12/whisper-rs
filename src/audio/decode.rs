//! symphonia-backed decoding to interleaved f32, downmixed to mono.
//!
//! Written against symphonia 0.6, whose `formats`/`codecs` APIs are a
//! significant reshape from 0.5: audio buffers are now generic
//! (`GenericAudioBufferRef`) rather than typed `AudioBufferRef`, decoders are
//! split by media kind (`AudioDecoder`/`VideoDecoder`/`SubtitleDecoder`), and
//! `FormatReader::next_packet` returns `Ok(None)` at end of stream instead of
//! signalling it via an IO error.

use crate::error::{Error, Result};
use std::fs::File;
use std::path::Path;
use symphonia::core::codecs::audio::{AudioDecoderOptions, CODEC_ID_NULL_AUDIO};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub fn decode_file(path: &Path) -> Result<DecodedAudio> {
    let file = File::open(path).map_err(|e| Error::AudioRead {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format: Box<dyn FormatReader> = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| Error::AudioFormat {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

    let track = format
        .tracks()
        .iter()
        .find(|t| match &t.codec_params {
            Some(CodecParameters::Audio(params)) => params.codec != CODEC_ID_NULL_AUDIO,
            _ => false,
        })
        .ok_or_else(|| Error::AudioEmpty {
            path: path.to_path_buf(),
        })?;
    let track_id = track.id;

    let audio_params = match &track.codec_params {
        Some(CodecParameters::Audio(params)) => params.clone(),
        _ => {
            return Err(Error::AudioEmpty {
                path: path.to_path_buf(),
            })
        }
    };

    let sample_rate = audio_params.sample_rate.ok_or_else(|| Error::AudioFormat {
        path: path.to_path_buf(),
        message: "stream declares no sample rate".into(),
    })?;

    // `Packet::dur` is expressed in the track's time base, which for audio is
    // *usually* 1/sample_rate (making `dur` a frame count) but is not
    // guaranteed to be -- a container is free to declare, say, a millisecond
    // time base. The gap-filling path below needs frames, so derive the
    // conversion once here instead of assuming the common case: `dur` ticks
    // times `numer/denom` seconds per tick, times `sample_rate` frames per
    // second. When the track declares no time base at all, fall back to
    // treating `dur` as frames, which is what the common case would give.
    let frames_per_tick = track.time_base.map(|tb| {
        f64::from(tb.numer.get()) / f64::from(tb.denom.get()) * f64::from(sample_rate)
    });

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())
        .map_err(|e| Error::AudioFormat {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

    let mut out: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            // End of stream: symphonia 0.6 signals this with `Ok(None)`, not
            // an I/O error (see the module docstring), so this is the only
            // place a clean end-of-stream is recognised.
            Ok(None) => break,
            // A genuine I/O error reading the container (truncated file,
            // disk error, ...) used to be symphonia 0.5's EOF signal; in 0.6
            // clean end of stream is *exclusively* `Ok(None)` (see the
            // module docstring), so any `IoError` reaching this point is a
            // real failure, not an alternate EOF spelling -- including
            // `UnexpectedEof`, which in practice is exactly what a
            // truncated/corrupt read surfaces as (confirmed empirically: a
            // WAV chopped off mid-frame produces `UnexpectedEof` here, not a
            // clean `Ok(None)`). Treating it as end-of-stream is precisely
            // the bug this review flagged -- it must not be swallowed, or a
            // mid-file read failure silently truncates the transcript
            // instead of failing loudly.
            Err(symphonia::core::errors::Error::IoError(e)) => {
                return Err(Error::AudioRead {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                });
            }
            // `next_packet`'s `ResetRequired` means the *track list* changed
            // (chained Ogg, a container splicing in a new stream, ...) and
            // per symphonia's own contract on `FormatReader::next_packet`,
            // "the track list must be re-examined and all `Decoder`s
            // re-created" -- a bigger operation than `Decoder::reset` (which
            // only covers the decoder-level reset `AudioDecoder::decode`
            // documents for in-place parameter changes). Silently `break`ing
            // here drops everything after the reset with no error, which is
            // exactly the failure mode this review flagged. Correctly
            // handling it would mean re-deriving the track, codec params,
            // and sample rate this function fixes once before the loop, for
            // a stream shape (multiple logical bitstreams concatenated in
            // one file) this crate does not otherwise claim to support, so
            // the safer choice is to fail loudly rather than guess: return
            // an error naming what happened instead of continuing with a
            // decoder or track assumptions that may no longer be valid.
            Err(symphonia::core::errors::Error::ResetRequired) => {
                return Err(Error::AudioRead {
                    path: path.to_path_buf(),
                    message: "the stream requires a reset (its track list changed \
                              mid-file, e.g. a chained/concatenated container) \
                              -- unsupported"
                        .into(),
                });
            }
            Err(e) => {
                return Err(Error::AudioRead {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })
            }
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(buf) => {
                let channels = buf.spec().channels().count().max(1);
                let mut interleaved: Vec<f32> = Vec::new();
                buf.copy_to_vec_interleaved(&mut interleaved);
                for frame in interleaved.chunks(channels) {
                    let sum: f32 = frame.iter().sum();
                    out.push(sum / channels as f32);
                }
            }
            // A corrupt packet mid-file must not lose the whole file, but it
            // also must not silently drop that packet's samples: doing so
            // shifts every later sample earlier by the packet's duration,
            // so every timestamp for the rest of the file is wrong. Push
            // silence for exactly the packet's declared duration instead,
            // keeping the timeline aligned, and warn so this is visible.
            Err(symphonia::core::errors::Error::DecodeError(msg)) => {
                let frames = match frames_per_tick {
                    Some(per_tick) => (packet.dur.get() as f64 * per_tick).round() as usize,
                    None => packet.dur.get() as usize,
                };
                tracing::warn!(
                    "corrupt packet in {}: {msg}; inserting {frames} frames of silence to keep the timeline aligned",
                    path.display(),
                );
                out.resize(out.len() + frames, 0.0);
            }
            Err(e) => {
                return Err(Error::AudioRead {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })
            }
        }
    }

    if out.is_empty() {
        return Err(Error::AudioEmpty {
            path: path.to_path_buf(),
        });
    }

    Ok(DecodedAudio {
        samples: out,
        sample_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a mono 16-bit integer PCM WAV of a 440 Hz sine at `rate` Hz for `secs` seconds.
    ///
    /// Real-world mp3/wav files are integer PCM, not float, so this exercises the
    /// symphonia sample-format conversion path that the brief's float-only fixtures
    /// (see `audio::tests::write_sine_wav`) never touch.
    fn write_sine_wav_i16(path: &std::path::Path, rate: u32, secs: f32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        let frames = (rate as f32 * secs) as usize;
        for i in 0..frames {
            let t = i as f32 / rate as f32;
            let v = (t * 440.0 * std::f32::consts::TAU).sin() * 0.5;
            let sample = (v * i16::MAX as f32) as i16;
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn a_wav_truncated_mid_data_is_not_silently_treated_as_a_success_shift() {
        // Write a valid WAV, then chop it off partway through the data chunk
        // so the RIFF header's declared byte count no longer matches the
        // bytes actually on disk. Before this fix, any IoError on
        // `next_packet` (which is exactly what a truncated read produces)
        // was unconditionally treated as a clean end of stream and the
        // function returned `Ok` with whatever samples had been decoded so
        // far -- a corrupt/truncated file looked identical to a short valid
        // one. This asserts the fix: a truncation deep enough to break the
        // container framing must surface as a real, typed error, not a
        // quiet partial success.
        let dir = std::env::temp_dir().join("whisper_rs_t2_truncated");
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.wav");
        write_sine_wav_i16(&good, 16_000, 2.0);

        let bytes = std::fs::read(&good).unwrap();
        let truncated = dir.join("truncated.wav");
        // Cut it off mid-header/mid-frame (not on a sample boundary), which
        // is what turns a "just fewer packets" truncation into a real
        // corrupt-container read failure rather than a clean short EOF.
        std::fs::write(&truncated, &bytes[..bytes.len() / 2 + 1]).unwrap();

        let result = decode_file(&truncated);

        // This must not silently succeed with a truncated/shifted sample
        // count and no error at all.
        assert!(
            result.is_err(),
            "a mid-file truncation must surface as an error, not a quiet partial decode"
        );
    }

    #[test]
    fn decodes_16bit_integer_pcm_wav_normalised() {
        let dir = std::env::temp_dir().join("whisper_rs_t2_pcm");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mono16k_i16.wav");
        write_sine_wav_i16(&path, 16_000, 1.0);

        let decoded = decode_file(&path).unwrap();

        assert_eq!(decoded.sample_rate, 16_000);
        assert!(
            (decoded.samples.len() as i64 - 16_000).abs() <= 1,
            "expected ~16000 samples, got {}",
            decoded.samples.len()
        );
        assert!(
            decoded.samples.iter().all(|s| s.abs() <= 1.0),
            "integer PCM must be normalised to [-1, 1], got a sample outside that range"
        );

        // A 0.5-amplitude 16-bit sine must decode to a clearly non-silent peak. This is
        // what catches a missing or wrong scale factor: a bug here shows up as either
        // near-zero output (never scaled up from ~1/32768) or values in the thousands
        // (never scaled down at all).
        let peak = decoded.samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(peak > 0.3, "expected a clearly non-silent peak, got {peak}");
    }
}
