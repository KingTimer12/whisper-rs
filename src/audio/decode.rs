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
            // End of stream.
            Ok(None) => break,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(symphonia::core::errors::Error::ResetRequired) => break,
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
            // A corrupt packet mid-file should not lose the whole file.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
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
