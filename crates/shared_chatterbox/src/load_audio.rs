use ortts_shared::AppError;
use std::{fs::File, path::PathBuf};
use symphonia::{
  core::{
    codecs::DecoderOptions,
    formats::FormatOptions,
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    probe::Hint,
  },
  default::get_probe,
};

use crate::resample_audio;

pub fn load_audio(path: PathBuf) -> Result<Vec<f32>, AppError> {
  // NOTICE: in python, librosa.load(..., sr=S3GEN_SR) resamples to 24000 Hz,
  // as the s3gen model requires 24kHz audio input, we will resample any audio
  // file into this target sample rate.
  const TARGET_SAMPLE_RATE: u32 = 24_000;

  let file = File::open(path)?;
  let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

  let mut hint = Hint::new();
  hint.with_extension("wav");

  let meta_opts = FormatOptions::default();
  let fmt_opts = MetadataOptions::default();

  let probed = get_probe().format(&hint, mss, &meta_opts, &fmt_opts)?;
  let mut format = probed.format;
  let track = format
    .tracks()
    .iter()
    .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
    .ok_or_else(|| anyhow::anyhow!("No supported audio tracks"))?;

  let source_sample_rate = track
    .codec_params
    .sample_rate
    .ok_or_else(|| anyhow::anyhow!("Missing sample rate in audio track"))?;
  let channel_count = track
    .codec_params
    .channels
    .map_or(1, symphonia::core::audio::Channels::count);

  let track_id = track.id;
  let mut decoder =
    symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

  let mut audio_buf = None;
  let mut samples = Vec::new();

  while let Ok(packet) = format.next_packet() {
    if packet.track_id() != track_id {
      continue;
    }

    match decoder.decode(&packet) {
      Ok(decoded) => {
        if audio_buf.is_none() {
          let spec = *decoded.spec();
          let duration = decoded.capacity() as u64;
          audio_buf = Some(symphonia::core::audio::SampleBuffer::<f32>::new(
            duration, spec,
          ));
        }

        if let Some(ref mut buf) = audio_buf {
          buf.copy_interleaved_ref(decoded);
          samples.extend_from_slice(buf.samples());
        }
      }
      Err(_) => break,
    }
  }

  if source_sample_rate == TARGET_SAMPLE_RATE {
    return Ok(samples);
  }

  resample_audio(
    &samples,
    source_sample_rate,
    TARGET_SAMPLE_RATE,
    channel_count,
  )
}
