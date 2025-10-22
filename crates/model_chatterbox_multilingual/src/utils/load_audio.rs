use ortts_shared::AppError;
use std::fs::File;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use symphonia::default::get_probe;

use super::resample_audio;

pub fn load_audio(path: &str) -> Result<Vec<f32>, AppError> {
  // NOTICE: in python, librosa.load(..., sr=S3GEN_SR) resamples to 24000 Hz,
  // as the s3gen model requires 24kHz audio input, we will resample any audio
  // file into this target sample rate.
  const TARGET_SAMPLE_RATE: u32 = 24_000;

  let file = File::open(path)?;
  let mss = MediaSourceStream::new(Box::new(file), Default::default());

  let mut hint = Hint::new();
  hint.with_extension("wav");

  let meta_opts = Default::default();
  let fmt_opts = Default::default();

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
  let channel_count = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

  let track_id = track.id;
  let mut decoder =
    symphonia::default::get_codecs().make(&track.codec_params, &Default::default())?;

  let mut audio_buf = None;
  let mut samples = Vec::new();

  loop {
    let packet = match format.next_packet() {
      Ok(packet) => packet,
      Err(_) => break,
    };

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
