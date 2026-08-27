use std::{fs::File, path::Path};

use anyhow::anyhow;
use ortts_shared::AppError;
use rubato::{
  Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::{
  core::{
    audio::SampleBuffer,
    codecs::DecoderOptions,
    formats::FormatOptions,
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    probe::Hint,
  },
  default::get_probe,
};

pub fn load_mono(path: &Path, target_rate: u32) -> Result<Vec<f32>, AppError> {
  let file = File::open(path)?;
  let stream = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
  let mut hint = Hint::new();
  if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
    hint.with_extension(extension);
  }

  let probed = get_probe().format(
    &hint,
    stream,
    &FormatOptions::default(),
    &MetadataOptions::default(),
  )?;
  let mut format = probed.format;
  let track = format
    .tracks()
    .iter()
    .find(|track| track.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
    .ok_or_else(|| anyhow!("reference audio has no supported audio track"))?;
  let source_rate = track
    .codec_params
    .sample_rate
    .ok_or_else(|| anyhow!("reference audio has no sample rate"))?;
  let channels = track
    .codec_params
    .channels
    .map_or(1, symphonia::core::audio::Channels::count);
  let track_id = track.id;
  let mut decoder =
    symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
  let mut interleaved = Vec::new();

  while let Ok(packet) = format.next_packet() {
    if packet.track_id() != track_id {
      continue;
    }
    let Ok(decoded_audio) = decoder.decode(&packet) else {
      break;
    };
    let mut buffer =
      SampleBuffer::<f32>::new(decoded_audio.capacity() as u64, *decoded_audio.spec());
    buffer.copy_interleaved_ref(decoded_audio);
    interleaved.extend_from_slice(buffer.samples());
  }

  if interleaved.is_empty() {
    return Err(anyhow!("reference audio contains no decoded samples").into());
  }

  let mono = downmix(&interleaved, channels);
  if source_rate == target_rate {
    return Ok(mono);
  }
  resample(&mono, source_rate, target_rate)
}

fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
  if channels <= 1 {
    return interleaved.to_vec();
  }
  let channels = u16::try_from(channels).expect("audio channel count should fit in u16");
  interleaved
    .chunks_exact(usize::from(channels))
    .map(|frame| frame.iter().sum::<f32>() / f32::from(channels))
    .collect()
}

fn resample(samples: &[f32], input_rate: u32, output_rate: u32) -> Result<Vec<f32>, AppError> {
  let parameters = SincInterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    interpolation: SincInterpolationType::Linear,
    oversampling_factor: 256,
    window: WindowFunction::BlackmanHarris2,
  };
  let mut resampler = SincFixedIn::<f32>::new(
    f64::from(output_rate) / f64::from(input_rate),
    2.0,
    parameters,
    samples.len().max(1),
    1,
  )
  .map_err(|error| anyhow!("failed to construct reference-audio resampler: {error}"))?;
  let mut output = resampler
    .process(&[samples.to_vec()], None)
    .map_err(|error| anyhow!("failed to resample reference audio: {error}"))?;
  let mut tail = resampler
    .process_partial::<Vec<f32>>(None, None)
    .map_err(|error| anyhow!("failed to flush reference-audio resampler: {error}"))?;
  let mut mono = output.pop().unwrap_or_default();
  mono.extend(tail.pop().unwrap_or_default());
  Ok(mono)
}

#[cfg(test)]
mod tests {
  use super::downmix;

  #[test]
  fn downmixes_stereo_frames() {
    assert_eq!(downmix(&[1.0, -1.0, 0.25, 0.75], 2), vec![0.0, 0.5]);
  }
}
