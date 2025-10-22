use ortts_shared::AppError;
use rubato::{
  Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub fn resample_audio(
  samples: &[f32],
  input_rate: u32,
  target_rate: u32,
  channels: usize,
) -> Result<Vec<f32>, AppError> {
  if input_rate == target_rate || samples.is_empty() {
    return Ok(samples.to_vec());
  }

  let frames_in = samples.len() / channels;
  if frames_in == 0 {
    return Ok(Vec::new());
  }

  let resample_ratio = target_rate as f64 / input_rate as f64;

  let params = SincInterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    interpolation: SincInterpolationType::Linear,
    oversampling_factor: 256,
    window: WindowFunction::BlackmanHarris2,
  };

  let chunk_size = frames_in.max(1);
  let mut resampler = SincFixedIn::<f32>::new(resample_ratio, 2.0, params, chunk_size, channels)
    .map_err(|e| {
      AppError::anyhow(&anyhow::anyhow!(format!(
        "Failed to construct resampler: {e}"
      )))
    })?;

  let mut channel_buffers = Vec::with_capacity(channels);
  for ch in 0..channels {
    let mut channel = Vec::with_capacity(frames_in);
    for frame in 0..frames_in {
      channel.push(samples[frame * channels + ch]);
    }
    channel_buffers.push(channel);
  }

  let mut resampled = resampler
    .process(&channel_buffers, None)
    .map_err(|e| AppError::anyhow(&anyhow::anyhow!(format!("Resampling failed: {e}"))))?;

  let residual = resampler
    .process_partial::<Vec<f32>>(None, None)
    .map_err(|e| {
      AppError::anyhow(&anyhow::anyhow!(format!(
        "Resampling tail flush failed: {e}"
      )))
    })?;

  for (channel, tail) in resampled.iter_mut().zip(residual.into_iter()) {
    channel.extend(tail);
  }

  if resampled.is_empty() {
    return Ok(Vec::new());
  }

  let frames_out = resampled[0].len();
  let mut interleaved = Vec::with_capacity(frames_out * channels);
  for frame in 0..frames_out {
    for ch in 0..channels {
      interleaved.push(resampled[ch][frame]);
    }
  }

  Ok(interleaved)
}
