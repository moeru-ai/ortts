use std::io::Cursor;

use ndarray::{Array, IxDyn, array};
use ort::value::Value;
use ortts_onnx::inference_session;
use ortts_shared::{AppError, Downloader, SpeechOptions};

use crate::utils::{Tokenizer, phonemize};

pub async fn inference(options: SpeechOptions) -> Result<Vec<u8>, AppError> {
  let downloader = Downloader::new("onnx-community/Kokoro-82M-v1.0-ONNX".to_owned());
  let tokenizer = Tokenizer::new().await?;

  let phonemes = phonemize(options.input, true).await?;

  let mut input_ids = vec![0i64];
  input_ids.extend(tokenizer.encode(&phonemes).into_iter());
  input_ids.push(0i64);

  let input_ids_shape = [1_usize, input_ids.len()];
  let input_ids_array = ndarray::Array2::<i64>::from_shape_vec(
    (input_ids_shape[0], input_ids_shape[1]),
    input_ids.clone(),
  )?;
  let input_ids_value = Value::from_array(input_ids_array)?;

  let voice_name = format!("voices/{}.bin", options.voice);
  let voice_path = downloader.get_path(&voice_name).await?;
  let voice_bytes = std::fs::read(voice_path)?;
  let voices: Vec<f32> = voice_bytes
    .chunks_exact(4)
    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
    .collect();

  let token_len = input_ids.len();
  let style_vector_size = 256;
  let style_vector_shape = IxDyn(&[1, style_vector_size]);

  let ref_s_start_index = token_len * style_vector_size;
  let ref_s_end_index = ref_s_start_index + style_vector_size;

  let ref_s_data: Vec<f32> = voices[ref_s_start_index..ref_s_end_index].to_vec();

  let ref_s_array = Array::from_shape_vec(style_vector_shape.clone(), ref_s_data)?.into_dyn();

  let speed_array = array![1.0f32].into_dyn();

  let model_path = downloader.get_path("onnx/model_q4f16.onnx").await?;

  let mut session = inference_session(model_path)?;

  let style_value = Value::from_array(ref_s_array)?;
  let speed_value = Value::from_array(speed_array)?;

  let outputs = session.run(ort::inputs![
    "input_ids" => input_ids_value,
    "style" => style_value,
    "speed" => speed_value
  ])?;

  let (_, wav) = outputs.into_iter().next().unwrap();
  let (wav_shape, wav_data) = wav.try_extract_tensor::<f32>()?;

  tracing::debug!("wav shape: {:?}, length: {}", wav_shape, wav_data.len());
  let wav_squeezed = wav_data.to_vec();

  tracing::debug!("Generated audio with {} samples", wav_squeezed.len());

  let spec = hound::WavSpec {
    channels: 2,
    sample_rate: 24000,
    bits_per_sample: 32,
    sample_format: hound::SampleFormat::Float,
  };

  let mut buffer = Cursor::new(Vec::<u8>::new());
  let mut writer = hound::WavWriter::new(&mut buffer, spec)?;
  for sample in wav_squeezed {
    writer.write_sample(sample)?;
    writer.write_sample(sample)?;
  }
  writer.finalize()?;

  Ok(buffer.into_inner())
}
