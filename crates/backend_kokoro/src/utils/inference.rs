use std::io::Cursor;

use anyhow::anyhow;
use ndarray::{Array, IxDyn, array};
use ort::value::Value;
use ortts_onnx::inference_session;
use ortts_shared::{AppError, Downloader, SpeechOptions};
// use tokenizers::Tokenizer;

pub async fn inference(options: SpeechOptions) -> Result<Vec<u8>, AppError> {
  /// TODO: use options.text
  /// You can generate token ids as follows:
  ///   1. Convert input text to phonemes using https://github.com/hexgrad/misaki
  ///   2. Map phonemes to ids using https://huggingface.co/hexgrad/Kokoro-82M/blob/785407d1adfa7ae8fbef8ffd85f34ca127da3039/config.json#L34-L148
  let text = vec![
    50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16,
    70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62,
    131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138,
    56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102,
    54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16,
    81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102,
    56, 16, 44, 156, 76, 158, 123, 56, 4,
  ];

  let downloader = Downloader::new();

  let voice_name = format!("voices/{}.bin", options.voice);
  let voice_path = downloader
    .get_path("onnx-community/Kokoro-82M-v1.0-ONNX", &voice_name)
    .await?;
  let voice_bytes = std::fs::read(voice_path)?;
  let voices: Vec<f32> = voice_bytes
    .chunks_exact(4)
    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
    .collect();

  let token_len = TOKENS.len();
  let style_vector_size = 256;
  let style_vector_shape = IxDyn(&[1, style_vector_size]);

  let ref_s_start_index = token_len * style_vector_size;
  let ref_s_end_index = ref_s_start_index + style_vector_size;

  let ref_s_data: Vec<f32> = voices[ref_s_start_index..ref_s_end_index].to_vec();

  let ref_s_array = Array::from_shape_vec(style_vector_shape.clone(), ref_s_data)?.into_dyn();

  let mut input_ids_vec = vec![0i64];
  input_ids_vec.extend(TOKENS.into_iter());
  input_ids_vec.push(0i64);

  let input_ids_shape = IxDyn(&[1, input_ids_vec.len()]);
  let input_ids_array = Array::from_shape_vec(input_ids_shape, input_ids_vec)?.into_dyn();

  let speed_array = array![1.0f32].into_dyn();

  let model_path = downloader
    .get_path(
      "onnx-community/Kokoro-82M-v1.0-ONNX",
      "onnx/model_q4f16.onnx",
    )
    .await?;

  let mut session = inference_session(model_path)?;

  let input_ids_value = Value::from_array(input_ids_array)?;
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
    channels: 1,
    sample_rate: 24000,
    bits_per_sample: 32,
    sample_format: hound::SampleFormat::Float,
  };

  let mut buffer = Cursor::new(Vec::<u8>::new());
  let mut writer = hound::WavWriter::new(&mut buffer, spec)?;
  for sample in wav_squeezed {
    writer.write_sample(sample)?;
  }
  writer.finalize()?;

  Ok(buffer.into_inner())
}
