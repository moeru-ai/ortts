use anyhow::anyhow;
use futures::stream;
use ndarray::{Array, Array2, IxDyn, array};
use ort::value::Value;
use ortts_onnx::{SessionPool, inference_session};
use ortts_shared::{AppError, AudioSpec, Downloader, SpeechAudioStream, SpeechOptions};

use crate::utils::{Tokenizer, prepare_segments};

const SAMPLE_RATE: u32 = 24_000;
const CHANNELS: u16 = 2;
const STYLE_VECTOR_SIZE: usize = 256;

pub async fn inference(options: SpeechOptions) -> Result<SpeechAudioStream, AppError> {
  let state = KokoroStream::prepare(options).await?;
  let audio_stream = stream::try_unfold(state, |state| async move {
    let (chunk, state) = tokio::task::spawn_blocking(move || {
      let mut state = state;
      let chunk = state.next_audio_chunk();
      (chunk, state)
    })
    .await
    .map_err(|error| AppError::from(anyhow!(error)))?;
    let chunk = chunk?;
    Ok(chunk.map(|chunk| (chunk, state)))
  });

  Ok(SpeechAudioStream::new(
    AudioSpec::new(CHANNELS, SAMPLE_RATE),
    audio_stream,
  ))
}

struct KokoroStream {
  session: SessionPool,
  voices: Vec<f32>,
  segments: Vec<Vec<i64>>,
  segment_index: usize,
}

impl KokoroStream {
  async fn prepare(options: SpeechOptions) -> Result<Self, AppError> {
    let downloader = Downloader::new("onnx-community/Kokoro-82M-v1.0-ONNX".to_owned())?;
    let model_path = downloader.get_path("onnx/model_q4f16.onnx").await?;
    let voice_path = downloader
      .get_path(&format!("voices/{}.bin", options.voice))
      .await?;
    let voice_bytes = std::fs::read(voice_path)?;
    let voices: Vec<f32> = voice_bytes
      .chunks_exact(4)
      .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte voice sample")))
      .collect();

    let tokenizer = Tokenizer::new().await?;
    let segments = prepare_segments(&options.input, &tokenizer).await?;

    Ok(Self {
      session: inference_session(&model_path)?,
      voices,
      segments,
      segment_index: 0,
    })
  }

  fn next_audio_chunk(&mut self) -> Result<Option<Vec<f32>>, AppError> {
    let Some(input_ids) = self.segments.get(self.segment_index) else {
      return Ok(None);
    };
    self.segment_index += 1;

    let input_ids_array = Array2::<i64>::from_shape_vec((1, input_ids.len()), input_ids.clone())?;
    let input_ids_value = Value::from_array(input_ids_array)?;
    let token_len = input_ids.len();
    let ref_s_start_index = token_len * STYLE_VECTOR_SIZE;
    let ref_s_end_index = ref_s_start_index + STYLE_VECTOR_SIZE;
    let ref_s_data = self
      .voices
      .get(ref_s_start_index..ref_s_end_index)
      .ok_or_else(|| anyhow!("voice does not contain style for {token_len} tokens"))?;
    let ref_s_array = Array::from_shape_vec(IxDyn(&[1, STYLE_VECTOR_SIZE]), ref_s_data.to_vec())?;
    let speed_value = Value::from_array(array![1.0_f32].into_dyn())?;
    let outputs = self.session.run(ort::inputs![
      "input_ids" => input_ids_value,
      "style" => Value::from_array(ref_s_array)?,
      "speed" => speed_value,
    ])?;
    let (_, wav) = outputs
      .into_iter()
      .next()
      .ok_or_else(|| anyhow!("Kokoro returned no waveform"))?;
    let (_, wav_data) = wav.try_extract_tensor::<f32>()?;

    let mut interleaved = Vec::with_capacity(wav_data.len() * CHANNELS as usize);
    for sample in wav_data.iter().copied() {
      interleaved.extend([sample, sample]);
    }
    Ok(Some(interleaved))
  }
}
