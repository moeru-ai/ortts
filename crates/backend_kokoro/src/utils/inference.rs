use anyhow::anyhow;
use futures::stream;
use ndarray::{Array, Array2, IxDyn, array};
use ort::value::Value;
use ortts_onnx::{SessionPool, inference_session};
use ortts_shared::{
  AppError, AudioSpec, Downloader, SpeechOptions, SpeechStream, collect_speech_stream,
};

use crate::utils::{Tokenizer, phonemize};

const SAMPLE_RATE: u32 = 24_000;
const CHANNELS: u16 = 2;
const STYLE_VECTOR_SIZE: usize = 256;
const MAX_SEGMENT_CHARS: usize = 240;

pub async fn inference(options: SpeechOptions) -> Result<Vec<u8>, AppError> {
  collect_speech_stream(inference_stream(options).await?).await
}

pub async fn inference_stream(options: SpeechOptions) -> Result<SpeechStream, AppError> {
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

  Ok(SpeechStream::new(
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
    let segments = split_text(&options.input);
    let mut phoneme_segments = Vec::with_capacity(segments.len());
    for segment in segments {
      let phonemes = phonemize(segment, true).await?;
      let mut input_ids = vec![0_i64];
      input_ids.extend(tokenizer.encode(&phonemes));
      input_ids.push(0_i64);
      if input_ids.len() > 2 {
        phoneme_segments.push(input_ids);
      }
    }

    Ok(Self {
      session: inference_session(&model_path)?,
      voices,
      segments: phoneme_segments,
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

fn split_text(text: &str) -> Vec<String> {
  let mut sentences = Vec::new();
  let mut start = 0;

  for (index, character) in text.char_indices() {
    if matches!(character, '.' | '!' | '?' | '。' | '！' | '？' | '\n') {
      let end = index + character.len_utf8();
      if let Some(segment) = text.get(start..end).map(str::trim)
        && !segment.is_empty()
      {
        sentences.push(segment.to_owned());
      }
      start = end;
    }
  }

  if let Some(segment) = text.get(start..).map(str::trim)
    && !segment.is_empty()
  {
    sentences.push(segment.to_owned());
  }

  let mut segments = Vec::new();
  for sentence in sentences {
    let mut current = String::new();
    let mut current_chars = 0;
    for word in sentence.split_whitespace() {
      let word_chars = word.chars().count();
      if word_chars > MAX_SEGMENT_CHARS {
        if !current.is_empty() {
          segments.push(std::mem::take(&mut current));
          current_chars = 0;
        }

        let characters: Vec<_> = word.chars().collect();
        for chunk in characters.chunks(MAX_SEGMENT_CHARS) {
          segments.push(chunk.iter().collect());
        }
        continue;
      }

      let separator_chars = if current.is_empty() { 0 } else { 1 };
      if current_chars + separator_chars + word_chars > MAX_SEGMENT_CHARS && !current.is_empty() {
        segments.push(std::mem::take(&mut current));
        current_chars = 0;
      }
      if !current.is_empty() {
        current.push(' ');
        current_chars += 1;
      }
      current.push_str(word);
      current_chars += word_chars;
    }
    if !current.is_empty() {
      segments.push(current);
    }
  }

  segments
}
