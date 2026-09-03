use std::collections::HashMap;

use anyhow::anyhow;
use futures::stream;
use half::f16;
use ndarray::{Array2, Array3, Array4, ArrayView3, Axis};
use ort::{
  inputs,
  tensor::TensorElementType,
  value::{Value, ValueType},
};
use ortts_onnx::{SessionPool, inference_session};
use ortts_shared::{AppError, AudioSpec, Downloader, SpeechAudioStream, SpeechOptions};
use ortts_shared_chatterbox::{RepetitionPenaltyLogitsProcessor, load_audio};
use tokenizers::Tokenizer;

const MAX_NEW_TOKENS: usize = 1024;
const STREAM_TOKENS: usize = 32;
const SAMPLE_RATE: u32 = 24_000;
const START_SPEECH_TOKEN: u32 = 6561;
const STOP_SPEECH_TOKEN: u32 = 6562;
const SILENCE_TOKEN: u32 = 4299;
const NUM_HIDDEN_LAYERS: i64 = 24;
const NUM_KV_HEADS: usize = 16;
const HEAD_DIM: usize = 64;

pub async fn inference(options: SpeechOptions) -> Result<SpeechAudioStream, AppError> {
  let state = TurboStream::prepare(options).await?;
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
    AudioSpec::new(1, SAMPLE_RATE),
    audio_stream,
  ))
}

struct TurboStream {
  embed_tokens_session: SessionPool,
  language_model_session: SessionPool,
  conditional_decoder_session: SessionPool,
  input_ids: Array2<i64>,
  attention_mask: Array2<i64>,
  position_ids: Array2<i64>,
  next_inputs_embeds: Option<Array3<f32>>,
  prompt_token: Array2<i64>,
  speaker_embeddings: Array2<f32>,
  speaker_features: Array3<f32>,
  repetition_penalty_processor: RepetitionPenaltyLogitsProcessor,
  past_key_values: HashMap<String, Value>,
  generate_tokens: Array2<usize>,
  speech_tokens: Vec<i64>,
  pending_tokens: usize,
  decoded_samples: usize,
  iteration: usize,
  batch_size: usize,
  finished: bool,
  final_audio_emitted: bool,
}

impl TurboStream {
  async fn prepare(options: SpeechOptions) -> Result<Self, AppError> {
    let downloader = Downloader::new("ResembleAI/chatterbox-turbo-ONNX".to_owned())?;
    let (
      speech_encoder_path,
      embed_tokens_path,
      language_model_path,
      conditional_decoder_path,
      tokenizer_path,
    ) = tokio::try_join!(
      downloader.get_onnx_with_data("onnx/speech_encoder.onnx"),
      downloader.get_onnx_with_data("onnx/embed_tokens.onnx"),
      downloader.get_onnx_with_data("onnx/language_model.onnx"),
      downloader.get_onnx_with_data("onnx/conditional_decoder.onnx"),
      downloader.get_tokenizer(),
    )?;

    let mut embed_tokens_session = inference_session(&embed_tokens_path)?;
    let mut speech_encoder_session = inference_session(&speech_encoder_path)?;
    let language_model_session = inference_session(&language_model_path)?;
    let conditional_decoder_session = inference_session(&conditional_decoder_path)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

    let target_voice_path = match options.voice.as_str() {
      "alloy" => {
        let voice_downloader =
          Downloader::new("onnx-community/chatterbox-multilingual-ONNX".to_owned())?;
        voice_downloader.get_path("default_voice.wav").await?
      }
      path => path.into(),
    };
    let audio_values = Value::from_array(load_audio(target_voice_path, Some(SAMPLE_RATE))?)?;
    let speech_encoder_output = speech_encoder_session.run(ort::inputs![
      "audio_values" => &audio_values
    ])?;
    let cond_emb: Array3<f32> = speech_encoder_output["audio_features"]
      .try_extract_array()?
      .to_owned()
      .into_dimensionality()?;
    let prompt_token: Array2<i64> = speech_encoder_output["audio_tokens"]
      .try_extract_array()?
      .to_owned()
      .into_dimensionality()?;
    let speaker_embeddings: Array2<f32> = speech_encoder_output["speaker_embeddings"]
      .try_extract_array()?
      .to_owned()
      .into_dimensionality()?;
    let speaker_features: Array3<f32> = speech_encoder_output["speaker_features"]
      .try_extract_array()?
      .to_owned()
      .into_dimensionality()?;

    let input_ids: Vec<i64> = tokenizer
      .encode(options.input, true)
      .map_err(|e| anyhow!(e))?
      .get_ids()
      .iter()
      .map(|&id| i64::from(id))
      .collect();
    let input_ids = Array2::from_shape_vec((1_usize, input_ids.len()), input_ids)?;
    let initial_inputs_embeds_value = embed_tokens_session
      .run(inputs! {
        "input_ids" => Value::from_array(input_ids.clone())?,
      })?
      .remove("inputs_embeds")
      .ok_or_else(|| anyhow!("embed_tokens output is missing inputs_embeds"))?;
    let (shape, data) = initial_inputs_embeds_value.try_extract_tensor::<f32>()?;
    let text_embeds = ArrayView3::from_shape(
      (shape[0] as usize, shape[1] as usize, shape[2] as usize),
      data,
    )?;
    let initial_inputs_embeds = ndarray::concatenate(Axis(1), &[cond_emb.view(), text_embeds])?;
    let batch_size = initial_inputs_embeds.shape()[0];
    let seq_len = initial_inputs_embeds.shape()[1];
    let position_ids = Array2::from_shape_vec(
      (1, seq_len),
      (0..seq_len).map(|position| position as i64).collect(),
    )?;

    let past_key_value_tensor_types: HashMap<String, TensorElementType> = language_model_session
      .inputs()
      .iter()
      .filter_map(|input| match input.dtype() {
        ValueType::Tensor { ty, .. } if input.name().starts_with("past_key_values") => {
          Some((input.name().to_owned(), *ty))
        }
        _ => None,
      })
      .collect();
    let mut past_key_values = HashMap::new();
    for layer in 0..NUM_HIDDEN_LAYERS {
      for kv in ["key", "value"] {
        let cache_key = format!("past_key_values.{layer}.{kv}");
        let cache_dtype = past_key_value_tensor_types
          .get(&cache_key)
          .copied()
          .unwrap_or(TensorElementType::Float32);
        let cache_shape = (batch_size, NUM_KV_HEADS, 0, HEAD_DIM);
        let cache_value = match cache_dtype {
          TensorElementType::Float16 => {
            Value::from_array(Array4::from_elem(cache_shape, f16::ZERO))?.into()
          }
          TensorElementType::Float32 => {
            Value::from_array(Array4::<f32>::zeros(cache_shape))?.into()
          }
          other => {
            return Err(AppError::from(anyhow!(
              "unsupported past_key_values element type: {other:?}"
            )));
          }
        };
        past_key_values.insert(cache_key, cache_value);
      }
    }

    Ok(Self {
      embed_tokens_session,
      language_model_session,
      conditional_decoder_session,
      input_ids,
      attention_mask: Array2::ones((batch_size, seq_len)),
      position_ids,
      next_inputs_embeds: Some(initial_inputs_embeds),
      prompt_token,
      speaker_embeddings,
      speaker_features,
      repetition_penalty_processor: RepetitionPenaltyLogitsProcessor::new(1.2_f32)?,
      past_key_values,
      generate_tokens: Array2::from_shape_vec((1, 1), vec![START_SPEECH_TOKEN as usize])?,
      speech_tokens: Vec::with_capacity(MAX_NEW_TOKENS),
      pending_tokens: 0,
      decoded_samples: 0,
      iteration: 0,
      batch_size,
      finished: false,
      final_audio_emitted: false,
    })
  }

  fn next_audio_chunk(&mut self) -> Result<Option<Vec<f32>>, AppError> {
    if self.finished && self.pending_tokens == 0 {
      if self.final_audio_emitted {
        return Ok(None);
      }

      self.final_audio_emitted = true;
      return Ok(Some(self.decode(true)?));
    }

    while self.pending_tokens < STREAM_TOKENS && self.iteration < MAX_NEW_TOKENS {
      let inputs_embeds = if let Some(inputs_embeds) = self.next_inputs_embeds.take() {
        inputs_embeds
      } else {
        let inputs_embeds_value = self
          .embed_tokens_session
          .run(inputs! {
            "input_ids" => Value::from_array(self.input_ids.clone())?,
          })?
          .remove("inputs_embeds")
          .ok_or_else(|| anyhow!("embed_tokens output is missing inputs_embeds"))?;
        let (shape, data) = inputs_embeds_value.try_extract_tensor::<f32>()?;
        Array3::from_shape_vec(
          (shape[0] as usize, shape[1] as usize, shape[2] as usize),
          data.to_vec(),
        )?
      };

      let mut language_model_inputs = inputs! {
        "inputs_embeds" => Value::from_array(inputs_embeds)?,
        "attention_mask" => Value::from_array(self.attention_mask.clone())?,
        "position_ids" => Value::from_array(self.position_ids.clone())?,
      };
      for (key, value) in &self.past_key_values {
        language_model_inputs.push((key.into(), value.into()));
      }

      let mut language_model_output = self.language_model_session.run(language_model_inputs)?;
      let logits = language_model_output
        .get("logits")
        .ok_or_else(|| anyhow!("language model output is missing logits"))?;
      let (logits_shape, logits_data) = logits.try_extract_tensor::<f32>()?;
      let logits_array = Array3::<f32>::from_shape_vec(
        (
          logits_shape[0] as usize,
          logits_shape[1] as usize,
          logits_shape[2] as usize,
        ),
        logits_data.to_vec(),
      )?;
      let last_token_logits = logits_array
        .index_axis(Axis(1), logits_shape[1] as usize - 1)
        .to_owned();
      let next_token_logits = self
        .repetition_penalty_processor
        .call(self.generate_tokens.row(0), &last_token_logits);
      let next_token_id = next_token_logits
        .row(0)
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .ok_or_else(|| anyhow!("language model returned empty logits"))?;
      let next_token_usize = Array2::<usize>::from_shape_vec((1, 1), vec![next_token_id])?;
      self.generate_tokens = ndarray::concatenate(
        Axis(1),
        &[self.generate_tokens.view(), next_token_usize.view()],
      )?;
      self.iteration += 1;

      if next_token_id == STOP_SPEECH_TOKEN as usize {
        self.finished = true;
        break;
      }

      self.speech_tokens.push(next_token_id as i64);
      self.pending_tokens += 1;
      self.input_ids = Array2::from_shape_vec((1, 1), vec![next_token_id as i64])?;
      let next_position = self.position_ids[[0, self.position_ids.shape()[1] - 1]] + 1;
      self.position_ids = Array2::from_elem((1, 1), next_position);
      self.attention_mask = ndarray::concatenate(
        Axis(1),
        &[
          self.attention_mask.view(),
          Array2::<i64>::ones((self.batch_size, 1)).view(),
        ],
      )?;

      for (key, value_slot) in &mut self.past_key_values {
        let present_suffix = key
          .strip_prefix("past_key_values")
          .expect("cache key should start with past_key_values");
        let present_key = format!("present{present_suffix}");
        let updated_value = language_model_output
          .remove(present_key.as_str())
          .ok_or_else(|| anyhow!("missing matching present key value tensor"))?;
        *value_slot = updated_value;
      }
    }

    if self.pending_tokens == 0 {
      return self.next_audio_chunk();
    }

    let final_chunk = self.finished || self.iteration == MAX_NEW_TOKENS;
    if final_chunk {
      self.finished = true;
    }
    self.pending_tokens = 0;
    let audio = self.decode(final_chunk)?;
    if final_chunk {
      self.final_audio_emitted = true;
    }
    Ok(Some(audio))
  }

  fn decode(&mut self, final_chunk: bool) -> Result<Vec<f32>, AppError> {
    // The exported decoder is non-autoregressive, so decode the growing prefix
    // and only return samples that were not sent by an earlier stream item.
    let silence_tokens = if final_chunk { 3 } else { 0 };
    let mut speech_tokens =
      Vec::with_capacity(self.prompt_token.len() + self.speech_tokens.len() + silence_tokens);
    speech_tokens.extend(self.prompt_token.iter().copied());
    speech_tokens.extend(self.speech_tokens.iter().copied());
    speech_tokens.extend(std::iter::repeat_n(SILENCE_TOKEN as i64, silence_tokens));
    let speech_tokens = Array2::from_shape_vec((1, speech_tokens.len()), speech_tokens)?;
    let outputs = self.conditional_decoder_session.run(inputs! {
      "speech_tokens" => Value::from_array(speech_tokens)?,
      "speaker_embeddings" => Value::from_array(self.speaker_embeddings.clone())?,
      "speaker_features" => Value::from_array(self.speaker_features.clone())?,
    })?;
    let (_, wav) = outputs["waveform"].try_extract_tensor::<f32>()?;
    if wav.len() < self.decoded_samples {
      return Err(
        anyhow!(
          "conditional decoder output shrank from {} to {} samples",
          self.decoded_samples,
          wav.len()
        )
        .into(),
      );
    }

    let audio = wav[self.decoded_samples..].to_vec();
    self.decoded_samples = wav.len();
    Ok(audio)
  }
}
