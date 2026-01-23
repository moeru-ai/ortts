use std::{io::Cursor, path::PathBuf};

use anyhow::anyhow;
use half::f16;
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use ort::{
  inputs,
  tensor::TensorElementType,
  value::{Value, ValueRef, ValueType},
};
use ortts_onnx::inference_session;
use ortts_shared::{AppError, Downloader, SpeechOptions};
use ortts_shared_chatterbox::{RepetitionPenaltyLogitsProcessor, load_audio};
use tokenizers::Tokenizer;

use crate::utils::{LanguagePreparer, validate_language_id};

pub async fn inference(options: SpeechOptions) -> Result<Vec<u8>, AppError> {
  const MAX_NEW_TOKENS: usize = 256;
  const S3GEN_SR: u32 = 24000;
  const START_SPEECH_TOKEN: u32 = 6561;
  const STOP_SPEECH_TOKEN: u32 = 6562;
  const NUM_HIDDEN_LAYERS: i64 = 30;
  const NUM_KEY_VALUE_HEADS: i64 = 16;
  const HEAD_DIM: i64 = 64;

  // Validate language_id
  let language_id = validate_language_id(&options.model)?;

  // Load model
  let downloader = Downloader::new("onnx-community/chatterbox-multilingual-ONNX".to_owned())?;
  let (
    speech_encoder_path,
    embed_tokens_path,
    language_model_path,
    conditional_decoder_path,
    tokenizer_path,
    default_voice_path,
  ) = tokio::try_join!(
    downloader.get_onnx_with_data("onnx/speech_encoder.onnx"),
    downloader.get_onnx_with_data("onnx/embed_tokens.onnx"),
    downloader.get_onnx_with_data("onnx/language_model_q4f16.onnx"),
    downloader.get_onnx_with_data("onnx/conditional_decoder.onnx"),
    downloader.get_tokenizer(),
    downloader.get_path("default_voice.wav"),
  )?;

  // Start inference sessions
  let mut embed_tokens_session = inference_session(&embed_tokens_path)?;
  let mut speech_encoder_session = inference_session(&speech_encoder_path)?;
  let mut llama_with_past_session = inference_session(&language_model_path)?;
  let mut conditional_decoder_session = inference_session(&conditional_decoder_path)?;
  let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

  let target_voice_path = match options.voice.as_str() {
    "alloy" => default_voice_path,
    path => PathBuf::from(path),
  };

  // NOTICE: in python, librosa.load(..., sr=S3GEN_SR) resamples to 24000 Hz,
  // as the s3gen model requires 24kHz audio input, we will resample any audio
  // file into this target sample rate.
  let audio_values = Value::from_array(load_audio(target_voice_path, Some(S3GEN_SR))?)?;

  // Prepare input
  let language_preparer = LanguagePreparer::new().await?;
  let text = language_preparer.prepare(options.input, &language_id);
  let input_ids: Vec<i64> = tokenizer
    .encode(text, true)
    .map_err(|e| anyhow!(e))?
    .get_ids()
    .iter()
    .map(|&id| i64::from(id))
    .collect();
  let mut input_ids = Array2::from_shape_vec((1_usize, input_ids.len()), input_ids)?;

  let position_ids: Vec<i64> = input_ids
    .iter()
    .enumerate()
    .map(|(i, &token_id)| {
      if token_id >= i64::from(START_SPEECH_TOKEN) {
        0
      } else {
        i as i64 - 1
      }
    })
    .collect();
  let mut position_ids = Array2::from_shape_vec((1_usize, position_ids.len()), position_ids)?;

  // TODO: custom exaggeration
  let exaggeration = 0.5_f32;
  let exaggeration = Value::from_array(Array1::from_shape_vec(1_usize, vec![exaggeration])?)?;

  // TODO: custom repetition_penalty
  let repetition_penalty = 1.2_f32;
  let repetition_penalty_processor = RepetitionPenaltyLogitsProcessor::new(repetition_penalty)?;

  // Generate tokens - for the first iteration, this would be [[START_SPEECH_TOKEN]]
  // Make it mutable so we can concatenate new tokens in each iteration
  let mut generate_tokens = Array2::<usize>::from_elem((1, 1), START_SPEECH_TOKEN as usize);

  let past_key_value_tensor_types: std::collections::HashMap<String, TensorElementType> =
    llama_with_past_session
      .inputs()
      .iter()
      .filter_map(|input| match &input.dtype() {
        ValueType::Tensor { ty, .. } if input.name().starts_with("past_key_values") => {
          Some((input.name().to_owned(), *ty))
        }
        _ => None,
      })
      .collect();

  let mut attention_mask = Array2::<i64>::zeros((0, 0));
  let mut batch_size = 0;

  // KV Cache
  let mut past_key_values: std::collections::HashMap<String, Value> =
    std::collections::HashMap::new();

  // // TODO: Speech conditional decoder model required
  let mut prompt_token_saved: Option<Array2<i64>> = None;
  let mut speaker_embeddings_saved: Option<Array2<f32>> = None;
  let mut speaker_features_saved: Option<Array3<f32>> = None;

  for i in 0..MAX_NEW_TOKENS {
    // inputs_embeds = embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]
    let inputs_embeds_value = embed_tokens_session
      .run(inputs! {
        "input_ids" => Value::from_array(input_ids)?,
        "position_ids" => Value::from_array(position_ids)?,
        "exaggeration" => exaggeration.view(),
      })?
      .remove("inputs_embeds")
      .unwrap();

    let (inputs_embeds_shape, inputs_embeds_data) =
      inputs_embeds_value.try_extract_tensor::<f32>().unwrap();

    let mut inputs_embeds = Array3::from_shape_vec(
      (
        inputs_embeds_shape[0] as usize,
        inputs_embeds_shape[1] as usize,
        inputs_embeds_shape[2] as usize,
      ),
      inputs_embeds_data.to_vec(),
    )?;

    tracing::debug!("inputs_embeds_value: {:?}", inputs_embeds.shape());

    if i == 0 {
      // cond_emb, prompt_token, speaker_embeddings, speaker_features = speech_encoder_session.run(None, ort_speech_encoder_input)
      let speech_encoder_output =
        speech_encoder_session.run(ort::inputs!["audio_values" => &audio_values])?;
      tracing::debug!("speech_encoder_output keys: {:?}", speech_encoder_output);
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

      tracing::debug!("cond_emb: {:?}", cond_emb.shape());
      tracing::debug!("prompt_token: {:?}", prompt_token.shape());
      tracing::debug!("speaker_embeddings: {:?}", speaker_embeddings.shape());
      tracing::debug!("speaker_features: {:?}", speaker_features.shape());

      prompt_token_saved = Some(prompt_token);
      speaker_embeddings_saved = Some(speaker_embeddings);
      speaker_features_saved = Some(speaker_features);

      // inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
      inputs_embeds =
        ndarray::concatenate(Axis(1), &[cond_emb.view(), inputs_embeds.view()])?.into_owned();

      // batch_size, seq_len, _ = inputs_embeds.shape
      batch_size = inputs_embeds.shape()[0];
      let seq_len = inputs_embeds.shape()[1];

      // { ... f"past_key_values.{layer}.{kv}": np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32) ... }
      for layer in 0..NUM_HIDDEN_LAYERS {
        for kv in ["key", "value"] {
          let cache_key = format!("past_key_values.{layer}.{kv}");
          let cache_dtype = past_key_value_tensor_types
            .get(&cache_key)
            .copied()
            .unwrap_or(TensorElementType::Float32);

          let cache_shape = (
            batch_size as usize,
            NUM_KEY_VALUE_HEADS as usize,
            0,
            HEAD_DIM as usize,
          );
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

      // attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
      attention_mask = Array2::<i64>::ones((batch_size as usize, seq_len as usize));
    }

    // let attention_mask_value = Value::from_array(attention_mask.clone()).unwrap();
    let mut llama_with_past_inputs: Vec<(
      std::borrow::Cow<'_, str>,
      ort::session::SessionInputValue<'_>,
    )> = ort::inputs![
      "inputs_embeds" => Value::from_array(inputs_embeds)?,
      "attention_mask" => Value::from_array(attention_mask.clone())?,
    ];
    for (key, value) in &past_key_values {
      llama_with_past_inputs.push((key.into(), value.into()));
    }

    // logits, *present_key_values = llama_with_past_session.run(...)
    let mut llama_with_past_output = llama_with_past_session.run(llama_with_past_inputs)?;
    let logits = llama_with_past_output.get("logits").unwrap();
    let present_key_values = llama_with_past_output
      .iter()
      .filter(|(name, _)| name.starts_with("present."))
      .map(|(_, v)| v)
      .collect::<Vec<ValueRef<'_>>>();

    tracing::debug!("logits: {:?}", logits.shape());
    tracing::debug!("present_key_values lengths: {}", present_key_values.len());

    let (logits_shape, logits_data) = logits.try_extract_tensor::<f32>()?;
    let logits_array = Array3::<f32>::from_shape_vec(
      (
        logits_shape[0] as usize,
        logits_shape[1] as usize,
        logits_shape[2] as usize,
      ),
      logits_data.to_vec(),
    )?;

    // logits = logits[:, -1, :]
    let last_token_logits = logits_array
      .index_axis(Axis(1), (logits_shape[1] as usize) - 1)
      .to_owned();
    tracing::debug!("logits[:, -1, :]: {:?}", last_token_logits.shape());
    // next_token_logits = repetition_penalty_processor(generate_tokens, logits)
    let next_token_logits =
      repetition_penalty_processor.call(generate_tokens.row(0), &last_token_logits);
    tracing::debug!("next_token_logits: {:?}", next_token_logits.shape());

    // next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
    let next_token_id = next_token_logits
      .row(0)
      .iter()
      .enumerate()
      .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
      .map(|(idx, _)| idx)
      .unwrap();

    tracing::debug!("next_token_id: {next_token_id}");

    // generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)
    let next_token_usize = Array2::<usize>::from_shape_vec((1, 1), vec![next_token_id])?;
    generate_tokens =
      ndarray::concatenate(Axis(1), &[generate_tokens.view(), next_token_usize.view()])?;

    // if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
    tracing::debug!("generate_tokens: {:?}", generate_tokens);
    if next_token_id == STOP_SPEECH_TOKEN as usize {
      break;
    }

    // next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
    input_ids = Array2::<i64>::from_shape_vec((1, 1), vec![next_token_id as i64])?;

    // position_ids = np.full(
    //   (input_ids.shape[0], 1),
    //   i + 1,
    //   dtype=np.int64,
    // )
    position_ids = Array2::<i64>::from_elem((1, 1), (i + 1) as i64);

    // np.ones((batch_size, 1), dtype=np.int64)
    let batch_size_ones = Array2::<i64>::ones((batch_size as usize, 1));
    // attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
    attention_mask =
      ndarray::concatenate(Axis(1), &[attention_mask.view(), batch_size_ones.view()])?;

    // for j, key in enumerate(past_key_values):
    //     past_key_values[key] = present_key_values[j]
    // NOTICE: HashMap iteration order loves to shuffle things around; if we zip by index we end up
    // assigning layer N's cache to layer M and the model goes off into la-la land. Always grab the
    // matching present.* tensor by name so each past_key_values slot stays lined up with the layer
    // that produced it.
    for (key, value_slot) in &mut past_key_values {
      let present_suffix = key
        .strip_prefix("past_key_values")
        .expect("cache key should start with past_key_values");
      let present_key = format!("present{present_suffix}");

      let updated_value = llama_with_past_output
        .remove(present_key.as_str())
        .expect("missing matching present key value tensor");

      *value_slot = updated_value;
    }
  }

  tracing::debug!(
    "generate_tokens shape: {:?}, value: {:?}",
    generate_tokens.shape(),
    generate_tokens
  );

  // speech_tokens = generate_tokens[:, 1:-1]
  let generate_tokens_shape = generate_tokens.shape();
  let speech_tokens = generate_tokens
    .slice(ndarray::s![.., 1..(generate_tokens_shape[1] - 1)])
    .to_owned();
  tracing::debug!("speech_tokens shape: {:?}", speech_tokens.shape());
  tracing::debug!("speech_tokens: {:?}", speech_tokens);

  // speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
  let speech_tokens_with_prompt = ndarray::concatenate(
    Axis(1),
    &[
      prompt_token_saved.unwrap().view(),
      speech_tokens.mapv(|x| x as i64).view(),
    ],
  )?;
  tracing::debug!(
    "speech_tokens_with_prompt shape: {:?}",
    speech_tokens_with_prompt.shape()
  );

  let speech_tokens_value = Value::from_array(speech_tokens_with_prompt)?;
  let speaker_embeddings_value = Value::from_array(speaker_embeddings_saved.unwrap())?;
  let speaker_features_value = Value::from_array(speaker_features_saved.unwrap())?;

  // wav = cond_decoder_session.run(None, cond_incoder_input)[0]
  let wav = &conditional_decoder_session.run(ort::inputs![
    "speech_tokens" => speech_tokens_value,
    "speaker_embeddings" => speaker_embeddings_value,
    "speaker_features" => speaker_features_value,
  ])?["waveform"];

  let (_, wav) = wav.try_extract_tensor::<f32>()?;

  tracing::debug!("wav length: {}", wav.len());
  // wav = np.squeeze(wav, axis=0)
  let wav = wav.to_vec();

  tracing::debug!("Generated audio with {} samples", wav.len());

  let spec = hound::WavSpec {
    channels: 1,
    sample_rate: S3GEN_SR,
    bits_per_sample: 32,
    sample_format: hound::SampleFormat::Float,
  };

  let mut buffer = Cursor::new(Vec::<u8>::new());
  let mut writer = hound::WavWriter::new(&mut buffer, spec)?;
  for sample in wav {
    writer.write_sample(sample)?;
  }
  writer.finalize()?;

  Ok(buffer.into_inner())
}
