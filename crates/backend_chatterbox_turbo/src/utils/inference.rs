use std::{collections::HashMap, f32, io::Cursor, path::PathBuf};

use anyhow::anyhow;
use half::f16;
use ndarray::{Array2, Array3, Array4, ArrayView3, Axis};
use ort::{
  inputs,
  tensor::TensorElementType,
  value::{Value, ValueType},
};
use ortts_onnx::inference_session;
use ortts_shared::{AppError, Downloader, SpeechOptions};
use ortts_shared_chatterbox::{RepetitionPenaltyLogitsProcessor, load_audio};
use tokenizers::Tokenizer;

pub async fn inference(options: SpeechOptions) -> Result<Vec<u8>, AppError> {
  const MAX_NEW_TOKENS: usize = 1024;
  const SAMPLE_RATE: u32 = 24000;
  const START_SPEECH_TOKEN: u32 = 6561;
  const STOP_SPEECH_TOKEN: u32 = 6562;
  const SILENCE_TOKEN: u32 = 4299;
  const NUM_KV_HEADS: usize = 16;
  const HEAD_DIM: usize = 64;

  // Load models
  let downloader = Downloader::new("ResembleAI/chatterbox-turbo-ONNX".to_owned())?;
  let (
    speech_encoder_path,
    embed_tokens_path,
    language_model_path,
    conditional_decoder_path,
    tokenizer_path,
    // default_voice_path,
  ) = tokio::try_join!(
    downloader.get_onnx_with_data("onnx/speech_encoder.onnx"),
    downloader.get_onnx_with_data("onnx/embed_tokens.onnx"),
    downloader.get_onnx_with_data("onnx/language_model.onnx"),
    downloader.get_onnx_with_data("onnx/conditional_decoder.onnx"),
    downloader.get_tokenizer(),
    // downloader.get_path("default_voice.wav"),
  )?;

  // Create ONNX sessions
  let mut embed_tokens_session = inference_session(&embed_tokens_path)?;
  let mut speech_encoder_session = inference_session(&speech_encoder_path)?;
  let mut language_model_session = inference_session(&language_model_path)?;
  let mut conditional_decoder_session = inference_session(&conditional_decoder_path)?;
  let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

  // Prepare audio input
  let target_voice_path = match options.voice.as_str() {
    "alloy" => {
      let downloader = Downloader::new("onnx-community/chatterbox-multilingual-ONNX".to_owned())?;
      downloader.get_path("default_voice.wav").await?
    }
    path => PathBuf::from(path),
  };
  let audio_values = load_audio(target_voice_path, Some(SAMPLE_RATE))?;
  let audio_values = Array2::from_shape_vec((1_usize, audio_values.len()), audio_values)?;
  let audio_values = Value::from_array(audio_values)?;

  // Prepare text input
  let input_ids: Vec<i64> = tokenizer
    .encode(options.input, true)
    .map_err(|e| anyhow!(e))?
    .get_ids()
    .iter()
    .map(|&id| i64::from(id))
    .collect();
  let input_ids = Array2::from_shape_vec((1_usize, input_ids.len()), input_ids)?;
  let mut input_ids = Value::from_array(input_ids)?;

  // Generation loop
  // TODO: custom repetition_penalty
  let repetition_penalty = 1.2_f32;
  let repetition_penalty_processor = RepetitionPenaltyLogitsProcessor::new(repetition_penalty)?;
  // Generate tokens - for the first iteration, this would be [[START_SPEECH_TOKEN]]
  // Make it mutable so we can concatenate new tokens in each iteration
  let mut generate_tokens =
    Array2::<usize>::from_shape_vec((1, 1), vec![START_SPEECH_TOKEN as usize])?;

  let mut attention_mask = Array2::<i64>::zeros((0, 0));
  let mut position_ids = Array2::<i64>::zeros((0, 0));
  let mut batch_size = 0;
  let mut past_key_values: HashMap<String, Value> = HashMap::new();
  let mut past_key_value_names: Vec<String> = Vec::new();
  let mut past_key_value_dtypes: HashMap<String, TensorElementType> = HashMap::new();

  let mut prompt_token_global = None;
  let mut speaker_embeddings_global = None;
  let mut speaker_features_global = None;

  for i in 0..MAX_NEW_TOKENS {
    let inputs_embeds_output =
      &embed_tokens_session.run(inputs!["input_ids" => input_ids.clone()])?["inputs_embeds"];

    let (inputs_embeds_shape, inputs_embeds_data) =
      inputs_embeds_output.try_extract_tensor::<f32>()?;

    let mut inputs_embeds = Array3::from_shape_vec(
      (
        inputs_embeds_shape[0] as usize,
        inputs_embeds_shape[1] as usize,
        inputs_embeds_shape[2] as usize,
      ),
      inputs_embeds_data.to_vec(),
    )?;

    if i == 0 {
      let speech_encoder_output =
        speech_encoder_session.run(ort::inputs!["audio_values" => &audio_values])?;
      let cond_emb = speech_encoder_output.get("audio_features").unwrap();
      let prompt_token = speech_encoder_output.get("audio_tokens").unwrap();
      let speaker_embeddings = speech_encoder_output.get("speaker_embeddings").unwrap();
      let speaker_features = speech_encoder_output.get("speaker_features").unwrap();

      let (cond_emb_shape, cond_emb_data) = cond_emb.try_extract_tensor::<f32>().unwrap();
      inputs_embeds = ndarray::concatenate(
        Axis(1),
        &[
          ArrayView3::from_shape(
            (
              cond_emb_shape[0] as usize,
              cond_emb_shape[1] as usize,
              cond_emb_shape[2] as usize,
            ),
            cond_emb_data,
          )?,
          inputs_embeds.view(),
        ],
      )?;

      prompt_token_global = Some({
        let (prompt_token_shape, prompt_token_data) = prompt_token.try_extract_tensor::<i64>()?;
        Array2::<i64>::from_shape_vec(
          (
            prompt_token_shape[0] as usize,
            prompt_token_shape[1] as usize,
          ),
          prompt_token_data.to_vec(),
        )?
      });

      speaker_embeddings_global = Some({
        let (speaker_embeddings_shape, speaker_embeddings_data) =
          speaker_embeddings.try_extract_tensor::<f32>()?;
        Array2::<f32>::from_shape_vec(
          (
            speaker_embeddings_shape[0] as usize,
            speaker_embeddings_shape[1] as usize,
          ),
          speaker_embeddings_data.to_vec(),
        )?
      });

      speaker_features_global = Some({
        let (speaker_features_shape, speaker_features_data) =
          speaker_features.try_extract_tensor::<f32>()?;
        Array3::<f32>::from_shape_vec(
          (
            speaker_features_shape[0] as usize,
            speaker_features_shape[1] as usize,
            speaker_features_shape[2] as usize,
          ),
          speaker_features_data.to_vec(),
        )?
      });

      batch_size = inputs_embeds.shape()[0];
      let seq_len = inputs_embeds.shape()[1];

      let past_key_value_specs: Vec<_> = language_model_session
        .inputs()
        .iter()
        .filter_map(|input| {
          if input.name().starts_with("past_key_values") {
            let dtype = match input.dtype() {
              ValueType::Tensor { ty, .. } => *ty,
              _ => return None,
            };
            return Some((input.name(), dtype));
          }

          None
        })
        .collect();

      println!("past_key_value_specs: {:?}", past_key_value_specs);

      for (name, dtype) in past_key_value_specs {
        past_key_value_names.push(name.to_string());
        past_key_value_dtypes.insert(name.to_string(), dtype);

        println!("Initializing past key value cache for: {}", name);

        let dtype = *past_key_value_dtypes
          .get(name)
          .unwrap_or(&TensorElementType::Float32);
        match dtype {
          TensorElementType::Float16 => {
            let zeros = Array4::from_shape_vec(
              (batch_size as usize, NUM_KV_HEADS, 0 as usize, HEAD_DIM),
              Vec::<f16>::new(),
            )?;
            past_key_values.insert(
              name.to_string(),
              Value::from_array(zeros)
                .expect("should create f16 cache")
                .into(),
            );
          }
          TensorElementType::Float32 => {
            let zeros = Array4::from_shape_vec(
              (batch_size as usize, NUM_KV_HEADS, 0 as usize, HEAD_DIM),
              Vec::<f32>::new(),
            )?;
            past_key_values.insert(
              name.to_string(),
              Value::from_array(zeros)
                .expect("should create f32 cache")
                .into(),
            );
          }
          other => {
            return Err(AppError::from(anyhow!(
              "unsupported past key value dtype: {other:?}"
            )));
          }
        }
      }

      println!("Initialized past key values: {:?}", past_key_values.keys());

      attention_mask = Array2::<i64>::ones((batch_size, seq_len));
      position_ids =
        Array2::from_shape_vec((1, seq_len), (0..seq_len).map(|x| x as i64).collect())?;
    }

    let mut language_model_inputs = inputs! {
      "inputs_embeds" => Value::from_array(inputs_embeds)?,
      "attention_mask" => Value::from_array(attention_mask.clone())?,
      "position_ids" => Value::from_array(position_ids.clone())?,
    };

    for name in &past_key_value_names {
      if let Some(v) = past_key_values.get(name) {
        language_model_inputs.push((name.clone().into(), v.into()));
      }
    }

    let language_model_output = language_model_session.run(language_model_inputs)?;
    let logits = language_model_output.get("logits").unwrap();
    // let present_key_values = language_model_output
    //   .iter()
    //   .filter(|(name, _)| name.starts_with("present."))
    //   .map(|(_, v)| v)
    //   .collect::<Vec<ValueRef<'_>>>();

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
      .index_axis(Axis(1), (logits_shape[1] as usize) - 1)
      .to_owned();
    // next_token_logits = repetition_penalty_processor(generate_tokens, logits)
    let next_token_logits =
      repetition_penalty_processor.call(generate_tokens.row(0), &last_token_logits);

    // input_ids = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
    let input_ids_next = next_token_logits
      .row(0)
      .iter()
      .enumerate()
      .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
      .map(|(idx, _)| idx)
      .unwrap();

    // generate_tokens = np.concatenate((generate_tokens, input_ids), axis=-1)
    let input_ids_usize = Array2::<usize>::from_shape_vec((1, 1), vec![input_ids_next])?;
    generate_tokens =
      ndarray::concatenate(Axis(1), &[generate_tokens.view(), input_ids_usize.view()])?;

    // if (input_ids.flatten() == STOP_SPEECH_TOKEN).all():
    if input_ids_next == STOP_SPEECH_TOKEN as usize {
      break;
    }

    // save input_ids
    let input_ids_next = Array2::<i64>::from_shape_vec((1, 1), vec![input_ids_next as i64])?;
    let input_ids_next = Value::from_array(input_ids_next)?;
    input_ids = input_ids_next;

    // np.ones((batch_size, 1), dtype=np.int64)
    let batch_size_ones = Array2::<i64>::ones((batch_size as usize, 1));
    // attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
    attention_mask =
      ndarray::concatenate(Axis(1), &[attention_mask.view(), batch_size_ones.view()])?;

    // position_ids = position_ids[:, -1:] + 1
    let last_val = position_ids[[0, position_ids.shape()[1] - 1]];
    position_ids = Array2::from_elem((1, 1), last_val + 1);

    // Update cache entries in a deterministic order to avoid HashMap ordering issues.
    for name in &past_key_value_names {
      if let Some(value_slot) = past_key_values.get_mut(name) {
        println!("Updating past key value for: {}", name);
        let present_suffix = name
          .strip_prefix("past_key_values")
          .expect("cache key should start with past_key_values");
        let present_key = format!("present{present_suffix}");
        let present_value = language_model_output
          .get(present_key.as_str())
          .expect("missing matching present key value tensor");
        let dtype = *past_key_value_dtypes
          .get(name)
          .unwrap_or(&TensorElementType::Float32);
        match dtype {
          TensorElementType::Float16 => {
            let (pres_shape, pres_data) = present_value.try_extract_tensor::<f16>().unwrap();
            let pres_array = Array4::<f16>::from_shape_vec(
              (
                pres_shape[0] as usize,
                pres_shape[1] as usize,
                pres_shape[2] as usize,
                pres_shape[3] as usize,
              ),
              pres_data.to_vec(),
            )?;
            *value_slot = Value::from_array(pres_array)?.into();
          }
          TensorElementType::Float32 => {
            let (pres_shape, pres_data) = present_value.try_extract_tensor::<f32>().unwrap();
            let pres_array = Array4::<f32>::from_shape_vec(
              (
                pres_shape[0] as usize,
                pres_shape[1] as usize,
                pres_shape[2] as usize,
                pres_shape[3] as usize,
              ),
              pres_data.to_vec(),
            )?;
            *value_slot = Value::from_array(pres_array)?.into();
          }
          other => {
            return Err(AppError::from(anyhow!(
              "unsupported present key value element type: {other:?}"
            )));
          }
        }
      }
    }
  }

  // speech_tokens = generate_tokens[:, 1:-1]
  let generate_tokens_shape = generate_tokens.shape();
  let speech_tokens = generate_tokens
    .slice(ndarray::s![.., 1..(generate_tokens_shape[1] - 1)])
    .to_owned();

  // silence_tokens = np.full((speech_tokens.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
  let silence_tokens = Array2::<i64>::from_elem((1, 1), SILENCE_TOKEN as i64);

  // speech_tokens = np.concatenate([prompt_token, speech_tokens, silence_token], axis=1)
  let speech_tokens_with_prompt = ndarray::concatenate(
    Axis(1),
    &[
      prompt_token_global.unwrap().view(),
      speech_tokens.mapv(|x| x as i64).view(),
      silence_tokens.view(),
    ],
  )?;
  tracing::debug!(
    "speech_tokens_with_prompt shape: {:?}",
    speech_tokens_with_prompt.shape()
  );

  let speech_tokens_value = Value::from_array(speech_tokens_with_prompt)?;
  let speaker_embeddings_value = Value::from_array(speaker_embeddings_global.unwrap())?;
  let speaker_features_value = Value::from_array(speaker_features_global.unwrap())?;
  let cond_decoder_output = conditional_decoder_session.run(inputs! {
      "speech_tokens" => speech_tokens_value,
      "speaker_embeddings" => speaker_embeddings_value,
      "speaker_features" => speaker_features_value,
  })?;

  let wav = cond_decoder_output.get("waveform").unwrap();
  let (wav_shape, wav_data) = wav.try_extract_tensor::<f32>()?;

  tracing::debug!("wav shape: {:?}, length: {}", wav_shape, wav_data.len());
  // wav = np.squeeze(wav, axis=0)
  // let wav_squeezed = if wav_shape.len() > 1 && wav_shape[0] == 1 {
  //   wav_data.to_vec()
  // } else {
  //   wav_data.to_vec()
  // };
  let wav_squeezed = wav_data.to_vec();

  tracing::debug!("Generated audio with {} samples", wav_squeezed.len());

  let spec = hound::WavSpec {
    channels: 1,
    sample_rate: SAMPLE_RATE,
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
