use std::io::Cursor;

use ort::value::{Value, ValueRef};
use ortts_shared::{AppError, Downloader};
use tokenizers::Tokenizer;
use tracing::info;

use super::{RepetitionPenaltyLogitsProcessor, create_session, load_audio};

pub async fn inference() -> Result<Vec<u8>, AppError> {
  const MAX_NEW_TOKENS: usize = 256;
  const S3GEN_SR: u32 = 24000;
  const START_SPEECH_TOKEN: u32 = 6561;
  const STOP_SPEECH_TOKEN: u32 = 6562;
  const NUM_HIDDEN_LAYERS: i64 = 30;
  const NUM_KEY_VALUE_HEADS: i64 = 16;
  const HEAD_DIM: i64 = 64;

  let repetition_penalty = 1.2_f32;
  let processor = RepetitionPenaltyLogitsProcessor::new(repetition_penalty)?;

  // Generate tokens - for the first iteration, this would be [[START_SPEECH_TOKEN]]
  // Make it mutable so we can concatenate new tokens in each iteration
  let mut generate_tokens =
    ndarray::Array2::<usize>::from_shape_vec((1, 1), vec![START_SPEECH_TOKEN as usize])?;

  let downloader = Downloader::new();

  let speech_encoder_path = downloader
    .get_path(
      "onnx-community/chatterbox-multilingual-ONNX",
      "onnx/speech_encoder.onnx",
    )
    .await?;
  let embed_tokens_path = downloader
    .get_path(
      "onnx-community/chatterbox-multilingual-ONNX",
      "onnx/embed_tokens.onnx",
    )
    .await?;
  let llama_with_path_path = downloader
    .get_path(
      "onnx-community/chatterbox-multilingual-ONNX",
      "onnx/language_model.onnx",
    )
    .await?;
  let conditional_decoder_path = downloader
    .get_path(
      "onnx-community/chatterbox-multilingual-ONNX",
      "onnx/conditional_decoder.onnx",
    )
    .await?;

  // let tokenizer_config_path = downloader
  //   .get_path(
  //     "onnx-community/chatterbox-multilingual-ONNX",
  //     "tokenizer.json",
  //   )
  //   .await?;

  // assert!(speech_encoder_path.exists());
  // assert!(embed_tokens_path.exists());
  // assert!(llama_with_path_path.exists());
  // assert!(conditional_decoder_path.exists());

  // assert!(tokenizer_config_path.exists());

  let mut embed_tokens_session = create_session(embed_tokens_path)?;
  let mut speech_encoder_session = create_session(speech_encoder_path)?;
  let mut llama_with_past_session = create_session(llama_with_path_path)?;
  let mut conditional_decoder_session = create_session(conditional_decoder_path)?;

  let tokenizer =
    Tokenizer::from_pretrained("onnx-community/chatterbox-multilingual-ONNX", None).unwrap();

  let target_voice_path = downloader
    .get_path(
      "onnx-community/chatterbox-multilingual-ONNX",
      "default_voice.wav",
    )
    .await?;

  // Convert to ort Value with shape [1, audio_length]
  let audio_value_data = load_audio(target_voice_path)?;
  info!("audio length: {}", audio_value_data.len());
  let audio_value_array = ndarray::Array2::<f32>::from_shape_vec(
    (1_usize, audio_value_data.len()),
    audio_value_data.clone(),
  )?;
  let audio_value = Value::from_array(audio_value_array)?;
  info!(
    "audio shape: {:?}, dtype: {:?}",
    audio_value.shape(),
    audio_value.data_type()
  );

  let text =
    "[en]Hello, this is a test message for multilingual text-to-speech synthesis.".to_string();

  // input_ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
  let tokenized_input = tokenizer.encode(text, true).unwrap();
  info!(
    "{:?}, shape: {:?}",
    tokenized_input.get_tokens(),
    tokenized_input.get_tokens().len()
  );
  info!(
    "{:?}, shape: {:?}",
    tokenized_input.get_ids(),
    tokenized_input.get_ids().len()
  );

  let input_ids_data: Vec<i64> = tokenized_input
    .get_ids()
    .iter()
    .map(|&id| id as i64)
    .collect();
  let input_ids_shape = vec![1_usize, input_ids_data.len()];
  let input_ids_array = ndarray::Array2::<i64>::from_shape_vec(
    (input_ids_shape[0], input_ids_shape[1]),
    input_ids_data.clone(),
  )?;
  let input_ids_value = Value::from_array(input_ids_array)?;

  // position_ids = np.where(
  //   input_ids >= START_SPEECH_TOKEN,
  //   0,
  //   np.arange(input_ids.shape[1])[np.newaxis, :] - 1
  // )
  let position_ids_data: Vec<i64> = input_ids_data
    .iter()
    .enumerate()
    .map(|(i, &token_id)| {
      if token_id >= START_SPEECH_TOKEN as i64 {
        0
      } else {
        i as i64 - 1
      }
    })
    .collect();
  let position_ids_array = ndarray::Array2::<i64>::from_shape_vec(
    (1_usize, input_ids_data.len()),
    position_ids_data.clone(),
  )?;
  let position_ids_value = Value::from_array(position_ids_array).unwrap();

  // exaggeration=0.5
  let exaggeration = 0.5_f32;
  // np.array([exaggeration], dtype=np.float32)
  let exaggeration_value =
    Value::from_array(ndarray::Array1::from_shape_vec(1_usize, vec![exaggeration]).unwrap())?;

  let mut attention_mask_array = ndarray::Array2::<i64>::zeros((0, 0));
  let mut batch_size = 0;
  // KV Cache
  let mut past_key_values: std::collections::HashMap<String, Value> =
    std::collections::HashMap::new();

  // // NOTICE: Reuseable during generation loop
  let mut ort_embed_tokens_input_ids = input_ids_value.clone();
  let mut ort_embed_tokens_position_ids = position_ids_value.clone();
  let ort_embed_tokens_exaggeration = exaggeration_value.clone();

  // // TODO: Speech conditional decoder model required
  let mut prompt_token_array: Option<ndarray::Array2<i64>> = None;
  let mut speaker_embeddings_array: Option<ndarray::Array2<f32>> = None;
  let mut speaker_features_array: Option<ndarray::Array3<f32>> = None;

  for i in 0..MAX_NEW_TOKENS {
    // inputs_embeds = embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]
    let mut ort_input_embeds_output = embed_tokens_session.run(ort::inputs![
    "input_ids" => &ort_embed_tokens_input_ids,
    "position_ids" => &ort_embed_tokens_position_ids,
    "exaggeration" => &ort_embed_tokens_exaggeration,
    ])?;

    let mut inputs_embeds_value: Value = ort_input_embeds_output.remove("inputs_embeds").unwrap();
    info!("inputs_embeds_value: {:?}", inputs_embeds_value.shape());

    if i == 0 {
      // cond_emb, prompt_token, speaker_embeddings, speaker_features = speech_encoder_session.run(None, ort_speech_encoder_input)
      let ort_speech_encoder_output = speech_encoder_session
        .run(ort::inputs!["audio_values" => &audio_value])
        .unwrap();
      info!(
        "ort_speech_encoder_output keys: {:?}",
        ort_speech_encoder_output
      );
      let cond_emb = ort_speech_encoder_output.get("audio_features").unwrap();
      let prompt_token = ort_speech_encoder_output.get("audio_tokens").unwrap();
      let ref_x_vector = ort_speech_encoder_output.get("speaker_embeddings").unwrap();
      let prompt_feat = ort_speech_encoder_output.get("speaker_features").unwrap();

      info!("cond_emb: {:?}", cond_emb.shape());
      info!("prompt_token: {:?}", prompt_token.shape());
      info!("ref_x_vector: {:?}", ref_x_vector.shape());
      info!("prompt_feat: {:?}", prompt_feat.shape());

      prompt_token_array = Some({
        let (prompt_token_shape, prompt_token_data) =
          prompt_token.try_extract_tensor::<i64>().unwrap();
        ndarray::Array2::<i64>::from_shape_vec(
          (
            prompt_token_shape[0] as usize,
            prompt_token_shape[1] as usize,
          ),
          prompt_token_data.to_vec(),
        )
        .unwrap()
      });

      speaker_embeddings_array = Some({
        let (speaker_embeddings_shape, speaker_embeddings_data) =
          ref_x_vector.try_extract_tensor::<f32>().unwrap();
        ndarray::Array2::<f32>::from_shape_vec(
          (
            speaker_embeddings_shape[0] as usize,
            speaker_embeddings_shape[1] as usize,
          ),
          speaker_embeddings_data.to_vec(),
        )
        .unwrap()
      });

      speaker_features_array = Some({
        let (speaker_features_shape, speaker_features_data) =
          prompt_feat.try_extract_tensor::<f32>().unwrap();
        ndarray::Array3::<f32>::from_shape_vec(
          (
            speaker_features_shape[0] as usize,
            speaker_features_shape[1] as usize,
            speaker_features_shape[2] as usize,
          ),
          speaker_features_data.to_vec(),
        )
        .unwrap()
      });

      // inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
      {
        let (cond_emb_shape, cond_emb_data) = cond_emb.try_extract_tensor::<f32>().unwrap();
        let (inputs_embeds_shape, inputs_embeds_data) =
          inputs_embeds_value.try_extract_tensor::<f32>().unwrap();

        use ndarray::ArrayView3;
        let inputs_embeds_concatenated = ndarray::concatenate(
          ndarray::Axis(1),
          &[
            ArrayView3::from_shape(
              (
                cond_emb_shape[0] as usize,
                cond_emb_shape[1] as usize,
                cond_emb_shape[2] as usize,
              ),
              cond_emb_data,
            )
            .unwrap(),
            ArrayView3::from_shape(
              (
                inputs_embeds_shape[0] as usize,
                inputs_embeds_shape[1] as usize,
                inputs_embeds_shape[2] as usize,
              ),
              inputs_embeds_data,
            )
            .unwrap(),
          ],
        )
        .unwrap();

        let inputs_embeds_concatenated_shape: Vec<usize> =
          inputs_embeds_concatenated.shape().to_vec();
        let inputs_embeds_concatenated_data: Vec<f32> =
          inputs_embeds_concatenated.iter().copied().collect();
        inputs_embeds_value = Value::from_array((
          inputs_embeds_concatenated_shape.as_slice(),
          inputs_embeds_concatenated_data,
        ))
        .unwrap()
        .into();
      }

      // batch_size, seq_len, _ = inputs_embeds.shape
      batch_size = inputs_embeds_value.shape()[0];
      let seq_len = inputs_embeds_value.shape()[1];

      // { ... f"past_key_values.{layer}.{kv}": np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32) ... }
      for layer in 0..NUM_HIDDEN_LAYERS {
        for kv in ["key", "value"] {
          let cache_key = format!("past_key_values.{}.{}", layer, kv);
          let cache = ndarray::Array4::<f32>::zeros((
            batch_size as usize,
            NUM_KEY_VALUE_HEADS as usize,
            0,
            HEAD_DIM as usize,
          ));
          let cache_value = Value::from_array(cache).unwrap().into();
          past_key_values.insert(cache_key, cache_value);
        }
      }

      // attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
      attention_mask_array = ndarray::Array2::<i64>::ones((batch_size as usize, seq_len as usize));
    }

    let attention_mask_value = Value::from_array(attention_mask_array.clone()).unwrap();
    let mut ort_llama_with_past_inputs = ort::inputs![
      "inputs_embeds" => inputs_embeds_value,
      "attention_mask" => attention_mask_value,
    ];
    for (key, value) in &past_key_values {
      ort_llama_with_past_inputs.push((key.into(), value.into()));
    }

    // logits, *present_key_values = llama_with_past_session.run(...)
    let ort_llama_with_past_output = llama_with_past_session
      .run(ort_llama_with_past_inputs)
      .unwrap();
    let logits = ort_llama_with_past_output.get("logits").unwrap();
    let present_key_values = ort_llama_with_past_output
      .iter()
      .into_iter()
      .filter(|(name, _)| name.starts_with("present."))
      .map(|(_, v)| v)
      .collect::<Vec<ValueRef<'_>>>();

    info!("logits: {:?}", logits.shape());
    info!("present_key_values lengths: {}", present_key_values.len());

    let (logits_shape, logits_data) = logits.try_extract_tensor::<f32>().unwrap();
    let logits_array = ndarray::Array3::<f32>::from_shape_vec(
      (
        logits_shape[0] as usize,
        logits_shape[1] as usize,
        logits_shape[2] as usize,
      ),
      logits_data.to_vec(),
    )
    .unwrap();

    // logits = logits[:, -1, :]
    let last_token_logits = logits_array
      .index_axis(ndarray::Axis(1), (logits_shape[1] as usize) - 1)
      .to_owned();
    info!("logits[:, -1, :]: {:?}", last_token_logits.shape(),);
    // next_token_logits = repetition_penalty_processor(generate_tokens, logits)
    let next_token_logits = processor.call(generate_tokens.row(0), &last_token_logits);
    info!("next_token_logits: {:?}", next_token_logits.shape(),);

    // next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
    let next_token_id = next_token_logits
      .row(0)
      .iter()
      .enumerate()
      .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
      .map(|(idx, _)| idx)
      .unwrap();

    info!("next_token_id: {}", next_token_id);

    // generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)
    let next_token_usize =
      ndarray::Array2::<usize>::from_shape_vec((1, 1), vec![next_token_id as usize]).unwrap();
    generate_tokens = ndarray::concatenate(
      ndarray::Axis(1),
      &[generate_tokens.view(), next_token_usize.view()],
    )
    .unwrap();

    // if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
    info!("generate_tokens: {:?}", generate_tokens);
    if next_token_id == STOP_SPEECH_TOKEN as usize {
      break;
    }

    // next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
    let next_token_i64 =
      ndarray::Array2::<i64>::from_shape_vec((1, 1), vec![next_token_id as i64]).unwrap();
    ort_embed_tokens_input_ids = Value::from_array(next_token_i64.clone()).unwrap();

    // position_ids = np.full(
    //   (input_ids.shape[0], 1),
    //   i + 1,
    //   dtype=np.int64,
    // )
    let position_ids_next =
      ndarray::Array2::<i64>::from_elem((input_ids_shape[0], 1), (i + 1) as i64);
    ort_embed_tokens_position_ids = Value::from_array(position_ids_next).unwrap();

    // np.ones((batch_size, 1), dtype=np.int64)
    let batch_size_ones = ndarray::Array2::<i64>::ones((batch_size as usize, 1));
    // attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
    attention_mask_array = ndarray::concatenate(
      ndarray::Axis(1),
      &[attention_mask_array.view(), batch_size_ones.view()],
    )
    .unwrap();

    // for j, key in enumerate(past_key_values):
    //     past_key_values[key] = present_key_values[j]
    // NOTICE: HashMap iteration order loves to shuffle things around; if we zip by index we end up
    // assigning layer N's cache to layer M and the model goes off into la-la land. Always grab the
    // matching present.* tensor by name so each past_key_values slot stays lined up with the layer
    // that produced it.
    for (key, value_slot) in past_key_values.iter_mut() {
      let present_suffix = key
        .strip_prefix("past_key_values")
        .expect("cache key should start with past_key_values");
      let present_key = format!("present{}", present_suffix);
      let present_value = ort_llama_with_past_output
        .get(present_key.as_str())
        .expect("missing matching present key value tensor");
      let (pres_shape, pres_data) = present_value.try_extract_tensor::<f32>().unwrap();
      let pres_array = ndarray::Array4::<f32>::from_shape_vec(
        (
          pres_shape[0] as usize,
          pres_shape[1] as usize,
          pres_shape[2] as usize,
          pres_shape[3] as usize,
        ),
        pres_data.to_vec(),
      )
      .unwrap();
      *value_slot = Value::from_array(pres_array).unwrap().into();
    }
  }

  info!(
    "generate_tokens shape: {:?}, value: {:?}",
    generate_tokens.shape(),
    generate_tokens
  );

  // speech_tokens = generate_tokens[:, 1:-1]
  let generate_tokens_shape = generate_tokens.shape();
  let speech_tokens = generate_tokens
    .slice(ndarray::s![.., 1..(generate_tokens_shape[1] - 1)])
    .to_owned();
  info!("speech_tokens shape: {:?}", speech_tokens.shape());
  info!("speech_tokens: {:?}", speech_tokens);

  // speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
  let speech_tokens_with_prompt = ndarray::concatenate(
    ndarray::Axis(1),
    &[
      prompt_token_array.unwrap().view(),
      speech_tokens.mapv(|x| x as i64).view(),
    ],
  )
  .unwrap();
  info!(
    "speech_tokens_with_prompt shape: {:?}",
    speech_tokens_with_prompt.shape()
  );

  let speech_tokens_value = Value::from_array(speech_tokens_with_prompt).unwrap();
  let speaker_embeddings_value = Value::from_array(speaker_embeddings_array.unwrap()).unwrap();
  let speaker_features_value = Value::from_array(speaker_features_array.unwrap()).unwrap();

  // wav = cond_decoder_session.run(None, cond_incoder_input)[0]
  let cond_decoder_output = conditional_decoder_session
    .run(ort::inputs![
      "speech_tokens" => speech_tokens_value,
      "speaker_embeddings" => speaker_embeddings_value,
      "speaker_features" => speaker_features_value,
    ])
    .unwrap();

  let wav = cond_decoder_output.get("waveform").unwrap();
  let (wav_shape, wav_data) = wav.try_extract_tensor::<f32>().unwrap();

  info!("wav shape: {:?}, length: {}", wav_shape, wav_data.len());
  // wav = np.squeeze(wav, axis=0)
  let wav_squeezed = if wav_shape.len() > 1 && wav_shape[0] == 1 {
    wav_data.to_vec()
  } else {
    wav_data.to_vec()
  };

  info!("Generated audio with {} samples", wav_squeezed.len());

  let spec = hound::WavSpec {
    channels: 1,
    sample_rate: S3GEN_SR,
    bits_per_sample: 32,
    sample_format: hound::SampleFormat::Float,
  };

  let mut buffer = Cursor::new(Vec::<u8>::new());
  let mut writer = hound::WavWriter::new(&mut buffer, spec)?;
  for sample in wav_squeezed.iter() {
    writer.write_sample(*sample)?;
  }
  writer.finalize()?;

  Ok(buffer.into_inner())
}
