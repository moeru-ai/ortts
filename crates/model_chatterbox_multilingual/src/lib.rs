use ortts_shared::AppError;

use ort::session::Session;

mod utils;

pub fn create_session(model_path: &str) -> Result<Session, AppError> {
  use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
  };
  use ort::session::builder::GraphOptimizationLevel;

  let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_parallel_execution(true)?
    .with_execution_providers([
      CUDAExecutionProvider::default().with_device_id(0).build(),
      CoreMLExecutionProvider::default().build(),
      DirectMLExecutionProvider::default()
        .with_device_id(0)
        .build(),
      CPUExecutionProvider::default().build(),
    ])?
    .commit_from_file(model_path)?;

  Ok(session)
}

pub fn load_audio(path: &str) -> Result<Vec<f32>, AppError> {
  use std::fs::File;
  use symphonia::core::io::MediaSourceStream;
  use symphonia::core::probe::Hint;
  use symphonia::default::get_probe;

  let file = File::open(path)?;
  let mss = MediaSourceStream::new(Box::new(file), Default::default());

  let mut hint = Hint::new();
  hint.with_extension("wav");

  let meta_opts = Default::default();
  let fmt_opts = Default::default();

  let probed = get_probe().format(&hint, mss, &meta_opts, &fmt_opts)?;
  let mut format = probed.format;
  let track = format
    .tracks()
    .iter()
    .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
    .ok_or_else(|| anyhow::anyhow!("No supported audio tracks"))?;

  let track_id = track.id;
  let mut decoder =
    symphonia::default::get_codecs().make(&track.codec_params, &Default::default())?;

  let mut audio_buf = None;
  let mut samples = Vec::new();

  loop {
    let packet = match format.next_packet() {
      Ok(packet) => packet,
      Err(_) => break,
    };

    if packet.track_id() != track_id {
      continue;
    }

    match decoder.decode(&packet) {
      Ok(decoded) => {
        if audio_buf.is_none() {
          let spec = *decoded.spec();
          let duration = decoded.capacity() as u64;
          audio_buf = Some(symphonia::core::audio::SampleBuffer::<f32>::new(
            duration, spec,
          ));
        }

        if let Some(ref mut buf) = audio_buf {
          buf.copy_interleaved_ref(decoded);
          samples.extend_from_slice(buf.samples());
        }
      }
      Err(_) => break,
    }
  }

  // Resample if needed (simplified - assumes same sample rate)
  // For proper resampling, you'd use a library like rubato

  Ok(samples)
}

#[cfg(test)]
mod tests {
  use ort::session::input;

  // Note this useful idiom: importing names from outer (for mod tests) scope.
  use super::*;

  #[tokio::test]
  #[tracing_test::traced_test]
  async fn test_inference() {
    use ort::value::Value;
    use ort::value::ValueRef;
    use ortts_shared::Downloader;
    use tokenizers::Tokenizer;
    use tracing::info;

    // TODO: Start with 5 for testing, Python uses 256
    const MAX_NEW_TOKENS: usize = 256;
    // const S3GEN_SR: u32 = 24000;
    const START_SPEECH_TOKEN: u32 = 6561;
    const STOP_SPEECH_TOKEN: u32 = 6562;
    const NUM_HIDDEN_LAYERS: i64 = 30;
    const NUM_KEY_VALUE_HEADS: i64 = 16;
    const HEAD_DIM: i64 = 64;

    use crate::utils::repetition_penalty_logits_processor::RepetitionPenaltyLogitsProcessor;

    let repetition_penalty = 1.2_f32;
    let processor = RepetitionPenaltyLogitsProcessor::new(repetition_penalty).unwrap();

    // Generate tokens - for the first iteration, this would be [[START_SPEECH_TOKEN]]
    // Make it mutable so we can concatenate new tokens in each iteration
    let mut generate_tokens =
      ndarray::Array2::<usize>::from_shape_vec((1, 1), vec![START_SPEECH_TOKEN as usize]).unwrap();

    let downloader = Downloader::new();

    let speech_encoder_path = downloader
      .get_path(
        "onnx-community/chatterbox-multilingual-ONNX",
        "onnx/speech_encoder.onnx",
      )
      .await
      .unwrap();
    let embed_tokens_path = downloader
      .get_path(
        "onnx-community/chatterbox-multilingual-ONNX",
        "onnx/embed_tokens.onnx",
      )
      .await
      .unwrap();
    let llama_with_path_path = downloader
      .get_path(
        "onnx-community/chatterbox-multilingual-ONNX",
        "onnx/language_model_q4.onnx",
      )
      .await
      .unwrap();
    let conditional_decoder_path = downloader
      .get_path(
        "onnx-community/chatterbox-multilingual-ONNX",
        "onnx/conditional_decoder.onnx",
      )
      .await
      .unwrap();

    let tokenizer_config_path = downloader
      .get_path(
        "onnx-community/chatterbox-multilingual-ONNX",
        "tokenizer.json",
      )
      .await
      .unwrap();

    assert!(speech_encoder_path.exists());
    assert!(embed_tokens_path.exists());
    assert!(llama_with_path_path.exists());
    assert!(conditional_decoder_path.exists());

    assert!(tokenizer_config_path.exists());

    let mut embed_tokens_session = create_session(embed_tokens_path.to_str().unwrap()).unwrap();
    let mut speech_encoder_session = create_session(speech_encoder_path.to_str().unwrap()).unwrap();
    let mut llama_with_past_session =
      create_session(llama_with_path_path.to_str().unwrap()).unwrap();
    let _conditional_decoder_session =
      create_session(conditional_decoder_path.to_str().unwrap()).unwrap();

    let tokenizer =
      Tokenizer::from_pretrained("onnx-community/chatterbox-multilingual-ONNX", None).unwrap();

    let target_voice_path = downloader
      .get_path(
        "onnx-community/chatterbox-multilingual-ONNX",
        "default_voice.wav",
      )
      .await
      .unwrap();

    // Convert to ort Value with shape [1, audio_length]
    let audio_values = load_audio(target_voice_path.to_str().unwrap()).unwrap();
    let audio_shape = vec![1_usize, audio_values.len()];
    let audio_value = Value::from_array((audio_shape.as_slice(), audio_values)).unwrap();
    info!(
      "audio shape: {:?}, dtype: {:?}",
      audio_value.shape(),
      audio_value.data_type()
    );

    let text = "[en]The Lord of the Rings is the greatest work of literature.";

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

    let input_ids = tokenized_input.get_ids();
    let input_ids_shape = vec![1_usize, input_ids.len()];
    let input_ids_data: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();

    // position_ids = np.where(
    //   input_ids >= START_SPEECH_TOKEN,
    //   0,
    //   np.arange(input_ids.shape[1])[np.newaxis, :] - 1
    // )
    let position_ids_shape = vec![1_usize, input_ids.len()];
    let position_ids_data: Vec<i64> = input_ids
      .iter()
      .enumerate()
      .map(|(i, &token_id)| {
        if token_id >= START_SPEECH_TOKEN {
          0
        } else {
          i as i64 - 1
        }
      })
      .collect();

    // REVIEW: might be able to be optimized
    let input_ids_value = Value::from_array((input_ids_shape.as_slice(), input_ids_data)).unwrap();
    let position_ids_value =
      Value::from_array((position_ids_shape.as_slice(), position_ids_data)).unwrap();

    // exaggeration=0.5
    let exaggeration = 0.5_f32;
    // np.array([exaggeration], dtype=np.float32)
    let exaggeration_shape = vec![1_usize];
    let exaggeration_data = vec![exaggeration];
    let exaggeration_value =
      Value::from_array((exaggeration_shape.as_slice(), exaggeration_data)).unwrap();

    let mut attention_mask_array = ndarray::Array2::<i64>::zeros((0, 0));
    let mut batch_size = 0;
    // KV Cache
    let mut past_key_values: std::collections::HashMap<String, Value> =
      std::collections::HashMap::new();

    // NOTICE: Reuseable during generation loop
    let mut ort_embed_tokens_input_ids = input_ids_value.clone();
    let mut ort_embed_tokens_position_ids = position_ids_value.clone();
    let ort_embed_tokens_exaggeration = exaggeration_value.clone();

    // TODO: Speech conditional decoder model required
    let mut prompt_token: Option<ndarray::Array2<i64>> = None;
    let mut speaker_embeddings_array: Option<ndarray::Array2<f32>> = None;
    let mut speaker_features_array: Option<ndarray::Array3<f32>> = None;

    for i in 0..MAX_NEW_TOKENS {
      // inputs_embeds = embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]
      let mut ort_input_embeds_output = embed_tokens_session
        .run(ort::inputs![
          "input_ids" => &ort_embed_tokens_input_ids,
          "position_ids" => &ort_embed_tokens_position_ids,
          "exaggeration" => &ort_embed_tokens_exaggeration,
        ])
        .unwrap();
      let mut inputs_embeds_value: Value = ort_input_embeds_output.remove("inputs_embeds").unwrap();

      if i == 0 {
        // cond_emb, prompt_token, speaker_embeddings, speaker_features = speech_encoder_session.run(None, ort_speech_encoder_input)
        let ort_speech_encoder_output = speech_encoder_session
          .run(ort::inputs!["audio_values" => &audio_value])
          .unwrap();
        let audio_features = ort_speech_encoder_output.get("audio_features").unwrap();
        let audio_tokens = ort_speech_encoder_output.get("audio_tokens").unwrap();
        let speaker_embeddings = ort_speech_encoder_output.get("speaker_embeddings").unwrap();
        let speaker_features = ort_speech_encoder_output.get("speaker_features").unwrap();

        info!("audio_features: {:?}", audio_features.shape());
        info!("audio_tokens: {:?}", audio_tokens.shape());
        info!("speaker_embeddings: {:?}", speaker_embeddings.shape());
        info!("speaker_features: {:?}", speaker_features.shape());

        // Store speaker data for later use
        let (speaker_emb_shape, speaker_emb_data) =
          speaker_embeddings.try_extract_tensor::<f32>().unwrap();
        speaker_embeddings_array = Some(
          ndarray::Array2::from_shape_vec(
            (speaker_emb_shape[0] as usize, speaker_emb_shape[1] as usize),
            speaker_emb_data.to_vec(),
          )
          .unwrap(),
        );

        let (speaker_feat_shape, speaker_feat_data) =
          speaker_features.try_extract_tensor::<f32>().unwrap();
        speaker_features_array = Some(
          ndarray::Array3::from_shape_vec(
            (
              speaker_feat_shape[0] as usize,
              speaker_feat_shape[1] as usize,
              speaker_feat_shape[2] as usize,
            ),
            speaker_feat_data.to_vec(),
          )
          .unwrap(),
        );

        let (audio_tok_shape, audio_tok_data) = audio_tokens.try_extract_tensor::<i64>().unwrap();
        prompt_token = Some(
          ndarray::Array2::from_shape_vec(
            (audio_tok_shape[0] as usize, audio_tok_shape[1] as usize),
            audio_tok_data.to_vec(),
          )
          .unwrap(),
        );

        // Concatenate audio_features with inputs_embeds
        let (audio_shape, audio_data) = audio_features.try_extract_tensor::<f32>().unwrap();
        let (embeds_shape, embeds_data) = inputs_embeds_value.try_extract_tensor::<f32>().unwrap();

        use ndarray::ArrayView3;
        let concatenated = ndarray::concatenate(
          ndarray::Axis(1),
          &[
            ArrayView3::from_shape(
              (
                audio_shape[0] as usize,
                audio_shape[1] as usize,
                audio_shape[2] as usize,
              ),
              audio_data,
            )
            .unwrap(),
            ArrayView3::from_shape(
              (
                embeds_shape[0] as usize,
                embeds_shape[1] as usize,
                embeds_shape[2] as usize,
              ),
              embeds_data,
            )
            .unwrap(),
          ],
        )
        .unwrap();

        let concat_shape: Vec<usize> = concatenated.shape().to_vec();
        let concat_data: Vec<f32> = concatenated.iter().copied().collect();
        inputs_embeds_value = Value::from_array((concat_shape.as_slice(), concat_data))
          .unwrap()
          .into();

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
        attention_mask_array =
          ndarray::Array2::<i64>::ones((batch_size as usize, seq_len as usize));
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
      // next_token_logits = repetition_penalty_processor(generate_tokens, logits)
      let next_token_logits = processor.call(generate_tokens.row(0), &last_token_logits);
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

      // Update KV Cache
      let past_keys: Vec<String> = past_key_values.keys().cloned().collect();
      for (j, key) in past_keys.iter().enumerate() {
        let present_value = &present_key_values[j];
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
        past_key_values.insert(key.clone(), Value::from_array(pres_array).unwrap().into());
      }
    }

    info!("Final generate_tokens: {:?}", generate_tokens);
  }
}
