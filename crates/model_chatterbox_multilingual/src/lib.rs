use ortts_shared::AppError;

use ort::{session::Session};

mod utils;

pub fn create_session(model_path: &str) -> Result<Session, AppError> {
  use ort::session::builder::GraphOptimizationLevel;
  use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider};

  let session = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_parallel_execution(true)?
      .with_execution_providers([
        CUDAExecutionProvider::default()
          .with_device_id(0)
          .build(),
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
  use symphonia::core::io::MediaSourceStream;
  use symphonia::core::probe::Hint;
  use symphonia::default::get_probe;
  use std::fs::File;

  let file = File::open(path)?;
  let mss = MediaSourceStream::new(Box::new(file), Default::default());

  let mut hint = Hint::new();
  hint.with_extension("wav");

  let meta_opts = Default::default();
  let fmt_opts = Default::default();

  let probed = get_probe().format(&hint, mss, &meta_opts, &fmt_opts)?;
  let mut format = probed.format;
  let track = format.tracks()
      .iter()
      .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
      .ok_or_else(|| anyhow::anyhow!("No supported audio tracks"))?;

  let track_id = track.id;
  let mut decoder = symphonia::default::get_codecs()
      .make(&track.codec_params, &Default::default())?;

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
                  audio_buf = Some(symphonia::core::audio::SampleBuffer::<f32>::new(duration, spec));
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
  // Note this useful idiom: importing names from outer (for mod tests) scope.
  use super::*;

  #[tokio::test]
  #[tracing_test::traced_test]
  async fn test_inference() {
    use ortts_shared::Downloader;
    use tokenizers::Tokenizer;
    use tracing::{info};
    use ort::{
      value::Value,
    };

    // const MAX_NEW_TOKENS: i64 = 256;
    // const S3GEN_SR: u32 = 24000;
    const START_SPEECH_TOKEN: u32 = 6561;
    // const STOP_SPEECH_TOKEN: u32 = 6562;
    const NUM_HIDDEN_LAYERS: i64 = 30;
    const NUM_KEY_VALUE_HEADS: i64 = 16;
    const HEAD_DIM: i64 = 64;

    let downloader = Downloader::new();

    // Then in your test at line 219, add:

    let speech_encoder_path = downloader.get_path("onnx-community/chatterbox-multilingual-ONNX", "onnx/speech_encoder.onnx").await.unwrap();
    let embed_tokens_path = downloader.get_path("onnx-community/chatterbox-multilingual-ONNX", "onnx/embed_tokens.onnx").await.unwrap();
    let llama_with_path_path = downloader.get_path("onnx-community/chatterbox-multilingual-ONNX", "onnx/language_model_q4.onnx").await.unwrap();
    let conditional_decoder_path = downloader.get_path("onnx-community/chatterbox-multilingual-ONNX", "onnx/conditional_decoder.onnx").await.unwrap();

    let tokenizer_config_path = downloader.get_path("onnx-community/chatterbox-multilingual-ONNX", "tokenizer.json").await.unwrap();

    assert!(speech_encoder_path.exists());
    assert!(embed_tokens_path.exists());
    assert!(llama_with_path_path.exists());
    assert!(conditional_decoder_path.exists());

    assert!(tokenizer_config_path.exists());

    let mut embed_tokens_session = create_session(embed_tokens_path.to_str().unwrap()).unwrap();
    let mut speech_encoder_session = create_session(speech_encoder_path.to_str().unwrap()).unwrap();
    let _llama_with_path_session = create_session(llama_with_path_path.to_str().unwrap()).unwrap();
    let _conditional_decoder_session = create_session(conditional_decoder_path.to_str().unwrap()).unwrap();

    let tokenizer = Tokenizer::from_pretrained("onnx-community/chatterbox-multilingual-ONNX", None).unwrap();

    let target_voice_path = downloader.get_path("onnx-community/chatterbox-multilingual-ONNX", "default_voice.wav").await.unwrap();

    // Convert to ort Value with shape [1, audio_length]
    let audio_values = load_audio(target_voice_path.to_str().unwrap()).unwrap();
    let audio_shape = vec![1_usize, audio_values.len()];
    let audio_value = Value::from_array((audio_shape.as_slice(), audio_values)).unwrap();
    info!("audio shape: {:?}, dtype: {:?}", audio_value.shape(), audio_value.data_type());

    let text = "[en]The Lord of the Rings is the greatest work of literature.";

    let tokenized_input = tokenizer.encode(text, true).unwrap();
    info!("{:?}, shape: {:?}", tokenized_input.get_tokens(), tokenized_input.get_tokens().len());
    info!("{:?}, shape: {:?}", tokenized_input.get_ids(), tokenized_input.get_ids().len());


    // For ort, we need to provide arrays as (shape, data) tuples
    let input_ids = tokenized_input.get_ids();
    let input_ids_shape = vec![1_usize, input_ids.len()];
    let input_ids_data: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();

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

    // Create ONNX Value objects from (shape, data) tuples
    let input_ids_value = Value::from_array((input_ids_shape.as_slice(), input_ids_data)).unwrap();
    let position_ids_value = Value::from_array((position_ids_shape.as_slice(), position_ids_data)).unwrap();

    // Create exaggeration scalar as 1D array with shape [1]
    let exaggeration = 0.5_f32;
    let exaggeration_shape = vec![1_usize];
    let exaggeration_data = vec![exaggeration];
    let exaggeration_value = Value::from_array((exaggeration_shape.as_slice(), exaggeration_data)).unwrap();

    // // Create ONNX session inputs
    // let ort_embed_tokens_input = ort::inputs![
    //   "input_ids" => &input_ids_value,
    //   "position_ids" => &position_ids_value,
    //   "exaggeration" => &exaggeration_value,
    // ];

    // info!("input_ids shape: {:?}, dtype: {:?}", input_ids_value.shape(), input_ids_value.data_type());
    // info!("position_ids shape: {:?}, dtype: {:?}", position_ids_value.shape(), position_ids_value.data_type());
    // info!("exaggeration shape: {:?}, dtype: {:?}", exaggeration_value.shape(), exaggeration_value.data_type());

    // # ---- Generation Loop using kv_cache ----
    // for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
    //     inputs_embeds = embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]
    //     if i == 0:
    //         ort_speech_encoder_input = {
    //             "audio_values": audio_values,
    //         }
    //         cond_emb, prompt_token, speaker_embeddings, speaker_features = speech_encoder_session.run(None, ort_speech_encoder_input)
    //         inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)

    //         ## Prepare llm inputs
    //         batch_size, seq_len, _ = inputs_embeds.shape
    //         past_key_values = {
    //             f"past_key_values.{layer}.{kv}": np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    //             for layer in range(num_hidden_layers)
    //             for kv in ("key", "value")
    //         }
    //         attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    //     logits, *present_key_values = llama_with_past_session.run(None, dict(
    //         inputs_embeds=inputs_embeds,
    //         attention_mask=attention_mask,
    //         **past_key_values,
    //     ))

    //     logits = logits[:, -1, :]
    //     next_token_logits = repetition_penalty_processor(generate_tokens, logits)

    //     next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
    //     generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)
    //     if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
    //         break

    //     # Get embedding for the new token.
    //     position_ids = np.full(
    //         (input_ids.shape[0], 1),
    //         i + 1,
    //         dtype=np.int64,
    //     )
    //     ort_embed_tokens_inputs["input_ids"] = next_token
    //     ort_embed_tokens_inputs["position_ids"] = position_ids

    //     ## Update values for next generation loop
    //     attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
    //     for j, key in enumerate(past_key_values):
    //         past_key_values[key] = present_key_values[j]

    // speech_tokens = generate_tokens[:, 1:-1]
    // speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)

    // cond_incoder_input = {
    //     "speech_tokens": speech_tokens,
    //     "speaker_embeddings": speaker_embeddings,
    //     "speaker_features": speaker_features,
    // }
    // wav = cond_decoder_session.run(None, cond_incoder_input)[0]
    // wav = np.squeeze(wav, axis=0)

    // # Optional: Apply watermark
    // if apply_watermark:
    //     import perth
    //     watermarker = perth.PerthImplicitWatermarker()
    //     wav = watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)

    // sf.write(output_file_name, wav, S3GEN_SR)
    // print(f"{output_file_name} was successfully saved")

    // for i in 0..MAX_NEW_TOKENS {
    let ort_input_embeds_output = embed_tokens_session.run(ort::inputs![
      "input_ids" => &input_ids_value,
      "position_ids" => &position_ids_value,
      "exaggeration" => &exaggeration_value,
    ]).unwrap();
    let inputs_embeds = ort_input_embeds_output.get("inputs_embeds").unwrap();
    println!("input_embeds: {:?}", inputs_embeds.shape());

    let ort_speech_encoder_output = speech_encoder_session.run(ort::inputs![
      "audio_values" => &audio_value
    ]).unwrap();
    let audio_features = ort_speech_encoder_output.get("audio_features").unwrap();
    let audio_tokens = ort_speech_encoder_output.get("audio_tokens").unwrap();
    let speaker_embeddings = ort_speech_encoder_output.get("speaker_embeddings").unwrap();
    let speaker_features = ort_speech_encoder_output.get("speaker_features").unwrap();
    println!("audio_features: {:?}", audio_features.shape());
    println!("audio_tokens: {:?}", audio_tokens.shape());
    println!("speaker_embeddings: {:?}", speaker_embeddings.shape());
    println!("speaker_features: {:?}", speaker_features.shape());

    let batch_size = inputs_embeds.shape()[0];
    let seq_len = inputs_embeds.shape()[1];

    let mut past_key_values = std::collections::HashMap::new();

    for layer in 0..NUM_HIDDEN_LAYERS {
        for kv in ["key", "value"] {
            let cache_key = format!("past_key_values.{}.{}", layer, kv);
            let cache_shape = vec![
                batch_size as usize,
                NUM_KEY_VALUE_HEADS as usize,
                1_usize, // ort doesn't support 0 dimension, use 1 and handle accordingly in model
                HEAD_DIM as usize
            ];

            let total_elements = cache_shape.iter().product::<usize>();
            let cache_data = vec![0.0f32; total_elements];
            let cache_value = Value::from_array((cache_shape.as_slice(), cache_data)).unwrap();
            println!("kv cache layer created {:?}, based on {:?}, {:?}", cache_value.shape(), batch_size, seq_len);
            past_key_values.insert(cache_key, cache_value);
        }
    }

    println!("Created KV cache with {} entries", past_key_values.len());
    // }
  }
}
