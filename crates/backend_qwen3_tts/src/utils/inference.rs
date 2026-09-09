use std::{borrow::Cow, path::PathBuf};

use anyhow::anyhow;
use futures::stream;
use ndarray::{Array2, Array3, Array4, Axis, concatenate, s};
use ort::{session::SessionInputValue, value::Value};
use ortts_onnx::{SessionPool, inference_session};
use ortts_shared::{AppError, AudioSpec, Downloader, SpeechAudioStream, SpeechOptions};
use rand::{SeedableRng, rngs::StdRng};

use super::{
  audio::load_mono,
  config::ModelConfig,
  language::Language,
  sampling::{SamplingOptions, apply_repetition_penalty, sample},
  tokenizer::{assistant_ids, load as load_tokenizer},
};

const MODEL_ID: &str = "onnx-community/Qwen3-TTS-12Hz-0.6B-Base";
const CONFIG_ID: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
#[cfg(not(feature = "ep_cuda"))]
const MODEL_VARIANT: &str = "cpu_int4";
#[cfg(feature = "ep_cuda")]
const MODEL_VARIANT: &str = "cuda_int4";
const SAMPLE_RATE: u32 = 24_000;
const CODE_GROUPS: usize = 16;
const DECODER_FRAMES: usize = 25;
const MAX_NEW_TOKENS: usize = 2_048;
const REPETITION_PENALTY: f32 = 1.05;

struct ModelPaths {
  code_predictor: PathBuf,
  codec_embed: PathBuf,
  residual_embed: PathBuf,
  speaker_encoder: PathBuf,
  talker_cache: PathBuf,
  text_embed: PathBuf,
  token_decoder: PathBuf,
  config: PathBuf,
  vocab: PathBuf,
  merges: PathBuf,
}

struct TalkerStep {
  logits: Array3<f32>,
  hidden: Array3<f32>,
  cache: Vec<Value>,
}

struct QwenRequest {
  input: String,
  language: Language,
  reference_path: PathBuf,
  paths: ModelPaths,
}

struct QwenStream {
  request: Option<QwenRequest>,
}

/// Synthesizes one Qwen3-TTS Base request into a mono 24 kHz WAV.
///
/// # Errors
///
/// Returns an error when the model assets, reference audio, tokenization, ONNX execution,
/// codec generation, or WAV encoding fails.
pub async fn inference(options: SpeechOptions) -> Result<SpeechAudioStream, AppError> {
  let state = QwenStream::prepare(options).await?;
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

impl QwenStream {
  async fn prepare(options: SpeechOptions) -> Result<Self, AppError> {
    let language = Language::from_model_name(&options.model)?;
    let reference_path = PathBuf::from(&options.voice);
    if !reference_path.is_file() {
      return Err(
        anyhow!(
          "Qwen3-TTS Base expects `voice` to be a local reference-audio file; '{}' is not a file",
          reference_path.display()
        )
        .into(),
      );
    }

    Ok(Self {
      request: Some(QwenRequest {
        input: options.input,
        language,
        reference_path,
        paths: Box::pin(download_model()).await?,
      }),
    })
  }

  fn next_audio_chunk(&mut self) -> Result<Option<Vec<f32>>, AppError> {
    let Some(request) = self.request.take() else {
      return Ok(None);
    };
    synthesize(request).map(Some)
  }
}

fn synthesize(request: QwenRequest) -> Result<Vec<f32>, AppError> {
  let config = ModelConfig::from_path(&request.paths.config)?;
  let tokenizer = load_tokenizer(&request.paths.vocab, &request.paths.merges)?;
  let input_ids = assistant_ids(&tokenizer, &request.input)?;
  if input_ids.len() < 9 {
    return Err(anyhow!("Qwen3-TTS input token sequence is unexpectedly short").into());
  }

  let mut code_predictor = inference_session(&request.paths.code_predictor)?;
  let mut codec_embed = inference_session(&request.paths.codec_embed)?;
  let mut residual_embed = inference_session(&request.paths.residual_embed)?;
  let mut speaker_encoder = inference_session(&request.paths.speaker_encoder)?;
  let mut talker_cache = inference_session(&request.paths.talker_cache)?;
  let mut text_embed = inference_session(&request.paths.text_embed)?;
  let mut token_decoder = inference_session(&request.paths.token_decoder)?;

  let reference_audio = load_mono(&request.reference_path, SAMPLE_RATE)?;
  let speaker = speaker_embedding(
    &mut speaker_encoder,
    reference_audio,
    config.talker_config.hidden_size,
  )?;
  let prefill = build_prefill(
    &config,
    request.language,
    &input_ids,
    &speaker,
    &mut text_embed,
    &mut codec_embed,
  )?;
  let trailing = embed_text_ids(
    &mut text_embed,
    Array2::from_shape_vec((1, 1), vec![config.tts_pad_token_id])?,
  )?
  .index_axis(Axis(1), 0)
  .to_owned();

  let codes = generate_codes(
    &config,
    prefill,
    &trailing,
    &mut talker_cache,
    &mut code_predictor,
    &mut residual_embed,
  )?;
  decode_codes(&codes, &mut token_decoder)
}

async fn download_model() -> Result<ModelPaths, AppError> {
  let models = Downloader::new(MODEL_ID.to_owned())?;
  let config = Downloader::new(CONFIG_ID.to_owned())?;
  let prefix = MODEL_VARIANT;
  let code_predictor_file = format!("{prefix}/code_predictor.onnx");
  let codec_embed_file = format!("{prefix}/codec_embed.onnx");
  let residual_embed_file = format!("{prefix}/residual_embed.onnx");
  let speaker_encoder_file = format!("{prefix}/speaker_encoder.onnx");
  let talker_cache_file = format!("{prefix}/talker_cache.onnx");
  let text_embed_file = format!("{prefix}/text_embed.onnx");
  let token_decoder_file = format!("{prefix}/tok_decoder.onnx");
  let (
    code_predictor,
    codec_embed,
    residual_embed,
    speaker_encoder,
    talker_cache,
    text_embed,
    token_decoder,
    config_path,
    vocab,
    merges,
  ) = tokio::try_join!(
    models.get_path(&code_predictor_file),
    models.get_path(&codec_embed_file),
    models.get_path(&residual_embed_file),
    models.get_path(&speaker_encoder_file),
    models.get_path(&talker_cache_file),
    models.get_path(&text_embed_file),
    models.get_path(&token_decoder_file),
    config.get_path("config.json"),
    config.get_path("vocab.json"),
    config.get_path("merges.txt"),
  )?;

  Ok(ModelPaths {
    code_predictor,
    codec_embed,
    residual_embed,
    speaker_encoder,
    talker_cache,
    text_embed,
    token_decoder,
    config: config_path,
    vocab,
    merges,
  })
}

fn speaker_embedding(
  session: &mut SessionPool,
  audio: Vec<f32>,
  hidden_size: usize,
) -> Result<Array3<f32>, AppError> {
  let audio = Array2::from_shape_vec((1, audio.len()), audio)?;
  let output_name = first_output_name(session)?;
  let output = session
    .run(ort::inputs!["audio" => Value::from_array(audio)?])?
    .remove(&output_name)
    .ok_or_else(|| anyhow!("speaker encoder did not return '{output_name}'"))?;
  let (_, values) = output.try_extract_tensor::<f32>()?;
  if values.len() != hidden_size {
    return Err(
      anyhow!(
        "speaker encoder returned {} values, expected {hidden_size}",
        values.len()
      )
      .into(),
    );
  }
  Ok(Array3::from_shape_vec(
    (1, 1, hidden_size),
    values.to_vec(),
  )?)
}

fn build_prefill(
  config: &ModelConfig,
  language: Language,
  input_ids: &[i64],
  speaker: &Array3<f32>,
  text_embed: &mut SessionPool,
  codec_embed: &mut SessionPool,
) -> Result<Array3<f32>, AppError> {
  let special = embed_text_ids(
    text_embed,
    Array2::from_shape_vec(
      (1, 3),
      vec![
        config.tts_bos_token_id,
        config.tts_eos_token_id,
        config.tts_pad_token_id,
      ],
    )?,
  )?;
  let bos = special.slice(s![.., 0..1, ..]).to_owned();
  let eos = special.slice(s![.., 1..2, ..]).to_owned();
  let pad = special.slice(s![.., 2..3, ..]).to_owned();

  let talker = &config.talker_config;
  let codec_prefill = if let Some(language) = language.config_key() {
    let language_id = talker
      .codec_language_id
      .get(language)
      .copied()
      .ok_or_else(|| anyhow!("Qwen3-TTS config does not contain language '{language}'"))?;
    vec![
      talker.codec_think_id,
      talker.codec_think_bos_id,
      language_id,
      talker.codec_think_eos_id,
    ]
  } else {
    vec![
      talker.codec_nothink_id,
      talker.codec_think_bos_id,
      talker.codec_think_eos_id,
    ]
  };
  let codec_prefix = embed_codec_ids(
    codec_embed,
    Array2::from_shape_vec((1, codec_prefill.len()), codec_prefill)?,
  )?;
  let codec_tail = embed_codec_ids(
    codec_embed,
    Array2::from_shape_vec((1, 2), vec![talker.codec_pad_id, talker.codec_bos_id])?,
  )?;
  let codec_input = concatenate(
    Axis(1),
    &[codec_prefix.view(), speaker.view(), codec_tail.view()],
  )?;

  let role = embed_text_ids(
    text_embed,
    Array2::from_shape_vec((1, 3), input_ids[..3].to_vec())?,
  )?;
  let pad_count = codec_input.shape()[1] - 2;
  let pad_prefix = pad
    .broadcast((1, pad_count, talker.hidden_size))
    .ok_or_else(|| anyhow!("failed to broadcast Qwen3-TTS pad embedding"))?
    .to_owned();
  let pad_block = concatenate(Axis(1), &[pad_prefix.view(), bos.view()])?;
  let codec_without_bos = codec_input.slice(s![.., ..-1, ..]);
  let prompt = pad_block + codec_without_bos;
  let mut prefill = concatenate(Axis(1), &[role.view(), prompt.view()])?;

  let body_ids = &input_ids[3..input_ids.len() - 5];
  let text_body = embed_text_ids(
    text_embed,
    Array2::from_shape_vec((1, body_ids.len()), body_ids.to_vec())?,
  )?;
  let text_and_eos = concatenate(Axis(1), &[text_body.view(), eos.view()])?;
  let codec_pad = embed_codec_ids(
    codec_embed,
    Array2::from_elem((1, body_ids.len() + 1), talker.codec_pad_id),
  )?;
  let text_block = text_and_eos + codec_pad;
  let codec_bos = embed_codec_ids(
    codec_embed,
    Array2::from_shape_vec((1, 1), vec![talker.codec_bos_id])?,
  )?;
  let final_block = pad + codec_bos;
  prefill = concatenate(
    Axis(1),
    &[prefill.view(), text_block.view(), final_block.view()],
  )?;
  Ok(prefill)
}

#[allow(clippy::too_many_lines)]
fn generate_codes(
  config: &ModelConfig,
  prefill: Array3<f32>,
  trailing: &Array2<f32>,
  talker_cache: &mut SessionPool,
  code_predictor: &mut SessionPool,
  residual_embed: &mut SessionPool,
) -> Result<Vec<[i64; CODE_GROUPS]>, AppError> {
  let cache_input_names: Vec<String> = talker_cache
    .inputs()
    .iter()
    .skip(3)
    .map(|input| input.name().to_owned())
    .collect();
  let cache_output_names: Vec<String> = talker_cache
    .outputs()
    .iter()
    .skip(2)
    .map(|output| output.name().to_owned())
    .collect();
  if cache_input_names.len() != cache_output_names.len() {
    return Err(
      anyhow!(
        "Qwen3-TTS talker cache has {} cache inputs but {} cache outputs",
        cache_input_names.len(),
        cache_output_names.len()
      )
      .into(),
    );
  }
  let expected_cache_values = config.talker_config.num_hidden_layers * 2;
  if cache_input_names.len() != expected_cache_values {
    return Err(
      anyhow!(
        "Qwen3-TTS talker cache has {} tensors, expected {expected_cache_values}",
        cache_input_names.len()
      )
      .into(),
    );
  }

  let mut past = Vec::with_capacity(cache_input_names.len());
  for _ in &cache_input_names {
    past.push(
      Value::from_array(Array4::<f32>::zeros((
        1,
        config.talker_config.num_key_value_heads,
        0,
        config.talker_config.head_dim,
      )))?
      .into(),
    );
  }

  let mut total_length = prefill.shape()[1];
  let position_ids = Array3::from_shape_fn((3, 1, total_length), |(_, _, position)| {
    i64::try_from(position).expect("Qwen3-TTS position should fit in i64")
  });
  let attention_mask = Array2::<i64>::ones((1, total_length));
  let initial = run_cached_talker(
    talker_cache,
    &cache_input_names,
    &cache_output_names,
    prefill,
    position_ids,
    attention_mask,
    &past,
  )?;
  let mut logits = initial.logits;
  let mut hidden = initial.hidden;
  past = initial.cache;

  let mut rng = StdRng::seed_from_u64(0);
  let mut generated = Vec::new();
  let mut previous_first = Vec::new();
  let talker = &config.talker_config;

  for step in 0..MAX_NEW_TOKENS {
    let mut first_logits = logits
      .slice(s![0, -1, ..])
      .iter()
      .copied()
      .collect::<Vec<_>>();
    for (token, logit) in first_logits
      .iter_mut()
      .enumerate()
      .take(talker.vocab_size)
      .skip(talker.vocab_size.saturating_sub(1_024))
    {
      if token != talker.codec_eos_token_id {
        *logit = f32::NEG_INFINITY;
      }
    }
    if step < 2 {
      first_logits[talker.codec_eos_token_id] = f32::NEG_INFINITY;
    }
    apply_repetition_penalty(&mut first_logits, &previous_first, REPETITION_PENALTY);
    let first = sample(&first_logits, SamplingOptions::main(), &mut rng);
    if first == talker.codec_eos_token_id {
      break;
    }
    previous_first.push(first);

    let talker_hidden = hidden.slice(s![0, -1, ..]).insert_axis(Axis(0)).to_owned();
    let mut frame = [0_i64; CODE_GROUPS];
    frame[0] = i64::try_from(first).expect("Qwen3-TTS codec token should fit in i64");
    for group in 1..CODE_GROUPS {
      let predictor_logits = predict_residual(code_predictor, &talker_hidden, &frame)?;
      let group_logits = predictor_logits
        .slice(s![0, group - 1, ..])
        .iter()
        .copied()
        .collect::<Vec<_>>();
      frame[group] = i64::try_from(sample(&group_logits, SamplingOptions::main(), &mut rng))
        .expect("Qwen3-TTS residual token should fit in i64");
    }
    generated.push(frame);

    let step_embedding = embed_residual(residual_embed, &frame)?;
    let next = (step_embedding + trailing).insert_axis(Axis(1));
    let next_position = Array3::from_elem(
      (3, 1, 1),
      i64::try_from(total_length).expect("Qwen3-TTS position should fit in i64"),
    );
    total_length += 1;
    let next_attention = Array2::<i64>::ones((1, total_length));
    let result = run_cached_talker(
      talker_cache,
      &cache_input_names,
      &cache_output_names,
      next,
      next_position,
      next_attention,
      &past,
    )?;
    logits = result.logits;
    hidden = result.hidden;
    past = result.cache;

    if (step + 1) % 25 == 0 {
      tracing::debug!(frames = step + 1, "Qwen3-TTS generated codec frames");
    }
  }

  if generated.is_empty() {
    return Err(anyhow!("Qwen3-TTS produced no audio codec frames").into());
  }
  Ok(generated)
}

#[allow(clippy::too_many_arguments)]
fn run_cached_talker(
  session: &mut SessionPool,
  cache_input_names: &[String],
  cache_output_names: &[String],
  inputs_embeds: Array3<f32>,
  position_ids: Array3<i64>,
  attention_mask: Array2<i64>,
  past: &[Value],
) -> Result<TalkerStep, AppError> {
  let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = ort::inputs![
    "inputs_embeds" => Value::from_array(inputs_embeds)?,
    "position_ids" => Value::from_array(position_ids)?,
    "attention_mask" => Value::from_array(attention_mask)?,
  ];
  for (name, value) in cache_input_names.iter().zip(past.iter()) {
    inputs.push((name.as_str().into(), value.into()));
  }

  let output_names: Vec<String> = session
    .outputs()
    .iter()
    .take(2)
    .map(|output| output.name().to_owned())
    .collect();
  let mut outputs = session.run(inputs)?;
  let logits = outputs
    .remove(&output_names[0])
    .ok_or_else(|| anyhow!("Qwen3-TTS talker did not return logits"))?;
  let hidden = outputs
    .remove(&output_names[1])
    .ok_or_else(|| anyhow!("Qwen3-TTS talker did not return hidden states"))?;
  let logits: Array3<f32> = logits
    .try_extract_array()?
    .to_owned()
    .into_dimensionality()?;
  let hidden: Array3<f32> = hidden
    .try_extract_array()?
    .to_owned()
    .into_dimensionality()?;
  let present = cache_output_names
    .iter()
    .map(|name| {
      outputs
        .remove(name)
        .ok_or_else(|| anyhow!("Qwen3-TTS talker did not return cache tensor '{name}'").into())
    })
    .collect::<Result<Vec<_>, AppError>>()?;
  Ok(TalkerStep {
    logits,
    hidden,
    cache: present,
  })
}

fn predict_residual(
  session: &mut SessionPool,
  talker_hidden: &Array2<f32>,
  frame: &[i64; CODE_GROUPS],
) -> Result<Array3<f32>, AppError> {
  let codec_ids = Array2::from_shape_vec((1, CODE_GROUPS), frame.to_vec())?;
  let output_name = first_output_name(session)?;
  let output = session
    .run(ort::inputs![
      "talker_hidden" => Value::from_array(talker_hidden.to_owned())?,
      "codec_ids" => Value::from_array(codec_ids)?,
    ])?
    .remove(&output_name)
    .ok_or_else(|| anyhow!("Qwen3-TTS code predictor did not return '{output_name}'"))?;
  Ok(
    output
      .try_extract_array()?
      .to_owned()
      .into_dimensionality()?,
  )
}

fn embed_residual(
  session: &mut SessionPool,
  frame: &[i64; CODE_GROUPS],
) -> Result<Array2<f32>, AppError> {
  let codec_ids = Array2::from_shape_vec((1, CODE_GROUPS), frame.to_vec())?;
  let output_name = first_output_name(session)?;
  let output = session
    .run(ort::inputs!["codec_ids" => Value::from_array(codec_ids)?])?
    .remove(&output_name)
    .ok_or_else(|| anyhow!("Qwen3-TTS residual embed did not return '{output_name}'"))?;
  Ok(
    output
      .try_extract_array()?
      .to_owned()
      .into_dimensionality()?,
  )
}

fn embed_text_ids(session: &mut SessionPool, ids: Array2<i64>) -> Result<Array3<f32>, AppError> {
  run_embedding(session, "text_ids", ids)
}

fn embed_codec_ids(session: &mut SessionPool, ids: Array2<i64>) -> Result<Array3<f32>, AppError> {
  run_embedding(session, "codec_ids", ids)
}

fn run_embedding(
  session: &mut SessionPool,
  input_name: &'static str,
  ids: Array2<i64>,
) -> Result<Array3<f32>, AppError> {
  let output_name = first_output_name(session)?;
  let inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
    vec![(input_name.into(), Value::from_array(ids)?.into())];
  let output = session
    .run(inputs)?
    .remove(&output_name)
    .ok_or_else(|| anyhow!("Qwen3-TTS embedding graph did not return '{output_name}'"))?;
  Ok(
    output
      .try_extract_array()?
      .to_owned()
      .into_dimensionality()?,
  )
}

fn decode_codes(
  codes: &[[i64; CODE_GROUPS]],
  session: &mut SessionPool,
) -> Result<Vec<f32>, AppError> {
  let output_name = first_output_name(session)?;
  let mut waveform = Vec::new();
  for chunk in codes.chunks(DECODER_FRAMES) {
    let actual_frames = chunk.len();
    let padded: Vec<i64> = (0..DECODER_FRAMES)
      .flat_map(|index| chunk[index % actual_frames])
      .collect();
    let input = Array3::from_shape_vec((1, DECODER_FRAMES, CODE_GROUPS), padded)?;
    let output = session
      .run(ort::inputs!["audio_codes" => Value::from_array(input)?])?
      .remove(&output_name)
      .ok_or_else(|| anyhow!("Qwen3-TTS token decoder did not return '{output_name}'"))?;
    let (_, samples) = output.try_extract_tensor::<f32>()?;
    let keep = if actual_frames == DECODER_FRAMES {
      samples.len()
    } else {
      samples.len() * actual_frames / DECODER_FRAMES
    };
    waveform.extend_from_slice(&samples[..keep]);
  }
  Ok(waveform)
}

fn first_output_name(session: &SessionPool) -> Result<String, AppError> {
  session
    .outputs()
    .first()
    .map(|output| output.name().to_owned())
    .ok_or_else(|| anyhow!("ONNX graph has no outputs").into())
}
