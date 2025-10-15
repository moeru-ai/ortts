use std::path::PathBuf;

use ort::{
  session::Session,
};
use anyhow::Result;

mod utils;

pub struct CachedHub {
  cache_api: hf_hub::Cache,
  api: hf_hub::api::tokio::Api,
}

impl CachedHub {
  pub fn new() -> Self {
    let cache_api = hf_hub::Cache::from_env();
    let api = hf_hub::api::tokio::Api::new().unwrap();

    Self { cache_api, api }
  }

  pub async fn get(&self, model_id: &str, filename: &str) -> Result<PathBuf> {
    let path = match self.cache_api.model(String::from(model_id))
      .get(filename) {
      Some(p) => p,
      None => {
        self.api.model(String::from(model_id))
          .get(filename)
          .await?
      }
    };

    Ok(path)
  }
}

pub fn create_session(model_path: &str) -> Result<Session> {
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

#[cfg(test)]
mod tests {

// Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

  #[tokio::test]
  #[tracing_test::traced_test]
  async fn test_inference() {
    use tokenizers::Tokenizer;
    use tracing::{info};
    use ort::{
      value::Value,
    };

    // const S3GEN_SR: u32 = 24000;
    const START_SPEECH_TOKEN: u32 = 6561;
    // const STOP_SPEECH_TOKEN: u32 = 6562;

    let fetcher = CachedHub::new();

    let speech_encoder_path = fetcher.get("onnx-community/chatterbox-multilingual-ONNX", "onnx/speech_encoder.onnx").await.unwrap();
    let embed_tokens_path = fetcher.get("onnx-community/chatterbox-multilingual-ONNX", "onnx/embed_tokens.onnx").await.unwrap();
    let llama_with_path_path = fetcher.get("onnx-community/chatterbox-multilingual-ONNX", "onnx/language_model_q4.onnx").await.unwrap();
    let conditional_decoder_path = fetcher.get("onnx-community/chatterbox-multilingual-ONNX", "onnx/conditional_decoder.onnx").await.unwrap();

    let tokenizer_config_path = fetcher.get("onnx-community/chatterbox-multilingual-ONNX", "tokenizer.json").await.unwrap();

    assert!(speech_encoder_path.exists());
    assert!(embed_tokens_path.exists());
    assert!(llama_with_path_path.exists());
    assert!(conditional_decoder_path.exists());

    assert!(tokenizer_config_path.exists());

    let _speech_encoder_session = create_session(speech_encoder_path.to_str().unwrap()).unwrap();
    let _embed_tokens_session = create_session(embed_tokens_path.to_str().unwrap()).unwrap();
    let _llama_with_path_session = create_session(llama_with_path_path.to_str().unwrap()).unwrap();
    let _conditional_decoder_session = create_session(conditional_decoder_path.to_str().unwrap()).unwrap();

    let tokenizer = Tokenizer::from_pretrained("onnx-community/chatterbox-multilingual-ONNX", None).unwrap();
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

    // Create ONNX session inputs
    let _ort_embed_tokens_input = ort::inputs![
      "input_ids" => &input_ids_value,
      "position_ids" => &position_ids_value,
      "exaggeration" => &exaggeration_value,
    ];

    info!("input_ids shape: {:?}, dtype: {:?}", input_ids_value.shape(), input_ids_value.data_type());
    info!("position_ids shape: {:?}, dtype: {:?}", position_ids_value.shape(), position_ids_value.data_type());
    info!("exaggeration shape: {:?}, dtype: {:?}", exaggeration_value.shape(), exaggeration_value.data_type());
  }
}
