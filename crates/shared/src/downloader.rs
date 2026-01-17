use std::path::PathBuf;

use crate::AppError;

#[derive(Debug)]
pub struct Downloader {
  api: hf_hub::api::tokio::Api,
  cache: hf_hub::Cache,
  model_id: String,
}

impl Downloader {
  pub fn new(model_id: String) -> Self {
    let cache = hf_hub::Cache::from_env();
    let api = hf_hub::api::tokio::ApiBuilder::from_env().build().unwrap();

    Self {
      cache,
      api,
      model_id,
    }
  }

  pub async fn get_path(&self, filename: &str) -> Result<PathBuf, AppError> {
    let path = match self.cache.model(self.model_id.clone()).get(filename) {
      Some(p) => p,
      None => self.api.model(self.model_id.clone()).get(filename).await?,
    };

    Ok(path)
  }

  pub async fn get_str(&self, filename: &str) -> Result<String, AppError> {
    let path = self.get_path(filename).await?;
    let str = std::fs::read_to_string(path)?;

    Ok(str)
  }

  /// get `filename.onnx` and `filename.onnx_data`
  pub async fn get_onnx_with_data(&self, filename: &str) -> Result<PathBuf, AppError> {
    let path = self.get_path(filename).await?;

    match self.get_path(&format!("{filename}_data")).await {
      Ok(_) => Ok(path),
      Err(e) => Err(e),
    }
  }

  /// get `tokenizer.json`
  pub async fn get_tokenizer(&self) -> Result<PathBuf, AppError> {
    self.get_path("tokenizer.json").await
  }
}
