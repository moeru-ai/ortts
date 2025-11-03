use std::path::PathBuf;

use crate::AppError;

#[derive(Debug)]
pub struct Downloader {
  cache_api: hf_hub::Cache,
  api: hf_hub::api::tokio::Api,
}

impl Default for Downloader {
  fn default() -> Self {
    Self::new()
  }
}

impl Downloader {
  pub fn new() -> Self {
    let cache_api = hf_hub::Cache::from_env();
    let api = hf_hub::api::tokio::Api::new().unwrap();

    Self { cache_api, api }
  }

  pub async fn get_path(&self, model_id: &str, filename: &str) -> Result<PathBuf, AppError> {
    let path = match self.cache_api.model(String::from(model_id)).get(filename) {
      Some(p) => p,
      None => self.api.model(String::from(model_id)).get(filename).await?,
    };

    Ok(path)
  }

  pub async fn get_str(&self, model_id: &str, filename: &str) -> Result<String, AppError> {
    let path = self.get_path(model_id, filename).await?;
    let str = std::fs::read_to_string(path)?;

    Ok(str)
  }

  /// get `filename.onnx` with `filename.onnx_data`
  pub async fn get_onnx_with_data(
    &self,
    model_id: &str,
    filename: &str,
  ) -> Result<PathBuf, AppError> {
    let path = self.get_path(model_id, filename).await?;
    // let _data_path = self.get_path(model_id, &format!("{filename}_data")).await?;

    // Ok(path)
    match self.get_path(model_id, &format!("{filename}_data")).await {
      Ok(_) => Ok(path),
      Err(e) => Err(e),
    }
  }
}
