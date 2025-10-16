use std::path::PathBuf;

use crate::AppError;

#[derive(Debug)]
pub struct Downloader {
  cache_api: hf_hub::Cache,
  api: hf_hub::api::tokio::Api,
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
}
