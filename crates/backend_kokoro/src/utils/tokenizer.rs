use std::{collections::HashMap, fs};

use ortts_shared::{AppError, Downloader};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Tokenizer {
  model: TokenizerModel,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerModel {
  vocab: HashMap<char, i64>,
}

impl Tokenizer {
  pub async fn new() -> Result<Self, AppError> {
    let downloader = Downloader::new("onnx-community/Kokoro-82M-v1.0-ONNX".to_owned());

    let path = downloader.get_tokenizer().await?;

    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
  }

  pub fn encode(&self, input: &str) -> Vec<i64> {
    input
      .chars()
      .filter_map(|c| self.model.vocab.get(&c).copied())
      .collect()
  }
}
