use std::{collections::HashMap, path::Path};

use ortts_shared::AppError;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
  pub tts_bos_token_id: i64,
  pub tts_eos_token_id: i64,
  pub tts_pad_token_id: i64,
  pub talker_config: TalkerConfig,
}

#[derive(Debug, Deserialize)]
pub struct TalkerConfig {
  pub hidden_size: usize,
  pub vocab_size: usize,
  pub codec_eos_token_id: usize,
  pub codec_pad_id: i64,
  pub codec_bos_id: i64,
  pub codec_nothink_id: i64,
  pub codec_think_id: i64,
  pub codec_think_bos_id: i64,
  pub codec_think_eos_id: i64,
  pub codec_language_id: HashMap<String, i64>,
  pub num_hidden_layers: usize,
  pub num_key_value_heads: usize,
  pub head_dim: usize,
}

impl ModelConfig {
  pub fn from_path(path: &Path) -> Result<Self, AppError> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
  }
}
