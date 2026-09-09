use std::path::Path;

use anyhow::anyhow;
use ortts_shared::AppError;
use tokenizers::{Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

const IM_START_TOKEN_ID: i64 = 151_644;
const IM_END_TOKEN_ID: i64 = 151_645;

pub fn load(vocab: &Path, merges: &Path) -> Result<Tokenizer, AppError> {
  let vocab = vocab
    .to_str()
    .ok_or_else(|| anyhow!("Qwen tokenizer vocabulary path is not valid UTF-8"))?;
  let merges = merges
    .to_str()
    .ok_or_else(|| anyhow!("Qwen tokenizer merges path is not valid UTF-8"))?;
  let model = BPE::from_file(vocab, merges)
    .build()
    .map_err(|error| anyhow!(error))?;
  let mut tokenizer = Tokenizer::new(model);
  tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
  Ok(tokenizer)
}

pub fn assistant_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>, AppError> {
  let assistant_and_text = tokenizer
    .encode(format!("assistant\n{text}"), false)
    .map_err(|error| anyhow!(error))?;
  let assistant_suffix = tokenizer
    .encode("assistant\n", false)
    .map_err(|error| anyhow!(error))?;
  let newline = tokenizer
    .encode("\n", false)
    .map_err(|error| anyhow!(error))?;

  let mut ids =
    Vec::with_capacity(assistant_and_text.len() + assistant_suffix.len() + newline.len() + 3);
  ids.push(IM_START_TOKEN_ID);
  ids.extend(assistant_and_text.get_ids().iter().map(|&id| i64::from(id)));
  ids.push(IM_END_TOKEN_ID);
  ids.extend(newline.get_ids().iter().map(|&id| i64::from(id)));
  ids.push(IM_START_TOKEN_ID);
  ids.extend(assistant_suffix.get_ids().iter().map(|&id| i64::from(id)));
  Ok(ids)
}

#[cfg(test)]
mod tests {
  use super::{assistant_ids, load};
  use std::path::PathBuf;

  fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("tests")
      .join("fixtures")
      .join(name)
  }

  #[test]
  #[ignore = "requires tokenizer fixtures downloaded from the Qwen model repository"]
  fn matches_qwen2_tokenizer_for_supported_languages() {
    let tokenizer = load(&fixture("vocab.json"), &fixture("merges.txt")).unwrap();
    assert_eq!(
      assistant_ids(&tokenizer, "Hello world!").unwrap(),
      vec![
        151_644, 77_091, 198, 9_707, 1_879, 0, 151_645, 198, 151_644, 77_091, 198
      ]
    );
    assert_eq!(
      assistant_ids(&tokenizer, "你好，世界！").unwrap(),
      vec![
        151_644, 77_091, 198, 108_386, 3_837, 99_489, 6_313, 151_645, 198, 151_644, 77_091, 198
      ]
    );
    assert_eq!(
      assistant_ids(&tokenizer, "こんにちは、世界！").unwrap(),
      vec![
        151_644, 77_091, 198, 89_015, 5_373, 99_489, 6_313, 151_645, 198, 151_644, 77_091, 198
      ]
    );
  }
}
