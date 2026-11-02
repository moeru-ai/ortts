//! Port of the GPT-SoVITS text frontend (`GPT_SoVITS/text` and the text parts of
//! `GPT_SoVITS/TTS_infer_pack`) at upstream commit 04a44eb.
//!
//! Module and function names mirror the Python sources so each item can be
//! checked against its counterpart. `lib.rs` itself mirrors `text/__init__.py`.

pub mod cantonese;
pub mod chinese2;
pub mod cleaner;
pub mod en_normalization;
pub mod english;
pub mod g2pw;
pub mod japanese;
pub mod lang_segmenter;
pub mod symbols;
pub mod symbols2;
pub mod tone_sandhi;
pub mod tts_infer_pack;
pub mod zh_normalization;

pub use cleaner::Cleaner;
pub use tts_infer_pack::text_preprocessor::{
  BertFeatureModel, BertInputs, PreprocessResult, TextPreprocessor,
};

use std::collections::HashMap;
use std::fmt;
use std::sync::LazyLock;

/// Builds `{s: i for i, s in enumerate(symbols)}`. A later duplicate
/// overwrites the earlier index, as in a Python dict comprehension.
fn build_symbol_to_id(symbols: &[&'static str]) -> HashMap<&'static str, usize> {
  let mut map = HashMap::with_capacity(symbols.len());
  for (i, s) in symbols.iter().enumerate() {
    map.insert(*s, i);
  }
  map
}

static SYMBOL_TO_ID_V1: LazyLock<HashMap<&'static str, usize>> =
  LazyLock::new(|| build_symbol_to_id(&symbols::SYMBOLS));
static SYMBOL_TO_ID_V2: LazyLock<HashMap<&'static str, usize>> =
  LazyLock::new(|| build_symbol_to_id(&symbols2::SYMBOLS));

/// Port of `_symbol_to_id_v1` (`{s: i for i, s in enumerate(symbols_v1.symbols)}`).
pub fn symbol_to_id_v1() -> &'static HashMap<&'static str, usize> {
  &SYMBOL_TO_ID_V1
}

/// Port of `_symbol_to_id_v2` (`{s: i for i, s in enumerate(symbols_v2.symbols)}`).
pub fn symbol_to_id_v2() -> &'static HashMap<&'static str, usize> {
  &SYMBOL_TO_ID_V2
}

/// The `KeyError` that `cleaned_text_to_sequence` raises for a symbol that is
/// not in the table. Holds the missing symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyError(pub String);

impl fmt::Display for KeyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "KeyError: {:?}", self.0)
  }
}

impl std::error::Error for KeyError {}

/// Port of `cleaned_text_to_sequence(cleaned_text, version=None)`.
///
/// `version = None` reads the `version` environment variable (default
/// `"v2"`). `"v1"` selects the v1 table; every other value selects the v2
/// table. An unknown symbol gives `KeyError`, as the Python dict lookup does.
pub fn cleaned_text_to_sequence<S: AsRef<str>>(
  cleaned_text: &[S],
  version: Option<&str>,
) -> Result<Vec<usize>, KeyError> {
  let env_version;
  let version = match version {
    Some(v) => v,
    None => {
      env_version = std::env::var("version").unwrap_or_else(|_| "v2".to_string());
      env_version.as_str()
    }
  };
  let table = if version == "v1" {
    symbol_to_id_v1()
  } else {
    symbol_to_id_v2()
  };
  cleaned_text
    .iter()
    .map(|symbol| {
      let symbol = symbol.as_ref();
      table
        .get(symbol)
        .copied()
        .ok_or_else(|| KeyError(symbol.to_string()))
    })
    .collect()
}
