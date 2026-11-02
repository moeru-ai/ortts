//! Subset port of the Python fast-langdetect 1.1.0 package (fasttext lid.176
//! inference) used by GPT-SoVITS, including a pure Rust port of the
//! fasttext-predict 0.9.2.4 prediction path ([`fasttext`]).

pub mod fasttext;
pub mod infer;

pub use infer::{
  DetectOptions, DetectResult, LangDetectConfig, LangDetector, Model, detect, set_default_detector,
};

/// Port of `FastLangdetectError` and its subclass `ModelLoadError`, plus the
/// standard Python exceptions that `detect()` lets propagate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FastLangdetectError {
  /// `FileNotFoundError` (missing model file or `custom_model_path`).
  FileNotFound(String),
  /// `ModelLoadError`.
  ModelLoad(String),
  /// `ValueError` from fastText (`std::invalid_argument`, e.g. `k <= 0`, or a
  /// newline passed to `_FastText.predict`).
  InvalidArgument(String),
  /// A code path of the Python package that this port leaves out.
  Unsupported(String),
}

impl FastLangdetectError {
  /// The error message.
  pub fn message(&self) -> &str {
    match self {
      Self::FileNotFound(m)
      | Self::ModelLoad(m)
      | Self::InvalidArgument(m)
      | Self::Unsupported(m) => m,
    }
  }
}

impl std::fmt::Display for FastLangdetectError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.message())
  }
}

impl std::error::Error for FastLangdetectError {}
