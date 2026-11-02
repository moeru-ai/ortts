//! Subset port of the Python split-lang 2.1.1 package used by GPT-SoVITS.
//!
//! Module layout mirrors `split_lang/`: [`config`], [`model`], [`split`]
//! (`splitter.py`, `utils.py`) and [`detect_lang`] (`detector.py`). The
//! [`budoux`] module ports the part of budoux 0.10.2 that split-lang calls
//! (`budoux.Parser.parse` with the default Japanese and Simplified Chinese
//! models).
//!
//! Language detection goes through the [`fast_langdetect`] crate. As in
//! Python, the detector functions use `fast_langdetect`'s module-level
//! default detector unless a [`LangSplitter`] is given its own detector
//! (the `detector` field of [`LangSplitter`]).

pub mod budoux;
pub mod config;
pub mod detect_lang;
pub mod model;
mod py;
pub mod split;

pub use config::{DEFAULT_LANG, DEFAULT_LANG_MAP};
pub use model::{LangSectionType, SubString, SubStringSection};
pub use split::splitter::LangSplitter;

use fast_langdetect::FastLangdetectError;

/// The Python exceptions that `LangSplitter.split_by_lang` can raise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitLangError {
  /// `IndexError` (for example `sections[0]` when the stripped text is empty).
  IndexError(String),
  /// `KeyError` from a dict lookup (holds the missing key).
  KeyError(String),
  /// An exception raised by `fast_langdetect.detect`.
  FastLangdetect(FastLangdetectError),
}

impl SplitLangError {
  /// The Python exception class name (`"IndexError"`, `"KeyError"`, ...).
  pub fn exception_name(&self) -> &'static str {
    match self {
      Self::IndexError(_) => "IndexError",
      Self::KeyError(_) => "KeyError",
      Self::FastLangdetect(FastLangdetectError::FileNotFound(_)) => "FileNotFoundError",
      Self::FastLangdetect(FastLangdetectError::ModelLoad(_)) => "ModelLoadError",
      Self::FastLangdetect(FastLangdetectError::InvalidArgument(_)) => "ValueError",
      Self::FastLangdetect(FastLangdetectError::Unsupported(_)) => "NotImplementedError",
    }
  }
}

impl std::fmt::Display for SplitLangError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::IndexError(m) => write!(f, "IndexError: {m}"),
      Self::KeyError(k) => write!(f, "KeyError: {k:?}"),
      Self::FastLangdetect(e) => write!(f, "{e}"),
    }
  }
}

impl std::error::Error for SplitLangError {}

impl From<FastLangdetectError> for SplitLangError {
  fn from(e: FastLangdetectError) -> Self {
    Self::FastLangdetect(e)
  }
}

/// `IndexError('list index out of range')`.
pub(crate) fn index_error() -> SplitLangError {
  SplitLangError::IndexError("list index out of range".into())
}
