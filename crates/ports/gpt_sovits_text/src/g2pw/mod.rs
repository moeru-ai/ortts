//! Port of `GPT_SoVITS/text/g2pw` (GPT-SoVITS @ 04a44eb).
//!
//! `__init__.py` is `from text.g2pw.g2pw import *`, so the public items of
//! [`g2pw`] are re-exported here.
//!
//! Credits: the package is modified from PaddleSpeech
//! (`paddlespeech/t2s/frontend/g2pw`, Apache-2.0), which is modified from
//! <https://github.com/GitYCC/g2pW>, and from
//! <https://github.com/mozillazg/pypinyin-g2pW> (`g2pw.py`).

pub mod dataset;
#[allow(clippy::module_inception)]
pub mod g2pw;
pub mod onnx_api;
pub mod opencc;
pub mod utils;

use std::fmt;

pub use self::g2pw::{Converter, G2PWPinyin, correct_pronunciation};

/// The Python exceptions that the g2pw code raises, plus the I/O, ONNX
/// Runtime and tokenizer failures of the port.
#[derive(Debug)]
pub enum G2PWError {
  /// `AssertionError`.
  AssertionError(String),
  /// `IndexError`.
  IndexError(String),
  /// `KeyError` (holds the key).
  KeyError(String),
  /// `ValueError`.
  ValueError(String),
  /// `TypeError`.
  TypeError(String),
  /// `FileNotFoundError` (also other errors reading a file).
  FileNotFoundError(String),
  /// `json.JSONDecodeError`.
  Json(String),
  /// A tokenizer error (`tokenizers`).
  Tokenizer(String),
  /// An ONNX Runtime error (`ort`).
  Ort(ort::Error),
}

impl fmt::Display for G2PWError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      G2PWError::AssertionError(m) => write!(f, "AssertionError: {m}"),
      G2PWError::IndexError(m) => write!(f, "IndexError: {m}"),
      G2PWError::KeyError(m) => write!(f, "KeyError: {m:?}"),
      G2PWError::ValueError(m) => write!(f, "ValueError: {m}"),
      G2PWError::TypeError(m) => write!(f, "TypeError: {m}"),
      G2PWError::FileNotFoundError(m) => write!(f, "FileNotFoundError: {m}"),
      G2PWError::Json(m) => write!(f, "JSONDecodeError: {m}"),
      G2PWError::Tokenizer(m) => write!(f, "tokenizer error: {m}"),
      G2PWError::Ort(e) => write!(f, "onnxruntime error: {e}"),
    }
  }
}

impl G2PWError {
  /// The name of the Python exception (`type(e).__name__`) for the
  /// variants that mirror one; `None` for the port's own failures.
  pub fn exception_name(&self) -> Option<&'static str> {
    match self {
      G2PWError::AssertionError(_) => Some("AssertionError"),
      G2PWError::IndexError(_) => Some("IndexError"),
      G2PWError::KeyError(_) => Some("KeyError"),
      G2PWError::ValueError(_) => Some("ValueError"),
      G2PWError::TypeError(_) => Some("TypeError"),
      G2PWError::FileNotFoundError(_) => Some("FileNotFoundError"),
      G2PWError::Json(_) => Some("JSONDecodeError"),
      G2PWError::Tokenizer(_) | G2PWError::Ort(_) => None,
    }
  }
}

impl std::error::Error for G2PWError {}

impl From<ort::Error> for G2PWError {
  fn from(e: ort::Error) -> Self {
    G2PWError::Ort(e)
  }
}
