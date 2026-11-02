//! Subset port of the Python cn2an 0.5.24 package used by GPT-SoVITS.
//!
//! GPT-SoVITS calls `cn2an.transform(text, "an2cn")`. This crate ports that
//! path: [`transform::Transform`] with method `"an2cn"` (and its `direct`
//! flag), [`an2cn::An2Cn`] with all four modes, the constants from
//! `cn2an/conf.py` that they use, and the two `proces.preprocess` pipelines
//! that `An2Cn.an2cn` runs. The `"cn2an"` direction is not ported.

pub mod an2cn;
pub mod conf;
pub mod proces;
mod py_re;
pub mod transform;

pub use an2cn::An2Cn;
pub use transform::Transform;

use std::sync::OnceLock;

/// Errors raised by the ported code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
  /// A Python `ValueError` (or another exception that Python raises for
  /// the same input, such as a failing `int()` call).
  ValueError(String),
  /// The requested behaviour exists in Python but is not ported.
  NotPorted(String),
}

impl std::fmt::Display for Error {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Error::ValueError(msg) => write!(f, "ValueError: {msg}"),
      Error::NotPorted(msg) => write!(f, "not ported: {msg}"),
    }
  }
}

impl std::error::Error for Error {}

/// Result type of the ported functions.
pub type Result<T> = std::result::Result<T, Error>;

fn default_an2cn() -> &'static An2Cn {
  static AN2CN: OnceLock<An2Cn> = OnceLock::new();
  AN2CN.get_or_init(An2Cn::new)
}

fn default_transform() -> &'static Transform {
  static TRANSFORM: OnceLock<Transform> = OnceLock::new();
  TRANSFORM.get_or_init(Transform::new)
}

/// Port of `cn2an.an2cn` (`An2Cn().an2cn` in `cn2an/__init__.py`).
pub fn an2cn(inputs: &str, mode: &str) -> Result<String> {
  default_an2cn().an2cn(inputs, mode)
}

/// Port of `cn2an.transform` (`Transform().transform` in `cn2an/__init__.py`).
pub fn transform(inputs: &str, method: &str, direct: bool) -> Result<String> {
  default_transform().transform(inputs, method, direct)
}
