//! Port of `GPT_SoVITS/text/en_normalization` (GPT-SoVITS @ 04a44eb).
//!
//! `expend.py` is the only file of the package. [`pyre`] and the generated
//! `py_tables` have no Python counterpart: they reproduce the Python `re`
//! character classes and `int()` that `expend.py` relies on.

pub mod expend;
mod py_tables;
pub mod pyre;

pub use expend::normalize;

use std::fmt;

/// An exception raised by the ported code. The variants are the Python
/// exception classes that `normalize` can raise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
  /// `ValueError(message)` from `int()`, for example `int(",5")` for
  /// `"$,5"`.
  ValueError(String),
  /// An exception raised by `inflect`, such as `NumOutOfRangeError` for a
  /// number above the decillions. Its message is always empty.
  Inflect(inflect::Error),
}

impl Error {
  /// The Python exception class name.
  pub fn name(&self) -> &'static str {
    match self {
      Error::ValueError(_) => "ValueError",
      Error::Inflect(e) => e.name(),
    }
  }

  /// `str(exception)`.
  pub fn message(&self) -> &str {
    match self {
      Error::ValueError(m) => m,
      Error::Inflect(_) => "",
    }
  }
}

impl From<inflect::Error> for Error {
  fn from(e: inflect::Error) -> Self {
    Error::Inflect(e)
  }
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}: {}", self.name(), self.message())
  }
}

impl std::error::Error for Error {}
