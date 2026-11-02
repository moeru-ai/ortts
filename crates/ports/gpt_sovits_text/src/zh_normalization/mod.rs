//! Port of `GPT_SoVITS/text/zh_normalization` (GPT-SoVITS @ 04a44eb, originally PaddleSpeech).
//!
//! `__init__.py` re-exports `text_normlization`, so this module re-exports
//! [`text_normlization::TextNormalizer`].
//!
//! # Ported scope
//!
//! Everything in the package: `char_convert.py` (both conversion directions and
//! the two character tables, kept verbatim in `data/zh_normalization/`),
//! `chronology.py`, `constants.py` (all translation tables and `RE_NSW`),
//! `num.py`, `phonecode.py`, `quantifier.py` and `text_normlization.py`
//! (`TextNormalizer` with `_split`, `_post_replace`, `normalize_sentence`,
//! `normalize`). Left out: the `if __name__ == "__main__"` demo in
//! `char_convert.py`.
//!
//! # Differences in form (not in output)
//!
//! - Python exceptions become [`Error`] values: `normalize` and the rule
//!   functions return `Result`. `KeyError` happens in Python when a non-ASCII
//!   decimal digit (for example `٣`) reaches `DIGITS`; the fixtures record these.
//! - Regex sources are kept as in Python and compiled through [`pyre`], which
//!   maps `\d` to the exact Python 3.11 (Unicode 14.0.0) digit set, unescapes
//!   `\×`, and escapes `[`, `&`, `~` inside classes. `regex` is used unless the
//!   pattern has lookarounds or backreferences (`RE_RANGE`, `RE_DATE2`,
//!   `RE_MOBILE_PHONE`, `RE_TELEPHONE`), which use `fancy-regex`.
//! - `F2H_SPACE` / `H2F_SPACE` are keyed by `str` upstream, so `str.translate`
//!   never applies them; [`constants::translate_str_keyed`] keeps that no-op.
//!
//! # Parity
//!
//! `tests/zh_normalization.rs` checks the fixtures from
//! `parity/zh_normalization/gen_fixtures.py` for exact
//! equality, including the exception cases. There are no known mismatches.

pub mod char_convert;
pub mod chronology;
pub mod constants;
pub mod num;
pub mod phonecode;
pub mod pyre;
pub mod quantifier;
pub mod text_normlization;

pub use text_normlization::TextNormalizer;

/// A Python exception raised by the ported code.
///
/// The Python functions raise on some inputs, for example `KeyError` when a
/// non-ASCII decimal digit (`\d` matches all Unicode `Nd` characters) reaches
/// the `DIGITS` table of `num.py`. The port returns the same exception as an
/// error instead of producing output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
  /// `KeyError(key)`.
  KeyError(String),
  /// `ValueError(message)`.
  ValueError(String),
}

impl std::fmt::Display for Error {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Error::KeyError(key) => write!(f, "KeyError: {key:?}"),
      Error::ValueError(msg) => write!(f, "ValueError: {msg}"),
    }
  }
}

impl std::error::Error for Error {}
