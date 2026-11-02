//! Subset port of the Python pypinyin 0.55.0 package used by GPT-SoVITS.
//!
//! Port of `pypinyin/__init__.py`: the package root re-exports [`pinyin`],
//! [`lazy_pinyin`] and [`Style`].
//!
//! The module layout mirrors the Python package (`pypinyin.core` is
//! [`core`], `pypinyin.contrib.tone_convert` is [`contrib::tone_convert`] and
//! so on). See `parity/README.md` for the ported scope and for how to express
//! the Python subclassing extension points (`Converter`, `Pinyin`).

pub mod _re_digit;
pub mod compat;
pub mod constants;
pub mod contrib;
pub mod converter;
pub mod core;
pub mod phonetic_symbol;
pub mod seg;
pub mod standard;
pub mod style;
pub mod utils;

pub use crate::constants::Style;
pub use crate::converter::{
  Converter, DefaultConverter, Errors, NoPinyinResult, UltimateConverter,
};
pub use crate::core::{
  Hans, LazyPinyinOptions, Pinyin, PinyinMethods, PinyinOptions, lazy_pinyin, pinyin,
};
