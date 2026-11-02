//! Port of `pypinyin/converter.py`.
//!
//! # Subclassing
//!
//! Python customises the conversion by subclassing `DefaultConverter` or
//! `UltimateConverter` and overriding methods. In the port:
//!
//! - [`Converter`] is a trait with one method per Python method. Every
//!   method has a default body that does what `DefaultConverter` does.
//! - The `DefaultConverter` bodies also exist as free functions in
//!   [`default_converter`], generic over the converter. They are the
//!   `super()` calls: `super().convert(...)` in a subclass of
//!   `DefaultConverter` (or of `UltimateConverter`, which does not override
//!   `convert`) is `default_converter::convert(self, ...)`. Methods called
//!   from inside those bodies still dispatch on `self`, as in Python.
//! - [`UltimateConverter`] overrides `post_convert_style` and `post_pinyin`.
//!   Its overrides do not call back into `self`, so a subclass of
//!   `UltimateConverter` holds an `UltimateConverter` and forwards those two
//!   methods (and [`Converter::is_ultimate_converter`]) to it.
//!
//! See `parity/README.md` for a full example (the GPT-SoVITS G2PW converter).

use std::fmt;
use std::sync::Arc;

use crate::constants::{PHRASES_DICT, PINYIN_DICT, RE_HANS, Style};
use crate::contrib::neutral_tone::NeutralToneWith5Mixin;
use crate::contrib::uv::V2UMixin;
use crate::style::convert as convert_style;
use crate::utils::_remove_dup_and_empty;

/// A value that Python's no-pinyin handling passes around (`py` in
/// `DefaultConverter.handle_nopinyin`): a string, a list of strings, or a
/// list of pinyin lists. Python's `None` is `Option::None` around it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoPinyinResult {
  /// A `str`.
  Str(String),
  /// A `list` of `str`.
  List(Vec<String>),
  /// A `list` of `list` of `str` (includes heteronym information).
  Nested(Vec<Vec<String>>),
}

impl NoPinyinResult {
  /// Python truthiness of the value.
  fn is_truthy(&self) -> bool {
    match self {
      Self::Str(s) => !s.is_empty(),
      Self::List(l) => !l.is_empty(),
      Self::Nested(l) => !l.is_empty(),
    }
  }
}

/// Callback type for [`Errors::Callable`].
pub type ErrorsCallback = Arc<dyn Fn(&str) -> Option<NoPinyinResult> + Send + Sync>;

/// The `errors` argument: how to handle characters without pinyin.
///
/// `'exception'` (raise `PinyinNotFoundException`) is not ported.
#[derive(Clone, Default)]
pub enum Errors {
  /// `'default'`: 保留原始字符.
  #[default]
  Default,
  /// `'ignore'`: 忽略该字符.
  Ignore,
  /// `'replace'`: 替换为去掉 `\u` 的 unicode 编码字符串.
  Replace,
  /// A callable: its return value is the result.
  Callable(ErrorsCallback),
}

impl Errors {
  /// Build [`Errors::Callable`] from a closure.
  pub fn callable(f: impl Fn(&str) -> Option<NoPinyinResult> + Send + Sync + 'static) -> Self {
    Self::Callable(Arc::new(f))
  }
}

impl fmt::Debug for Errors {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Default => f.write_str("'default'"),
      Self::Ignore => f.write_str("'ignore'"),
      Self::Replace => f.write_str("'replace'"),
      Self::Callable(_) => f.write_str("<callable>"),
    }
  }
}

/// Port of `pypinyin.converter.Converter` together with the method bodies of
/// `pypinyin.converter.DefaultConverter`, which are the default methods.
///
/// Override a method to customise it, like a Python subclass. The
/// `DefaultConverter` bodies are available as functions in
/// [`default_converter`] for `super()` calls.
pub trait Converter {
  /// Port of `DefaultConverter.convert`: 根据参数把汉字转成相应风格的拼音结果.
  fn convert(
    &self,
    words: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    default_converter::convert(self, words, style, heteronym, errors, strict)
  }

  /// Port of `DefaultConverter.pre_convert_style`. `Some` replaces
  /// `orig_pinyin` before the style conversion.
  fn pre_convert_style(
    &self,
    han: &str,
    orig_pinyin: &str,
    style: Style,
    strict: bool,
  ) -> Option<String> {
    let _ = (han, orig_pinyin, style, strict);
    None
  }

  /// Port of `DefaultConverter.convert_style`.
  fn convert_style(&self, han: &str, orig_pinyin: &str, style: Style, strict: bool) -> String {
    default_converter::convert_style(self, han, orig_pinyin, style, strict)
  }

  /// Port of `DefaultConverter.post_convert_style`. `Some` replaces the
  /// converted pinyin.
  fn post_convert_style(
    &self,
    han: &str,
    orig_pinyin: &str,
    converted_pinyin: &str,
    style: Style,
    strict: bool,
  ) -> Option<String> {
    let _ = (han, orig_pinyin, converted_pinyin, style, strict);
    None
  }

  /// Port of `DefaultConverter.pre_handle_nopinyin`. `Some` is used as the
  /// result instead of the built-in handling.
  fn pre_handle_nopinyin(
    &self,
    chars: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Option<NoPinyinResult> {
    let _ = (chars, style, heteronym, errors, strict);
    None
  }

  /// Port of `DefaultConverter.handle_nopinyin`: 处理没有拼音的字符串.
  fn handle_nopinyin(
    &self,
    chars: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    default_converter::handle_nopinyin(self, chars, style, heteronym, errors, strict)
  }

  /// Port of `DefaultConverter.post_handle_nopinyin`. `pinyin` is the
  /// result so far (`None` when there is none). `Some` replaces it.
  #[allow(clippy::too_many_arguments)]
  fn post_handle_nopinyin(
    &self,
    chars: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
    pinyin: Option<&NoPinyinResult>,
  ) -> Option<NoPinyinResult> {
    let _ = (chars, style, heteronym, errors, strict, pinyin);
    None
  }

  /// Port of `DefaultConverter.post_pinyin`: called with the raw pinyin of
  /// `han` (a character or a phrase). `Some` replaces it.
  fn post_pinyin(
    &self,
    han: &str,
    heteronym: bool,
    pinyin: &[Vec<String>],
  ) -> Option<Vec<Vec<String>>> {
    let _ = (han, heteronym, pinyin);
    None
  }

  /// Port of `DefaultConverter._phrase_pinyin`: 词语拼音转换.
  fn _phrase_pinyin(
    &self,
    phrase: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    default_converter::_phrase_pinyin(self, phrase, style, heteronym, errors, strict)
  }

  /// Port of `DefaultConverter.convert_styles`: 转换多个汉字的拼音结果的风格.
  fn convert_styles(
    &self,
    pinyin_list: Vec<Vec<String>>,
    phrase: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    default_converter::convert_styles(self, pinyin_list, phrase, style, heteronym, errors, strict)
  }

  /// Port of `DefaultConverter._single_pinyin`: 单字拼音转换.
  fn _single_pinyin(
    &self,
    han: char,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    default_converter::_single_pinyin(self, han, style, heteronym, errors, strict)
  }

  /// Port of `DefaultConverter._convert_style`.
  fn _convert_style(
    &self,
    han: &str,
    pinyin: &str,
    style: Style,
    strict: bool,
    default: &str,
  ) -> String {
    convert_style(pinyin, style, strict, default, han)
  }

  /// Port of `DefaultConverter._convert_nopinyin_chars`: 转换没有拼音的字符.
  fn _convert_nopinyin_chars(
    &self,
    chars: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Option<NoPinyinResult> {
    default_converter::_convert_nopinyin_chars(self, chars, style, heteronym, errors, strict)
  }

  /// `isinstance(converter, UltimateConverter)`, which
  /// `pypinyin.core.Pinyin.pinyin` checks for list input. `true` for
  /// [`UltimateConverter`] and for converters that port its subclasses.
  fn is_ultimate_converter(&self) -> bool {
    false
  }
}

/// The method bodies of `pypinyin.converter.DefaultConverter`, generic over
/// the converter so that overrides can call them as `super()`.
pub mod default_converter {
  use super::*;

  /// Port of `DefaultConverter.convert`.
  pub fn convert<C: Converter + ?Sized>(
    this: &C,
    words: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    let mut pys = Vec::new();
    // 初步过滤没有拼音的字符
    if RE_HANS.is_match(words) {
      pys = this._phrase_pinyin(words, style, heteronym, errors, strict);
      if let Some(post_data) = this.post_pinyin(words, heteronym, &pys) {
        pys = post_data;
      }

      pys = this.convert_styles(pys, words, style, heteronym, errors, strict);
    } else {
      let py = this.handle_nopinyin(words, style, heteronym, errors, strict);
      if !py.is_empty() {
        pys.extend(py);
      }
    }

    _remove_dup_and_empty(&pys)
  }

  /// Port of `DefaultConverter.convert_style`.
  pub fn convert_style<C: Converter + ?Sized>(
    this: &C,
    han: &str,
    orig_pinyin: &str,
    style: Style,
    strict: bool,
  ) -> String {
    let pinyin = match this.pre_convert_style(han, orig_pinyin, style, strict) {
      Some(pre_data) => pre_data,
      None => orig_pinyin.to_string(),
    };

    let converted_pinyin = this._convert_style(han, &pinyin, style, strict, &pinyin);

    match this.post_convert_style(han, &pinyin, &converted_pinyin, style, strict) {
      Some(post_data) => post_data,
      None => converted_pinyin,
    }
  }

  /// Port of `DefaultConverter.handle_nopinyin`.
  ///
  /// Panics when a nested result has an empty item list and `heteronym` is
  /// off, like the `IndexError` of `x[0]` in Python.
  pub fn handle_nopinyin<C: Converter + ?Sized>(
    this: &C,
    chars: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    let mut py = match this.pre_handle_nopinyin(chars, style, heteronym, errors, strict) {
      Some(pre_data) => Some(pre_data),
      None => this._convert_nopinyin_chars(chars, style, heteronym, errors, strict),
    };

    if let Some(post_data) =
      this.post_handle_nopinyin(chars, style, heteronym, errors, strict, py.as_ref())
    {
      py = Some(post_data);
    }

    let Some(py) = py.filter(NoPinyinResult::is_truthy) else {
      return Vec::new();
    };
    match py {
      // 包含多音字信息
      NoPinyinResult::Nested(py) => {
        if heteronym {
          return py;
        }
        // [[a, b], [c, d]]
        // [[a], [c]]
        py.into_iter().map(|x| vec![x[0].clone()]).collect()
      }
      NoPinyinResult::List(py) => py.into_iter().map(|i| vec![i]).collect(),
      NoPinyinResult::Str(py) => vec![vec![py]],
    }
  }

  /// Port of `DefaultConverter._phrase_pinyin`.
  pub fn _phrase_pinyin<C: Converter + ?Sized>(
    this: &C,
    phrase: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    let mut pinyin_list = Vec::new();
    if let Some(pys) = PHRASES_DICT.get(phrase) {
      pinyin_list = pys.clone();
    } else {
      for han in phrase.chars() {
        let py = this._single_pinyin(han, style, heteronym, errors, strict);
        pinyin_list.extend(py);
      }
    }

    pinyin_list
  }

  /// Port of `DefaultConverter.convert_styles`.
  ///
  /// Panics when `pinyin_list` is longer than `phrase`, like the
  /// `IndexError` of `phrase[idx]` in Python.
  pub fn convert_styles<C: Converter + ?Sized>(
    this: &C,
    mut pinyin_list: Vec<Vec<String>>,
    phrase: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    let _ = errors;
    let phrase: Vec<char> = phrase.chars().collect();
    for (idx, item) in pinyin_list.iter_mut().enumerate() {
      let han = phrase[idx].to_string();
      if heteronym {
        *item = item
          .iter()
          .map(|x| this.convert_style(&han, x, style, strict))
          .collect();
      } else {
        let orig_pinyin = &item[0];
        *item = vec![this.convert_style(&han, orig_pinyin, style, strict)];
      }
    }

    pinyin_list
  }

  /// Port of `DefaultConverter._single_pinyin`.
  pub fn _single_pinyin<C: Converter + ?Sized>(
    this: &C,
    han: char,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    let num = han as u32;
    // 处理没有拼音的字符
    let Some(pys) = PINYIN_DICT.get(&num) else {
      return this.handle_nopinyin(&han.to_string(), style, heteronym, errors, strict);
    };

    // 字的拼音列表
    vec![pys.split(',').map(str::to_string).collect()]
  }

  /// Port of `DefaultConverter._convert_nopinyin_chars`.
  pub fn _convert_nopinyin_chars<C: Converter + ?Sized>(
    this: &C,
    chars: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Option<NoPinyinResult> {
    let _ = (this, style, heteronym, strict);
    match errors {
      Errors::Callable(f) => f(chars),
      Errors::Default => Some(NoPinyinResult::Str(chars.to_string())),
      Errors::Ignore => None,
      Errors::Replace => {
        // Both Python branches ('%x' % ord(x) per character) give the same
        // result, except that `ord('')` raises for an empty string.
        assert!(
          !chars.is_empty(),
          "TypeError: ord() expected a character, but string of length 0 found"
        );
        Some(NoPinyinResult::Str(
          chars.chars().map(|x| format!("{:x}", x as u32)).collect(),
        ))
      }
    }
  }
}

/// Port of `pypinyin.converter.DefaultConverter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultConverter;

impl DefaultConverter {
  /// Port of `DefaultConverter.__init__`.
  pub fn new() -> Self {
    Self
  }
}

impl Converter for DefaultConverter {}

/// Port of `pypinyin.converter.UltimateConverter`.
///
/// `tone_sandhi` is not ported (see the crate README).
#[derive(Clone, Copy, Debug, Default)]
pub struct UltimateConverter {
  /// `self._v_to_u`.
  pub _v_to_u: bool,
  /// `self._neutral_tone_with_five`.
  pub _neutral_tone_with_five: bool,
}

impl UltimateConverter {
  /// Port of `UltimateConverter.__init__(v_to_u=False, neutral_tone_with_five=False)`.
  pub fn new(v_to_u: bool, neutral_tone_with_five: bool) -> Self {
    Self {
      _v_to_u: v_to_u,
      _neutral_tone_with_five: neutral_tone_with_five,
    }
  }
}

impl Converter for UltimateConverter {
  /// Port of `UltimateConverter.post_convert_style`.
  fn post_convert_style(
    &self,
    han: &str,
    orig_pinyin: &str,
    converted_pinyin: &str,
    style: Style,
    strict: bool,
  ) -> Option<String> {
    // super().post_convert_style() is DefaultConverter's, which returns None.
    let mut converted_pinyin = converted_pinyin.to_string();

    if self._v_to_u
      && let Some(post_data) =
        V2UMixin.post_convert_style(han, orig_pinyin, &converted_pinyin, style, strict)
    {
      converted_pinyin = post_data;
    }

    if self._neutral_tone_with_five
      && let Some(post_data) =
        NeutralToneWith5Mixin.post_convert_style(han, orig_pinyin, &converted_pinyin, style, strict)
    {
      converted_pinyin = post_data;
    }

    Some(converted_pinyin)
  }

  /// Port of `UltimateConverter.post_pinyin`.
  fn post_pinyin(
    &self,
    han: &str,
    heteronym: bool,
    pinyin: &[Vec<String>],
  ) -> Option<Vec<Vec<String>>> {
    // super().post_pinyin() is DefaultConverter's, which returns None.
    // The tone_sandhi branch is not ported.
    let _ = (han, heteronym);
    Some(pinyin.to_vec())
  }

  fn is_ultimate_converter(&self) -> bool {
    true
  }
}
