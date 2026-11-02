//! Port of `pypinyin/core.py`.
//!
//! # Subclassing `Pinyin`
//!
//! Python customises segmentation by subclassing `pypinyin.core.Pinyin` and
//! overriding `pre_seg`, `seg`, `get_seg` or `post_seg`. In the port the
//! methods live in the [`PinyinMethods`] trait, whose default methods are
//! the Python method bodies. [`Pinyin`] is the plain class, generic over its
//! converter. A Python subclass becomes a struct that implements
//! [`PinyinMethods`] and overrides what the subclass overrides.

use crate::constants::{RE_HANS, Style};
use crate::converter::{Converter, DefaultConverter, Errors, UltimateConverter};
use crate::seg::simpleseg::seg;

/// The `hans` argument: a string, or a list of strings that are already
/// segmented (`'你好吗'` or `['你好', '吗']`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Hans<'a> {
  /// A `str`.
  Str(&'a str),
  /// A `list` of `str`.
  List(Vec<&'a str>),
}

impl<'a> From<&'a str> for Hans<'a> {
  fn from(s: &'a str) -> Self {
    Self::Str(s)
  }
}

impl<'a> From<&'a String> for Hans<'a> {
  fn from(s: &'a String) -> Self {
    Self::Str(s)
  }
}

impl<'a> From<Vec<&'a str>> for Hans<'a> {
  fn from(list: Vec<&'a str>) -> Self {
    Self::List(list)
  }
}

impl<'a> From<&'a [&'a str]> for Hans<'a> {
  fn from(list: &'a [&'a str]) -> Self {
    Self::List(list.to_vec())
  }
}

impl<'a> From<&'a [String]> for Hans<'a> {
  fn from(list: &'a [String]) -> Self {
    Self::List(list.iter().map(String::as_str).collect())
  }
}

impl<'a> From<&'a Vec<String>> for Hans<'a> {
  fn from(list: &'a Vec<String>) -> Self {
    Self::List(list.iter().map(String::as_str).collect())
  }
}

/// The methods of `pypinyin.core.Pinyin`. The default methods are the Python
/// method bodies; implement the trait to port a subclass.
pub trait PinyinMethods {
  /// The converter type (`self._converter`).
  type Converter: Converter;

  /// `self._converter`.
  fn converter(&self) -> &Self::Converter;

  /// Port of `Pinyin.pinyin(hans, style=Style.TONE, heteronym=False,
  /// errors='default', strict=True)`: 将汉字转换为拼音，返回汉字的拼音列表.
  fn pinyin<'a>(
    &self,
    hans: impl Into<Hans<'a>>,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    // 对字符串进行分词处理
    let han_list: Vec<String> = match hans.into() {
      Hans::Str(s) => self.seg(s),
      Hans::List(list) => {
        // The `ToneSandhiMixin` half of the isinstance check is not
        // ported (tone_sandhi is left out).
        if self.converter().is_ultimate_converter() {
          let mut han_list = Vec::new();
          for h in list {
            if !RE_HANS.is_match(h) {
              han_list.extend(self.seg(h));
            } else {
              han_list.push(h.to_string());
            }
          }
          han_list
        } else {
          list.into_iter().flat_map(|x| self.seg(x)).collect()
        }
      }
    };

    let mut pys = Vec::new();
    for words in &han_list {
      pys.extend(
        self
          .converter()
          .convert(words, style, heteronym, errors, strict),
      );
    }
    pys
  }

  /// Port of `Pinyin.lazy_pinyin(hans, style=Style.NORMAL,
  /// errors='default', strict=True)`: 返回不包含多音字结果的拼音列表.
  fn lazy_pinyin<'a>(
    &self,
    hans: impl Into<Hans<'a>>,
    style: Style,
    errors: &Errors,
    strict: bool,
  ) -> Vec<String> {
    self
      .pinyin(hans, style, false, errors, strict)
      .into_iter()
      .flatten()
      .collect()
  }

  /// Port of `Pinyin.pre_seg`: `Some` is used as the segmentation result.
  fn pre_seg(&self, hans: &str) -> Option<Vec<String>> {
    let _ = hans;
    None
  }

  /// Port of `Pinyin.seg`: 对汉字进行分词.
  fn seg(&self, hans: &str) -> Vec<String> {
    let seg_data = match self.pre_seg(hans) {
      Some(pre_data) => pre_data,
      None => self.get_seg()(hans),
    };

    if let Some(post_data) = self.post_seg(hans, &seg_data) {
      return post_data;
    }

    seg_data
  }

  /// Port of `Pinyin.get_seg`: 获取分词函数.
  fn get_seg(&self) -> fn(&str) -> Vec<String> {
    seg
  }

  /// Port of `Pinyin.post_seg`: `Some` replaces the segmentation result.
  fn post_seg(&self, hans: &str, seg_data: &[String]) -> Option<Vec<String>> {
    let _ = (hans, seg_data);
    None
  }
}

/// Port of `pypinyin.core.Pinyin`.
#[derive(Clone, Debug, Default)]
pub struct Pinyin<C: Converter = DefaultConverter> {
  /// `self._converter`.
  pub _converter: C,
}

impl<C: Converter> Pinyin<C> {
  /// Port of `Pinyin.__init__(converter)`.
  pub fn new(converter: C) -> Self {
    Self {
      _converter: converter,
    }
  }
}

impl<C: Converter> PinyinMethods for Pinyin<C> {
  type Converter = C;

  fn converter(&self) -> &C {
    &self._converter
  }
}

/// Keyword arguments of [`pinyin`].
#[derive(Clone, Debug)]
pub struct PinyinOptions {
  /// `style`: 拼音风格.
  pub style: Style,
  /// `heteronym`: 是否启用多音字.
  pub heteronym: bool,
  /// `errors`: 如何处理没有拼音的字符.
  pub errors: Errors,
  /// `strict`: 是否严格遵照《汉语拼音方案》来处理声母和韵母.
  pub strict: bool,
  /// `v_to_u`: 无声调相关拼音风格下的结果是否使用 `ü` 代替原来的 `v`.
  pub v_to_u: bool,
  /// `neutral_tone_with_five`: 声调使用数字表示的相关拼音风格下的结果是否使用 5 标识轻声.
  pub neutral_tone_with_five: bool,
}

impl Default for PinyinOptions {
  fn default() -> Self {
    Self {
      style: Style::TONE,
      heteronym: false,
      errors: Errors::Default,
      strict: true,
      v_to_u: false,
      neutral_tone_with_five: false,
    }
  }
}

/// Port of `pypinyin.core.pinyin` (`pypinyin.pinyin`): 将汉字转换为拼音，返回汉字的拼音列表.
pub fn pinyin<'a>(hans: impl Into<Hans<'a>>, options: &PinyinOptions) -> Vec<Vec<String>> {
  let _pinyin = Pinyin::new(UltimateConverter::new(
    options.v_to_u,
    options.neutral_tone_with_five,
  ));
  _pinyin.pinyin(
    hans,
    options.style,
    options.heteronym,
    &options.errors,
    options.strict,
  )
}

/// Keyword arguments of [`lazy_pinyin`]. `tone_sandhi` is not ported.
#[derive(Clone, Debug)]
pub struct LazyPinyinOptions {
  /// `style`: 拼音风格.
  pub style: Style,
  /// `errors`: 如何处理没有拼音的字符.
  pub errors: Errors,
  /// `strict`: 是否严格遵照《汉语拼音方案》来处理声母和韵母.
  pub strict: bool,
  /// `v_to_u`: 无声调相关拼音风格下的结果是否使用 `ü` 代替原来的 `v`.
  pub v_to_u: bool,
  /// `neutral_tone_with_five`: 声调使用数字表示的相关拼音风格下的结果是否使用 5 标识轻声.
  pub neutral_tone_with_five: bool,
}

impl Default for LazyPinyinOptions {
  fn default() -> Self {
    Self {
      style: Style::NORMAL,
      errors: Errors::Default,
      strict: true,
      v_to_u: false,
      neutral_tone_with_five: false,
    }
  }
}

/// Port of `pypinyin.core.lazy_pinyin` (`pypinyin.lazy_pinyin`): 将汉字转换为拼音，返回不包含多音字结果的拼音列表.
pub fn lazy_pinyin<'a>(hans: impl Into<Hans<'a>>, options: &LazyPinyinOptions) -> Vec<String> {
  let _pinyin = Pinyin::new(UltimateConverter::new(
    options.v_to_u,
    options.neutral_tone_with_five,
  ));
  _pinyin.lazy_pinyin(hans, options.style, &options.errors, options.strict)
}
