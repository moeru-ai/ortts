//! Port of `GPT_SoVITS/text/g2pw/g2pw.py` (GPT-SoVITS @ 04a44eb).
//!
//! Credits: this code is modified from
//! <https://github.com/mozillazg/pypinyin-g2pW> (MIT).
//!
//! `G2PWPinyin` and `Converter` subclass `pypinyin.core.Pinyin` and
//! `pypinyin.converter.UltimateConverter`; the port uses the extension
//! points of the `pypinyin` crate (see `crates/ports/pypinyin/README.md`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use pypinyin::constants::RE_HANS;
use pypinyin::contrib::tone_convert::to_tone;
use pypinyin::converter::default_converter;
use pypinyin::seg::simpleseg::simple_seg;
use pypinyin::{Errors, PinyinMethods, Style, UltimateConverter};

use super::G2PWError;
use super::onnx_api::{G2PWOnnxConverter, G2PWOnnxConverterOptions, G2PWStyle};

/// `polyphonic.rep` (`PP_DICT_PATH`), embedded.
pub const POLYPHONIC_REP: &str = include_str!("../../data/g2pw/polyphonic.rep");
/// `polyphonic-fix.rep` (`PP_FIX_DICT_PATH`), embedded.
pub const POLYPHONIC_FIX_REP: &str = include_str!("../../data/g2pw/polyphonic-fix.rep");

/// Keyword arguments of `G2PWPinyin(model_dir, model_source=None,
/// enable_non_tradional_chinese=True, v_to_u=False,
/// neutral_tone_with_five=False, tone_sandhi=False)`.
///
/// `tone_sandhi` is not ported (the `pypinyin` crate leaves it out and
/// GPT-SoVITS does not set it).
#[derive(Clone, Debug)]
pub struct G2PWPinyinOptions {
  /// `model_source`, see [`G2PWOnnxConverterOptions::model_source`].
  pub model_source: Option<std::path::PathBuf>,
  /// `enable_non_tradional_chinese` (upstream spelling).
  pub enable_non_tradional_chinese: bool,
  /// `v_to_u`.
  pub v_to_u: bool,
  /// `neutral_tone_with_five`.
  pub neutral_tone_with_five: bool,
}

impl Default for G2PWPinyinOptions {
  fn default() -> Self {
    Self {
      model_source: None,
      enable_non_tradional_chinese: true,
      v_to_u: false,
      neutral_tone_with_five: false,
    }
  }
}

impl G2PWPinyinOptions {
  /// The arguments that `chinese2.py` passes: `v_to_u=False,
  /// neutral_tone_with_five=True` (and the default
  /// `enable_non_tradional_chinese=True`).
  pub fn chinese2() -> Self {
    Self {
      neutral_tone_with_five: true,
      ..Self::default()
    }
  }
}

/// Port of `class G2PWPinyin(Pinyin)`.
pub struct G2PWPinyin {
  /// `self._g2pw` (`G2PWOnnxConverter(style="pinyin")`), shared with the
  /// converter. `chinese2.py` calls it directly (`g2pw._g2pw(batch)`).
  pub _g2pw: Arc<G2PWOnnxConverter>,
  /// `self._converter`.
  pub _converter: Converter,
}

impl G2PWPinyin {
  /// Port of `G2PWPinyin.__init__`.
  pub fn new(model_dir: impl AsRef<Path>, options: &G2PWPinyinOptions) -> Result<Self, G2PWError> {
    let g2pw = Arc::new(G2PWOnnxConverter::new(
      model_dir,
      &G2PWOnnxConverterOptions {
        style: G2PWStyle::Pinyin,
        model_source: options.model_source.clone(),
        enable_non_tradional_chinese: options.enable_non_tradional_chinese,
      },
    )?);
    let converter = Converter::new(g2pw.clone(), options.v_to_u, options.neutral_tone_with_five);
    Ok(Self {
      _g2pw: g2pw,
      _converter: converter,
    })
  }
}

impl PinyinMethods for G2PWPinyin {
  type Converter = Converter;

  fn converter(&self) -> &Converter {
    &self._converter
  }

  /// Port of `G2PWPinyin.get_seg`: `simple_seg`.
  fn get_seg(&self) -> fn(&str) -> Vec<String> {
    |hans| simple_seg(hans)
  }
}

/// Port of `class Converter(UltimateConverter)`.
pub struct Converter {
  /// The `UltimateConverter` base (`super()`).
  pub inner: UltimateConverter,
  /// `self._g2pw`.
  pub _g2pw: Arc<G2PWOnnxConverter>,
}

impl Converter {
  /// Port of `Converter.__init__(g2pw_instance, v_to_u=False,
  /// neutral_tone_with_five=False, tone_sandhi=False)`.
  pub fn new(
    g2pw_instance: Arc<G2PWOnnxConverter>,
    v_to_u: bool,
    neutral_tone_with_five: bool,
  ) -> Self {
    Self {
      inner: UltimateConverter::new(v_to_u, neutral_tone_with_five),
      _g2pw: g2pw_instance,
    }
  }

  /// Port of `Converter._to_pinyin(han, style, heteronym, errors, strict)`.
  ///
  /// The pypinyin converter methods cannot fail, so an error of
  /// `self._g2pw(han)` panics, as Python exceptions do in the `pypinyin`
  /// crate.
  pub fn _to_pinyin(
    &self,
    han: &str,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    let mut pinyins = Vec::new();

    let g2pw_pinyin = self
      ._g2pw
      .call(&[han])
      .unwrap_or_else(|e| panic!("G2PWOnnxConverter({han:?}): {e}"));

    if g2pw_pinyin.is_empty() {
      // g2pw 不支持的汉字改为使用 pypinyin 原有逻辑
      return default_converter::convert(self, han, Style::TONE, heteronym, errors, strict);
    }

    let chars: Vec<char> = han.chars().collect();
    for (i, item) in g2pw_pinyin[0].iter().enumerate() {
      match item {
        None => {
          // g2pw 不支持的汉字改为使用 pypinyin 原有逻辑
          let py = default_converter::convert(
            self,
            &chars[i].to_string(),
            Style::TONE,
            heteronym,
            errors,
            strict,
          );
          pinyins.extend(py);
        }
        Some(item) => pinyins.push(vec![to_tone(item)]),
      }
    }

    pinyins
  }
}

impl pypinyin::Converter for Converter {
  /// Port of `Converter.convert(words, style, heteronym, errors, strict)`.
  fn convert(
    &self,
    words: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    let mut pys = Vec::new();
    if RE_HANS.is_match(words) {
      pys = self._to_pinyin(words, heteronym, errors, strict);
      if let Some(post_data) = self.post_pinyin(words, heteronym, &pys) {
        pys = post_data;
      }

      pys = self.convert_styles(pys, words, style, heteronym, errors, strict);
    } else {
      let py = self.handle_nopinyin(words, style, heteronym, errors, strict);
      if !py.is_empty() {
        pys.extend(py);
      }
    }

    _remove_dup_and_empty(&pys)
  }

  /// Inherited from `UltimateConverter.post_convert_style`.
  fn post_convert_style(
    &self,
    han: &str,
    orig_pinyin: &str,
    converted_pinyin: &str,
    style: Style,
    strict: bool,
  ) -> Option<String> {
    self
      .inner
      .post_convert_style(han, orig_pinyin, converted_pinyin, style, strict)
  }

  /// Inherited from `UltimateConverter.post_pinyin`.
  fn post_pinyin(
    &self,
    han: &str,
    heteronym: bool,
    pinyin: &[Vec<String>],
  ) -> Option<Vec<Vec<String>>> {
    self.inner.post_pinyin(han, heteronym, pinyin)
  }

  /// `isinstance(self, UltimateConverter)`.
  fn is_ultimate_converter(&self) -> bool {
    true
  }
}

/// Port of `_remove_dup_items(lst, remove_empty=False)` (the copy in
/// `g2pw.py`).
pub fn _remove_dup_items(lst: &[String], remove_empty: bool) -> Vec<String> {
  let mut new_lst: Vec<String> = Vec::new();
  for item in lst {
    if remove_empty && item.is_empty() {
      continue;
    }
    if !new_lst.contains(item) {
      new_lst.push(item.clone());
    }
  }
  new_lst
}

/// Port of `_remove_dup_and_empty(lst_list)` (the copy in `g2pw.py`).
pub fn _remove_dup_and_empty(lst_list: &[Vec<String>]) -> Vec<Vec<String>> {
  let mut new_lst_list = Vec::new();
  for lst in lst_list {
    let lst = _remove_dup_items(lst, true);
    if !lst.is_empty() {
      new_lst_list.push(lst);
    } else {
      new_lst_list.push(vec![String::new()]);
    }
  }
  new_lst_list
}

/// Parses `eval(value_str.strip())` for the values of the `.rep` files: a
/// list literal of quoted strings (`['hu2','po1']`). Other expressions are
/// not supported and give `ValueError`.
fn eval_str_list(value: &str) -> Result<Vec<String>, G2PWError> {
  let err = || G2PWError::ValueError(format!("unsupported .rep value {value:?}"));
  let inner = value
    .strip_prefix('[')
    .and_then(|v| v.strip_suffix(']'))
    .ok_or_else(err)?;
  let chars: Vec<char> = inner.chars().collect();
  let mut out = Vec::new();
  let mut pos = 0;
  let skip_ws = |pos: &mut usize| {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
      *pos += 1;
    }
  };
  loop {
    skip_ws(&mut pos);
    if pos >= chars.len() {
      break;
    }
    let quote = chars[pos];
    if quote != '\'' && quote != '"' {
      return Err(err());
    }
    pos += 1;
    let start = pos;
    while pos < chars.len() && chars[pos] != quote {
      if chars[pos] == '\\' {
        return Err(err());
      }
      pos += 1;
    }
    if pos >= chars.len() {
      return Err(err());
    }
    out.push(chars[start..pos].iter().collect());
    pos += 1;
    skip_ws(&mut pos);
    if pos >= chars.len() {
      break;
    }
    if chars[pos] != ',' {
      return Err(err());
    }
    pos += 1;
  }
  Ok(out)
}

/// The loop of `read_dict` over one file: `key, value_str =
/// line.split(":")` for each line (`readline()` keeps the `\n`, which
/// `strip()` removes), later lines replacing earlier ones.
fn read_rep(
  text: &str,
  polyphonic_dict: &mut HashMap<String, Vec<String>>,
) -> Result<(), G2PWError> {
  for line in text.split_inclusive('\n') {
    let parts: Vec<&str> = line.split(':').collect();
    let [key, value_str] = parts.as_slice() else {
      return Err(G2PWError::ValueError(format!(
        "expected 2 values to unpack, got {} in {line:?}",
        parts.len()
      )));
    };
    let value = eval_str_list(super::onnx_api::py_strip(value_str))?;
    polyphonic_dict.insert(super::onnx_api::py_strip(key).to_string(), value);
  }
  Ok(())
}

/// Port of `read_dict()`: `polyphonic.rep`, then `polyphonic-fix.rep`
/// (whose entries replace those of the first file).
pub fn read_dict() -> Result<HashMap<String, Vec<String>>, G2PWError> {
  let mut polyphonic_dict = HashMap::new();
  read_rep(POLYPHONIC_REP, &mut polyphonic_dict)?;
  read_rep(POLYPHONIC_FIX_REP, &mut polyphonic_dict)?;
  Ok(polyphonic_dict)
}

/// Port of `get_dict()` without the `polyphonic.pickle` cache: the cache
/// only stores `read_dict()`, so the port always calls it.
pub fn get_dict() -> HashMap<String, Vec<String>> {
  read_dict().expect("embedded .rep files parse")
}

/// Port of `pp_dict = get_dict()`.
pub static PP_DICT: LazyLock<HashMap<String, Vec<String>>> = LazyLock::new(get_dict);

/// Port of `correct_pronunciation(word, word_pinyins)`.
///
/// When `word` is in `pp_dict`, its entry is returned. Otherwise each
/// character of `word` that is in `pp_dict` replaces `word_pinyins[idx]`
/// with the first pinyin of its entry, and the (modified) `word_pinyins` is
/// returned. Python mutates the argument in place; an `idx` past the end of
/// `word_pinyins` gives `IndexError`, and an empty entry for a character
/// gives `IndexError` too.
pub fn correct_pronunciation(
  word: &str,
  mut word_pinyins: Vec<String>,
) -> Result<Vec<String>, G2PWError> {
  match PP_DICT.get(word) {
    None => {
      for (idx, w) in word.chars().enumerate() {
        if let Some(w_pinyin) = PP_DICT.get(w.to_string().as_str()) {
          let first = w_pinyin
            .first()
            .ok_or_else(|| G2PWError::IndexError("list index out of range".to_string()))?;
          let slot = word_pinyins.get_mut(idx).ok_or_else(|| {
            G2PWError::IndexError("list assignment index out of range".to_string())
          })?;
          *slot = first.clone();
        }
      }
      Ok(word_pinyins)
    }
    Some(new_pinyins) => Ok(new_pinyins.clone()),
  }
}
