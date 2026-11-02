//! Port of `split_lang/split/splitter.py`.
//!
//! Python mutates `SubString` / `SubStringSection` objects that several
//! lists reference. In every function here the aliased objects are only
//! mutated after the last read through the other reference, so owned values
//! give the same result; comments mark the places where this matters.
//!
//! Debug logging (`debug`, `log_level`) is left out.

use std::sync::Arc;

use fast_langdetect::LangDetector;

use crate::budoux;
use crate::config::{
  DEFAULT_LANG, DEFAULT_LANG_MAP, NO_ZH_JA_LANG_MAP, ZH_JA_LANG_MAP, lang_map_get,
};
use crate::detect_lang::detector::{detect_lang_combined, possible_detection_list};
use crate::model::{LangSectionType, SubString, SubStringSection};
use crate::split::utils::{PUNCTUATION, contains_hangul, contains_ja_kana, contains_zh_ja};
use crate::{SplitLangError, index_error, py};

/// Which lang map `_init_substr_lang` / `_get_languages` use (`lang_map=None`
/// means `self.lang_map`).
#[derive(Clone, Copy)]
enum LangMapArg {
  SelfMap,
  ZhJa,
  NoZhJa,
}

/// Port of `class LangSplitter`.
#[derive(Debug, Clone)]
pub struct LangSplitter {
  /// `lang_map`.
  pub lang_map: Vec<(String, String)>,
  /// `default_lang`.
  pub default_lang: String,
  /// `punctuation`.
  pub punctuation: String,
  /// `not_merge_punctuation`.
  pub not_merge_punctuation: String,
  /// `merge_across_punctuation`.
  pub merge_across_punctuation: bool,
  /// `merge_across_digit`.
  pub merge_across_digit: bool,
  /// `merge_across_newline`.
  pub merge_across_newline: bool,
  /// Not in Python: the fast-langdetect detector to use. `None` (the
  /// default) uses `fast_langdetect`'s module-level default detector, which
  /// is what Python does.
  pub detector: Option<Arc<LangDetector>>,
}

/// Keyword arguments of `LangSplitter.__init__` (`debug` and `log_level` only
/// control logging and are left out).
#[derive(Debug, Clone)]
pub struct LangSplitterOptions {
  /// `lang_map` (default `DEFAULT_LANG_MAP`).
  pub lang_map: Vec<(String, String)>,
  /// `default_lang` (default `DEFAULT_LANG`).
  pub default_lang: String,
  /// `punctuation` (default `PUNCTUATION`).
  pub punctuation: String,
  /// `not_merge_punctuation` (default `""`).
  pub not_merge_punctuation: String,
  /// `merge_across_punctuation` (default `True`).
  pub merge_across_punctuation: bool,
  /// `merge_across_digit` (default `True`).
  pub merge_across_digit: bool,
  /// `merge_across_newline` (default `True`).
  pub merge_across_newline: bool,
}

impl Default for LangSplitterOptions {
  fn default() -> Self {
    Self {
      lang_map: to_owned_map(DEFAULT_LANG_MAP),
      default_lang: DEFAULT_LANG.to_string(),
      punctuation: PUNCTUATION.to_string(),
      not_merge_punctuation: String::new(),
      merge_across_punctuation: true,
      merge_across_digit: true,
      merge_across_newline: true,
    }
  }
}

/// Converts a `(key, value)` slice to an owned lang map.
pub fn to_owned_map(map: &[(&str, &str)]) -> Vec<(String, String)> {
  map
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

impl Default for LangSplitter {
  /// `LangSplitter()`.
  fn default() -> Self {
    Self::new(LangSplitterOptions::default())
  }
}

/// The lang of `substrings[i]`, or `IndexError`.
fn at<T>(v: &[T], i: usize) -> Result<&T, SplitLangError> {
  v.get(i).ok_or_else(index_error)
}

/// Python `l[-1]`, or `IndexError`.
fn last_mut<T>(v: &mut [T]) -> Result<&mut T, SplitLangError> {
  v.last_mut().ok_or_else(index_error)
}

/// Recomputes `index` as in the `# NOTE: 重新计算 index` blocks of the
/// `_merge_substrings_across_*_based_on_sections` methods.
fn recompute_section_indices(sections: &mut [SubStringSection]) -> Result<(), SplitLangError> {
  for section_index in 0..sections.len() {
    if section_index == 0 {
      chain_indices(&mut sections[0].substrings, 1);
    } else if !sections[section_index].substrings.is_empty() {
      let prev = sections[section_index - 1]
        .substrings
        .last()
        .ok_or_else(index_error)?;
      let start = prev.index + prev.length;
      sections[section_index].substrings[0].index = start;
      chain_indices(&mut sections[section_index].substrings, 1);
    }
  }
  Ok(())
}

/// `for index, substr in enumerate(substrings): if index >= from: substr.index = prev.index + prev.length`.
fn chain_indices(substrings: &mut [SubString], from: usize) {
  for i in from.max(1)..substrings.len() {
    substrings[i].index = substrings[i - 1].index + substrings[i - 1].length;
  }
}

/// `# NOTE: 将相同类型的 section 合并`.
fn merge_same_type_sections(new_sections: Vec<SubStringSection>) -> Vec<SubStringSection> {
  let mut merged: Vec<SubStringSection> = Vec::new();
  for (index, section) in new_sections.into_iter().enumerate() {
    if index == 0 {
      merged.push(section);
      continue;
    }
    let last = merged.last_mut().expect("non-empty");
    if last.lang_section_type == section.lang_section_type {
      last.text += &section.text;
      last.substrings.extend(section.substrings);
    } else {
      merged.push(section);
    }
  }
  merged
}

impl LangSplitter {
  /// Port of `LangSplitter.__init__(...)`.
  pub fn new(options: LangSplitterOptions) -> Self {
    Self {
      lang_map: options.lang_map,
      default_lang: options.default_lang,
      punctuation: options.punctuation,
      not_merge_punctuation: options.not_merge_punctuation,
      merge_across_punctuation: options.merge_across_punctuation,
      merge_across_digit: options.merge_across_digit,
      merge_across_newline: options.merge_across_newline,
      detector: None,
    }
  }

  fn detector(&self) -> Option<&LangDetector> {
    self.detector.as_deref()
  }

  fn map_get(&self, lang_map: LangMapArg, key: &str) -> String {
    match lang_map {
      LangMapArg::SelfMap => lang_map_get(&self.lang_map, key, &self.default_lang),
      LangMapArg::ZhJa => lang_map_get(ZH_JA_LANG_MAP, key, &self.default_lang),
      LangMapArg::NoZhJa => lang_map_get(NO_ZH_JA_LANG_MAP, key, &self.default_lang),
    }
  }

  fn possible_detection_list(&self, text: &str) -> Result<Vec<String>, SplitLangError> {
    possible_detection_list(text, self.detector())
  }

  /// Port of `LangSplitter.split_by_lang(text)`.
  pub fn split_by_lang(&self, text: &str) -> Result<Vec<SubString>, SplitLangError> {
    let pre_split_section = self.pre_split(text);

    let mut sections = self._split(pre_split_section)?;

    if self.merge_across_digit {
      sections = self._merge_substrings_across_digit_based_on_sections(sections)?;
    }

    if self.merge_across_punctuation {
      sections = self._merge_substrings_across_punctuation_based_on_sections(sections)?;
    }

    if self.merge_across_newline {
      sections = self._merge_substrings_across_newline_based_on_sections(sections)?;
    }

    for section in &mut sections {
      if section.lang_section_type == LangSectionType::ZhJa {
        let substrings = std::mem::take(&mut section.substrings);
        section.substrings = self._special_merge_for_zh_ja(substrings)?;
      }
    }

    Ok(sections.into_iter().flat_map(|s| s.substrings).collect())
  }

  /// Port of `LangSplitter.pre_split(text)`.
  pub fn pre_split(&self, text: &str) -> Vec<SubStringSection> {
    let text: Vec<char> = py::strip(text).chars().collect();
    let mut sections: Vec<SubStringSection> = Vec::new();
    let mut current_lang = LangSectionType::Others;
    let mut current_text: Vec<char> = Vec::new();

    let add_substring = |sections: &mut Vec<SubStringSection>,
                         current_text: &mut Vec<char>,
                         lang_section_type: LangSectionType| {
      if !current_text.is_empty() {
        sections.push(SubStringSection {
          lang_section_type,
          text: current_text.iter().collect(),
          substrings: Vec::new(),
        });
        current_text.clear();
      }
    };

    for (index, &ch) in text.iter().enumerate() {
      let mut buf = [0u8; 4];
      let ch_str: &str = ch.encode_utf8(&mut buf);
      if contains_zh_ja(ch_str) {
        if current_lang != LangSectionType::ZhJa {
          add_substring(&mut sections, &mut current_text, current_lang);
          current_lang = LangSectionType::ZhJa;
        }
      } else if contains_hangul(ch_str) {
        if current_lang != LangSectionType::Ko {
          add_substring(&mut sections, &mut current_text, current_lang);
          current_lang = LangSectionType::Ko;
        }
      } else if py::isdigit(ch) {
        if current_lang != LangSectionType::Digit {
          add_substring(&mut sections, &mut current_text, current_lang);
          current_lang = LangSectionType::Digit;
        }
      } else if self.punctuation.contains(ch) {
        if ch == '\'' && index > 0 && !py::isspace(text[index - 1]) {
          // For English, "'" is a part of word
        } else {
          add_substring(&mut sections, &mut current_text, current_lang);
          current_lang = LangSectionType::Punctuation;
        }
      } else if py::isspace(ch) {
        if ch == '\n' || ch == '\r' {
          add_substring(&mut sections, &mut current_text, current_lang);
          current_lang = LangSectionType::Newline;
        }
      } else if current_lang != LangSectionType::Others {
        add_substring(&mut sections, &mut current_text, current_lang);
        current_lang = LangSectionType::Others;
      }

      current_text.push(ch);
    }

    add_substring(&mut sections, &mut current_text, current_lang);
    sections
  }

  /// Port of `LangSplitter._split(pre_split_section)`.
  pub fn _split(
    &self,
    mut pre_split_section: Vec<SubStringSection>,
  ) -> Result<Vec<SubStringSection>, SplitLangError> {
    let mut section_index = 0usize;
    for section in &mut pre_split_section {
      let section_len = py::len(&section.text);
      let fixed_lang = match section.lang_section_type {
        LangSectionType::Punctuation => Some("punctuation"),
        LangSectionType::Digit => Some("digit"),
        LangSectionType::Newline => Some("newline"),
        _ => None,
      };
      if let Some(lang) = fixed_lang {
        section.substrings.push(SubString {
          text: section.text.clone(),
          lang: lang.to_string(),
          index: section_index,
          length: section_len,
        });
      } else {
        let mut substrings_with_lang: Vec<SubString> = match section.lang_section_type {
          LangSectionType::ZhJa => {
            let temp_substrings = self._parse_zh_ja(&section.text);
            self._init_substr_lang(&temp_substrings, LangSectionType::ZhJa, LangMapArg::ZhJa)?
          }
          LangSectionType::Ko => vec![SubString {
            text: section.text.clone(),
            lang: "ko".to_string(),
            index: section_index,
            length: section_len,
          }],
          _ => {
            let temp_substrings = self._parse_without_zh_ja(&section.text);
            self._init_substr_lang(
              &temp_substrings,
              LangSectionType::Others,
              LangMapArg::SelfMap,
            )?
          }
        };
        // 更新 index (the KO substring already starts at section_index and
        // gets it added a second time, as in Python)
        for substr in &mut substrings_with_lang {
          substr.index += section_index;
        }
        section.substrings = substrings_with_lang;
      }
      section_index += section_len;
    }

    self._smart_merge_all(pre_split_section)
  }

  /// Port of `LangSplitter._smart_merge_all(pre_split_section)`.
  pub fn _smart_merge_all(
    &self,
    mut pre_split_section: Vec<SubStringSection>,
  ) -> Result<Vec<SubStringSection>, SplitLangError> {
    for section in &mut pre_split_section {
      if section.lang_section_type == LangSectionType::Punctuation
        || section.lang_section_type == LangSectionType::Newline
      {
        continue;
      }
      let substrings = std::mem::take(&mut section.substrings);
      section.substrings = self._smart_merge(substrings, section.lang_section_type)?;
    }
    Ok(pre_split_section)
  }

  /// Port of `LangSplitter._parse_without_zh_ja(text)`.
  pub fn _parse_without_zh_ja(&self, text: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut exist_space = false;
    let mut chars = String::new();
    for ch in text.chars() {
      if !py::isspace(ch) {
        if exist_space {
          words.push(std::mem::take(&mut chars));
          exist_space = false;
        }
        chars.push(ch);
      } else {
        exist_space = true;
        chars.push(ch);
      }
    }
    if !chars.is_empty() {
      words.push(chars);
    }
    words
  }

  /// Port of `LangSplitter._parse_zh_ja(text)`.
  ///
  /// Python raises `IndexError` (`splitted_texts_zh_jp[0]`) for an empty
  /// `text`, which `_split` never passes; this port returns an empty list.
  pub fn _parse_zh_ja(&self, text: &str) -> Vec<String> {
    let splitted_texts_jp = budoux::load_default_japanese_parser().parse(text);

    let mut splitted_texts_zh_jp: Vec<String> = Vec::new();
    for substring in &splitted_texts_jp {
      splitted_texts_zh_jp
        .extend(budoux::load_default_simplified_chinese_parser().parse(substring));
    }

    let mut iter = splitted_texts_zh_jp.into_iter();
    let Some(first) = iter.next() else {
      return Vec::new();
    };
    let mut new_substrings = vec![first];
    for substring in iter {
      let is_left_ja_kana = contains_ja_kana(new_substrings.last().expect("non-empty"));
      let is_cur_ja_kana = contains_ja_kana(&substring);
      let is_both_same_lang = is_left_ja_kana == is_cur_ja_kana;
      let is_both_ja_kana = is_left_ja_kana && is_both_same_lang;
      let is_both_zh = !is_left_ja_kana && is_both_same_lang;
      if is_both_ja_kana || (is_both_zh && py::len(&substring) == 1) {
        *new_substrings.last_mut().expect("non-empty") += &substring;
      } else {
        new_substrings.push(substring);
      }
    }

    if new_substrings.len() >= 2 && py::len(&new_substrings[0]) == 1 {
      let first = new_substrings.remove(0);
      new_substrings[0] = first + &new_substrings[0];
    }

    new_substrings
  }

  /// Port of `LangSplitter._smart_merge(substr_list, lang_section_type)`.
  fn _smart_merge(
    &self,
    substr_list: Vec<SubString>,
    lang_section_type: LangSectionType,
  ) -> Result<Vec<SubString>, SplitLangError> {
    let lang_map = if lang_section_type == LangSectionType::ZhJa {
      LangMapArg::ZhJa
    } else {
      LangMapArg::NoZhJa
    };
    let mut l = substr_list;
    l = self._merge_substrings(l);
    l = self._merge_middle_substr_to_two_side(l)?;
    l = self._merge_substrings(l);
    l = self._get_languages(l, lang_section_type, lang_map)?;
    l = self._merge_middle_substr_to_two_side(l)?;
    l = self._merge_substrings(l);
    l = self._fill_unknown_language(l);
    l = self._merge_side_substr_to_near(l)?;
    l = self._merge_substrings(l);
    l = self._get_languages(l, lang_section_type, lang_map)?;
    l = self._merge_middle_substr_to_two_side(l)?;
    l = self._merge_substrings(l);
    Ok(l)
  }

  /// Port of `LangSplitter._is_merge_middle_to_two_side(left, middle, right)`.
  fn _is_merge_middle_to_two_side(
    &self,
    left: &SubString,
    middle: &SubString,
    right: &SubString,
  ) -> Result<bool, SplitLangError> {
    let is_middle_contains_no_kana = !contains_ja_kana(&middle.text);

    let middle_lang_is_possible_the_same_as_left = self
      .possible_detection_list(&middle.text)?
      .contains(&left.lang)
      && is_middle_contains_no_kana;
    let middle_lang_is_x = middle.lang == "x";

    let is_middle_short_and_two_side_long =
      middle.length <= 4 && left.length + right.length >= 6 && is_middle_contains_no_kana;

    Ok(
      middle_lang_is_possible_the_same_as_left
        || middle_lang_is_x
        || is_middle_short_and_two_side_long,
    )
  }

  /// Port of `LangSplitter._is_cur_short_and_near_long(cur, near)`.
  fn _is_cur_short_and_near_long(
    &self,
    cur: &SubString,
    near: &SubString,
  ) -> Result<bool, SplitLangError> {
    let is_cur_contains_no_kana = !contains_ja_kana(&cur.text);
    let is_near_contains_no_kana = !contains_ja_kana(&near.text);
    let is_both_contains_no_kana = is_cur_contains_no_kana && is_near_contains_no_kana;
    let is_near_str_is_long_zh = near.length > cur.length && near.length >= 6 && near.lang == "zh";
    let is_cur_short_and_near_is_zh_and_long = is_near_str_is_long_zh && is_both_contains_no_kana;
    let cur_lang_is_possible_the_same_as_near = self
      .possible_detection_list(&cur.text)?
      .contains(&near.lang)
      && is_both_contains_no_kana;
    Ok(is_cur_short_and_near_is_zh_and_long || cur_lang_is_possible_the_same_as_near)
  }

  /// Port of `LangSplitter._merge_middle_substr_to_two_side(substrings)`.
  fn _merge_middle_substr_to_two_side(
    &self,
    mut substrings: Vec<SubString>,
  ) -> Result<Vec<SubString>, SplitLangError> {
    let substr_len = substrings.len();
    if substr_len <= 2 {
      return Ok(substrings);
    }
    for index in 0..substr_len - 2 {
      let left_block = &substrings[index];
      let middle_block = &substrings[index + 1];
      let right_block = &substrings[index + 2];

      if left_block.lang == right_block.lang
        && left_block.lang != "x"
        && left_block.lang != "newline"
        && self._is_merge_middle_to_two_side(left_block, middle_block, right_block)?
      {
        substrings[index + 1].lang = substrings[index].lang.clone();
      }
    }
    Ok(substrings)
  }

  /// Port of `LangSplitter._merge_side_substr_to_near(substrings)`.
  fn _merge_side_substr_to_near(
    &self,
    mut substrings: Vec<SubString>,
  ) -> Result<Vec<SubString>, SplitLangError> {
    // NOTE: Merge leftest substr
    let is_lang_x = at(&substrings, 0)?.lang == "x";
    let mut is_cur_short_and_near_long = false;
    let mut is_possible_same_lang_with_near = false;
    if substrings.len() >= 2 {
      is_cur_short_and_near_long =
        self._is_cur_short_and_near_long(&substrings[0], &substrings[1])?;
      is_possible_same_lang_with_near = self
        .possible_detection_list(&substrings[0].text)?
        .contains(&substrings[1].lang)
        && substrings[1].length <= 5;
    }
    if is_lang_x || is_cur_short_and_near_long || is_possible_same_lang_with_near {
      substrings[0].lang = self._get_nearest_lang_with_direction(&substrings, 0, false);
    }

    // NOTE: Merge rightest substr
    let n = substrings.len();
    let is_lang_x = substrings[n - 1].lang == "x";
    let mut is_cur_short_and_near_long = false;
    let mut is_possible_same_lang_with_near = false;
    if n >= 2 {
      is_cur_short_and_near_long =
        self._is_cur_short_and_near_long(&substrings[n - 1], &substrings[n - 2])?;
      is_possible_same_lang_with_near = self
        .possible_detection_list(&substrings[n - 1].text)?
        .contains(&substrings[n - 2].lang)
        && substrings[n - 2].length <= 5;
    }
    if is_lang_x || is_cur_short_and_near_long || is_possible_same_lang_with_near {
      substrings[n - 1].lang = self._get_nearest_lang_with_direction(&substrings, n - 1, true);
    }
    Ok(substrings)
  }

  /// Port of `LangSplitter._fill_unknown_language(substrings)`.
  fn _fill_unknown_language(&self, mut substrings: Vec<SubString>) -> Vec<SubString> {
    let n = substrings.len();
    for index in 0..n {
      if substrings[index].lang == "x" {
        let search_left = if index == 0 {
          false
        } else if index == n - 1 {
          true
        } else {
          self._get_merge_direction(&substrings, index)
        };
        substrings[index].lang =
          self._get_nearest_lang_with_direction(&substrings, index, search_left);
      }
    }
    substrings
  }

  /// Port of `LangSplitter._get_nearest_lang_with_direction(substrings, index, search_left)`.
  fn _get_nearest_lang_with_direction(
    &self,
    substrings: &[SubString],
    index: usize,
    search_left: bool,
  ) -> String {
    let n = substrings.len();
    for i in 1..n {
      let j = if search_left {
        match index.checked_sub(i) {
          Some(j) => j,
          None => continue,
        }
      } else {
        let j = index + i;
        if j >= n {
          continue;
        }
        j
      };
      if substrings[j].lang != "x" && substrings[j].lang != "digit" {
        return substrings[j].lang.clone();
      }
    }
    substrings[index].lang.clone()
  }

  /// Port of `LangSplitter._get_merge_direction(substrings, index)`
  /// (`True` means search left).
  fn _get_merge_direction(&self, substrings: &[SubString], index: usize) -> bool {
    if index == 0 {
      return false;
    } else if index == substrings.len() - 1 {
      return true;
    }
    let left_block = &substrings[index - 1];
    let right_block = &substrings[index + 1];
    py::len(&left_block.text) >= py::len(&right_block.text)
  }

  /// Port of `LangSplitter._merge_substrings(substrings)`.
  fn _merge_substrings(&self, substrings: Vec<SubString>) -> Vec<SubString> {
    let mut smart_concat_result: Vec<SubString> = Vec::new();
    let mut lang = String::new();
    for block in substrings {
      let cur_lang = block.lang.clone();
      if cur_lang != lang {
        smart_concat_result.push(block);
      } else {
        let last = smart_concat_result.last_mut().expect("non-empty");
        last.text += &block.text;
        last.length += block.length;
      }
      lang = cur_lang;
    }
    smart_concat_result
  }

  /// Port of `LangSplitter._special_merge_for_zh_ja(substrings)`.
  fn _special_merge_for_zh_ja(
    &self,
    substrings: Vec<SubString>,
  ) -> Result<Vec<SubString>, SplitLangError> {
    if substrings.len() == 1 {
      return Ok(substrings);
    }
    let mut new_substrings: Vec<SubString> = Vec::new();
    // NOTE: 统计每个语言的字符串长度 (dict in insertion order)
    let mut len_by_lang: Vec<(&str, usize)> = vec![
      ("zh", 0),
      ("ja", 0),
      ("x", 0),
      ("digit", 0),
      ("punctuation", 0),
      ("newline", 0),
    ];
    let n = substrings.len();
    let is_zh_ja = |l: &str| l == "zh" || l == "ja";
    let is_zh_ja_x = |l: &str| l == "zh" || l == "ja" || l == "x";
    let mut index = 0;
    while index < n {
      let current_block = &substrings[index];
      match len_by_lang
        .iter_mut()
        .find(|(k, _)| *k == current_block.lang)
      {
        Some((_, v)) => *v += current_block.length,
        None => return Err(SplitLangError::KeyError(current_block.lang.clone())),
      }
      if index == 0 {
        let right = &substrings[index + 1];
        if is_zh_ja(&right.lang)
          && is_zh_ja_x(&current_block.lang)
          && current_block.length * 10 < right.length
        {
          new_substrings.push(SubString {
            lang: right.lang.clone(),
            text: current_block.text.clone() + &right.text,
            length: current_block.length + right.length,
            index: current_block.index,
          });
          index += 1;
        } else {
          new_substrings.push(current_block.clone());
        }
      } else if index == n - 1 {
        let left_block = last_mut(&mut new_substrings)?;
        if is_zh_ja(&left_block.lang)
          && is_zh_ja_x(&current_block.lang)
          && current_block.length * 10 < left_block.length
        {
          left_block.text += &current_block.text;
          left_block.length += current_block.length;
          index += 1;
        } else {
          new_substrings.push(current_block.clone());
        }
      } else {
        let right_block = &substrings[index + 1];
        let left_block = last_mut(&mut new_substrings)?;
        if left_block.lang == right_block.lang
          && is_zh_ja(&left_block.lang)
          && current_block.lang != "en"
          && current_block.length * 10 < left_block.length + right_block.length
        {
          left_block.text += &current_block.text;
          left_block.text += &right_block.text;
          left_block.length += current_block.length + right_block.length;
          index += 1;
        } else {
          new_substrings.push(current_block.clone());
        }
      }
      index += 1;
    }

    let get = |k: &str| {
      len_by_lang
        .iter()
        .find(|(key, _)| *key == k)
        .map_or(0, |(_, v)| *v)
    };
    // NOTE: 如果 substring_count 中 存在 x，则将 x 设置为最多的 lang
    if get("x") > 0 {
      // max(dict, key=dict.get): the first key with the largest value.
      let mut max_lang = len_by_lang[0];
      for &(k, v) in &len_by_lang[1..] {
        if v > max_lang.1 {
          max_lang = (k, v);
        }
      }
      for substr in &mut new_substrings {
        if substr.lang == "x" {
          substr.lang = max_lang.0.to_string();
        }
      }
    }
    // NOTE: 如果 ja 数量是 zh 数量的 10 倍以上，则该 zh 设置为 ja
    if get("ja") >= get("zh") * 10 {
      for substr in &mut new_substrings {
        if substr.lang == "zh" {
          substr.lang = "ja".to_string();
        }
      }
    }
    Ok(self._merge_substrings(new_substrings))
  }

  /// Shared body of `_merge_substrings_across_newline` /
  /// `_merge_substrings_across_punctuation` (identical apart from `kind`).
  fn merge_substrings_across(kind: &str, substrings: Vec<SubString>) -> Vec<SubString> {
    let mut new_substrings: Vec<SubString> = Vec::new();
    let mut last_lang = String::new();
    for substring in substrings {
      let substring_lang = substring.lang.clone();
      if let Some(last) = new_substrings.last_mut() {
        if substring.lang == kind {
          last.text += &substring.text;
          last.length += substring.length;
        } else if substring.lang == last_lang || last_lang.is_empty() {
          last.text += &substring.text;
          last.length += substring.length;
          if last.lang == kind {
            last.lang = substring.lang.clone();
          }
        } else {
          new_substrings.push(substring);
        }
      } else {
        new_substrings.push(substring);
      }
      last_lang = substring_lang;
    }
    new_substrings
  }

  /// Port of `LangSplitter._merge_substrings_across_newline(substrings)`.
  fn _merge_substrings_across_newline(&self, substrings: Vec<SubString>) -> Vec<SubString> {
    Self::merge_substrings_across("newline", substrings)
  }

  /// Port of `LangSplitter._merge_substrings_across_newline_based_on_sections(sections)`.
  fn _merge_substrings_across_newline_based_on_sections(
    &self,
    sections: Vec<SubStringSection>,
  ) -> Result<Vec<SubStringSection>, SplitLangError> {
    let mut iter = sections.into_iter();
    let first = iter.next().ok_or_else(index_error)?;
    let mut new_sections: Vec<SubStringSection> = vec![first];
    for current_section in iter {
      let last = new_sections.last_mut().expect("non-empty");
      if last.lang_section_type == LangSectionType::Newline {
        last.lang_section_type = current_section.lang_section_type;
        last.text += &current_section.text;
        last.substrings.extend(current_section.substrings);
        chain_indices(&mut last.substrings, 1);
      } else if current_section.lang_section_type == LangSectionType::Newline {
        last.text += &current_section.text;
        last.substrings.extend(current_section.substrings);
        let n = last.substrings.len();
        if n < 2 {
          return Err(index_error());
        }
        last.substrings[n - 1].index = last.substrings[n - 2].index + last.substrings[n - 2].length;
      } else {
        new_sections.push(current_section);
      }
    }
    let mut merged = merge_same_type_sections(new_sections);
    recompute_section_indices(&mut merged)?;
    for section in &mut merged {
      let substrings = std::mem::take(&mut section.substrings);
      section.substrings = self._merge_substrings_across_newline(substrings);
    }
    Ok(merged)
  }

  /// Port of `LangSplitter._merge_substrings_across_digit(substrings)`.
  fn _merge_substrings_across_digit(&self, substrings: Vec<SubString>) -> Vec<SubString> {
    let mut new_substrings: Vec<SubString> = Vec::new();
    let mut last_lang = String::new();
    for substring in substrings {
      let substring_lang = substring.lang.clone();
      if let Some(last) = new_substrings.last_mut() {
        if substring.lang == "digit" {
          last.text += &substring.text;
          last.length += substring.length;
        } else if substring.lang == last_lang || last_lang.is_empty() {
          last.text += &substring.text;
          last.length += substring.length;
          if substring.lang != "punctuation" {
            last.lang = substring.lang.clone();
          }
        } else {
          new_substrings.push(substring);
        }
      } else {
        new_substrings.push(substring);
      }
      last_lang = substring_lang;
    }
    new_substrings
  }

  /// Port of `LangSplitter._merge_substrings_across_digit_based_on_sections(sections)`.
  fn _merge_substrings_across_digit_based_on_sections(
    &self,
    sections: Vec<SubStringSection>,
  ) -> Result<Vec<SubStringSection>, SplitLangError> {
    let mut iter = sections.into_iter();
    let first = iter.next().ok_or_else(index_error)?;
    let mut new_sections: Vec<SubStringSection> = vec![first];
    for current_section in iter {
      let last = new_sections.last_mut().expect("non-empty");
      let one_of_section_is_digit = last.lang_section_type == LangSectionType::Digit
        || current_section.lang_section_type == LangSectionType::Digit;
      if one_of_section_is_digit {
        let cur_first = at(&current_section.substrings, 0)?.clone();
        if last.lang_section_type == LangSectionType::Digit {
          if current_section.lang_section_type != LangSectionType::Punctuation {
            last.lang_section_type = current_section.lang_section_type;
          }
          last.text += &current_section.text;
          let last_sub = last_mut(&mut last.substrings)?;
          last_sub.text += &cur_first.text;
          last_sub.length += cur_first.length;
          if cur_first.lang != "punctuation" {
            last_sub.lang = cur_first.lang.clone();
          }
          last
            .substrings
            .extend(current_section.substrings.into_iter().skip(1));
          chain_indices(&mut last.substrings, 1);
        } else if current_section.lang_section_type == LangSectionType::Digit {
          last.text += &current_section.text;
          let last_sub = last_mut(&mut last.substrings)?;
          last_sub.text += &cur_first.text;
          last_sub.length += cur_first.length;
          if last_sub.lang == "punctuation" {
            last_sub.lang = "digit".to_string();
          }
          if last.lang_section_type == LangSectionType::Punctuation {
            last.lang_section_type = LangSectionType::Digit;
          }
        }
      } else {
        new_sections.push(current_section);
      }
    }
    let mut merged = merge_same_type_sections(new_sections);
    recompute_section_indices(&mut merged)?;
    for section in &mut merged {
      let substrings = std::mem::take(&mut section.substrings);
      section.substrings = self._merge_substrings_across_digit(substrings);
    }
    Ok(merged)
  }

  /// Port of `LangSplitter._merge_substrings_across_punctuation(substrings)`.
  fn _merge_substrings_across_punctuation(&self, substrings: Vec<SubString>) -> Vec<SubString> {
    Self::merge_substrings_across("punctuation", substrings)
  }

  /// Port of `LangSplitter._merge_substrings_across_punctuation_based_on_sections(sections)`.
  fn _merge_substrings_across_punctuation_based_on_sections(
    &self,
    sections: Vec<SubStringSection>,
  ) -> Result<Vec<SubStringSection>, SplitLangError> {
    let mut iter = sections.into_iter();
    let first = iter.next().ok_or_else(index_error)?;
    let mut new_sections: Vec<SubStringSection> = vec![first];
    for current_section in iter {
      let last = new_sections.last_mut().expect("non-empty");
      let one_of_section_is_punctuation = last.lang_section_type == LangSectionType::Punctuation
        || current_section.lang_section_type == LangSectionType::Punctuation;
      if !one_of_section_is_punctuation {
        new_sections.push(current_section);
        continue;
      }
      if last.lang_section_type == LangSectionType::Punctuation
        && !self
          .not_merge_punctuation
          .contains(at(&last.substrings, 0)?.text.as_str())
      {
        let cur_first = at(&current_section.substrings, 0)?.clone();
        last.text += &current_section.text;
        last.lang_section_type = current_section.lang_section_type;
        let last_sub = last_mut(&mut last.substrings)?;
        last_sub.text += &cur_first.text;
        last_sub.length += cur_first.length;
        last_sub.lang = cur_first.lang.clone();
        last
          .substrings
          .extend(current_section.substrings.into_iter().skip(1));
        chain_indices(&mut last.substrings, 1);
      } else if current_section.lang_section_type == LangSectionType::Punctuation
        && !self
          .not_merge_punctuation
          .contains(at(&current_section.substrings, 0)?.text.as_str())
      {
        let cur_first = current_section.substrings[0].clone();
        last.text += &current_section.text;
        let last_sub = last_mut(&mut last.substrings)?;
        last_sub.text += &cur_first.text;
        last_sub.length += cur_first.length;
        last
          .substrings
          .extend(current_section.substrings.into_iter().skip(1));
        chain_indices(&mut last.substrings, 1);
      } else {
        new_sections.push(current_section);
      }
    }
    let mut merged = merge_same_type_sections(new_sections);
    recompute_section_indices(&mut merged)?;
    for section in &mut merged {
      let substrings = std::mem::take(&mut section.substrings);
      section.substrings = self._merge_substrings_across_punctuation(substrings);
    }
    Ok(merged)
  }

  /// Port of `LangSplitter._init_substr_lang(texts, lang_section_type, lang_map)`.
  fn _init_substr_lang(
    &self,
    texts: &[String],
    lang_section_type: LangSectionType,
    lang_map: LangMapArg,
  ) -> Result<Vec<SubString>, SplitLangError> {
    let mut substrings = Vec::with_capacity(texts.len());
    let mut substring_index = 0;
    for text in texts {
      let length = py::len(text);
      let cur_lang = detect_lang_combined(text, lang_section_type, self.detector())?;
      let cur_lang = self.map_get(lang_map, &cur_lang);
      substrings.push(SubString {
        lang: cur_lang,
        text: text.clone(),
        length,
        index: substring_index,
      });
      substring_index += length;
    }
    Ok(substrings)
  }

  /// Port of `LangSplitter._get_languages(lang_text_list, lang_section_type, lang_map)`.
  ///
  /// As in Python, the detected language is computed and then discarded, so
  /// the list is returned unchanged; the detector is still called so that
  /// its errors propagate.
  fn _get_languages(
    &self,
    lang_text_list: Vec<SubString>,
    lang_section_type: LangSectionType,
    lang_map: LangMapArg,
  ) -> Result<Vec<SubString>, SplitLangError> {
    if matches!(
      lang_section_type,
      LangSectionType::Digit | LangSectionType::Ko | LangSectionType::Punctuation
    ) {
      return Ok(lang_text_list);
    }
    for substr in &lang_text_list {
      let cur_lang = detect_lang_combined(&substr.text, lang_section_type, self.detector())?;
      let _cur_lang = self.map_get(lang_map, &cur_lang);
    }
    Ok(lang_text_list)
  }
}
