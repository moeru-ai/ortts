//! Port of `GPT_SoVITS/text/LangSegmenter/langsegmenter.py` (GPT-SoVITS @ 04a44eb).
//!
//! # Language detection model
//!
//! At import time the Python module replaces fast-langdetect's module-level
//! default detector so split-lang uses the full `lid.176.bin` from
//! `GPT_SoVITS/pretrained_models/fast_langdetect`:
//!
//! ```python
//! fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
//!     fast_langdetect.infer.LangDetectConfig(cache_dir=.../"pretrained_models"/"fast_langdetect"))
//! ```
//!
//! Rust has no import-time side effects, so call [`set_fast_langdetect_cache_dir`]
//! (the same replacement, with the directory as an argument) or
//! [`fast_langdetect::set_default_detector`] once before
//! [`LangSegmenter::getTexts`]. Without it, fast-langdetect's own default
//! (`$FTLANG_CACHE` or `<tempdir>/fasttext-langdetect`) is used, and a
//! missing model is an error (the port never downloads it).
//!
//! The `jieba.setLogLevel` call at the top of the Python module only silences
//! logging and has no counterpart.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::LazyLock;

use fast_langdetect::{LangDetectConfig, LangDetector};
use regex::Regex;
use split_lang::SplitLangError;
use split_lang::split::splitter::{LangSplitter, LangSplitterOptions, to_owned_map};

/// Port of the module-level `fast_langdetect.infer._default_detector = LangDetector(LangDetectConfig(cache_dir=...))`.
///
/// `cache_dir` is the directory holding `lid.176.bin` (upstream:
/// `GPT_SoVITS/pretrained_models/fast_langdetect`).
pub fn set_fast_langdetect_cache_dir(cache_dir: impl Into<PathBuf>) {
  fast_langdetect::set_default_detector(LangDetector::new(LangDetectConfig {
    cache_dir: cache_dir.into(),
    ..LangDetectConfig::default()
  }));
}

/// One `{'lang': ..., 'text': ...}` dict of the `getTexts` result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LangText {
  /// `'lang'`.
  pub lang: String,
  /// `'text'`.
  pub text: String,
}

impl LangText {
  fn new(lang: impl Into<String>, text: impl Into<String>) -> Self {
    Self {
      lang: lang.into(),
      text: text.into(),
    }
  }
}

/// A Python dict shared by reference (lists in `getTexts` alias them).
type Item = Rc<RefCell<LangText>>;

fn item(lang: impl Into<String>, text: impl Into<String>) -> Item {
  Rc::new(RefCell::new(LangText::new(lang, text)))
}

/// Python `str.isspace()` for one character (Rust `White_Space` plus
/// U+001C..U+001F).
fn py_isspace(c: char) -> bool {
  c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Port of `full_en(text)`:
/// `re.match(pattern, text)` with `pattern = r"^(?=.*[A-Za-z])[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$"`.
///
/// The lookahead `.*` does not cross `\n`, so a letter must appear before the
/// first newline. Every character must be in the class (`\s` is Python's
/// Unicode whitespace); the `$` before a final `\n` changes nothing because
/// `\n` is in the class.
pub fn full_en(text: &str) -> bool {
  let first_line = text.split('\n').next().unwrap_or("");
  if !first_line.chars().any(|c| c.is_ascii_alphabetic()) {
    return false;
  }
  !text.is_empty()
    && text.chars().all(|c| {
      c.is_ascii_alphanumeric()
        || py_isspace(c)
        || ('\u{0020}'..='\u{007E}').contains(&c)
        || ('\u{2000}'..='\u{206F}').contains(&c)
        || ('\u{3000}'..='\u{303F}').contains(&c)
        || ('\u{FF00}'..='\u{FFEF}').contains(&c)
    })
}

/// `cjk_ranges` of `full_cjk`.
const CJK_RANGES: [(u32, u32); 10] = [
  (0x4E00, 0x9FFF),
  (0x3400, 0x4DB5),
  (0x20000, 0x2A6DD),
  (0x2A700, 0x2B73F),
  (0x2B740, 0x2B81F),
  (0x2B820, 0x2CEAF),
  (0x2CEB0, 0x2EBEF),
  (0x30000, 0x3134A),
  (0x31350, 0x323AF),
  (0x2EBF0, 0x2EE5D),
];

/// `re.match(r'[0-9、-〜。！？.!?… /]+$', char)` for one character.
fn full_cjk_punct(c: char) -> bool {
  c.is_ascii_digit()
    || ('、'..='〜').contains(&c)
    || matches!(c, '。' | '！' | '？' | '.' | '!' | '?' | '…' | ' ' | '/')
}

/// Port of `full_cjk(text)`: the characters of `text` that are in
/// `cjk_ranges` or match `[0-9、-〜。！？.!?… /]`.
pub fn full_cjk(text: &str) -> String {
  text
    .chars()
    .filter(|&c| {
      let cp = c as u32;
      CJK_RANGES.iter().any(|&(s, e)| s <= cp && cp <= e) || full_cjk_punct(c)
    })
    .collect()
}

static JA_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
  Regex::new(
    r"([\x{3041}-\x{3096}\x{3099}\x{309A}\x{30A1}-\x{30FA}\x{30FC}]+(?:[0-9\x{3001}-\x{301C}。！？.!?… ]+[\x{3041}-\x{3096}\x{3099}\x{309A}\x{30A1}-\x{30FA}\x{30FC}]*)*)",
  )
  .expect("ja pattern")
});
static KO_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
  Regex::new(
    r"([\x{1100}-\x{11FF}\x{3130}-\x{318F}\x{AC00}-\x{D7AF}]+(?:[0-9\x{3001}-\x{301C}。！？.!?… ]+[\x{1100}-\x{11FF}\x{3130}-\x{318F}\x{AC00}-\x{D7AF}]*)*)",
  )
  .expect("ko pattern")
});

/// Port of `split_jako(tag_lang, item)`: splits the kana (`tag_lang ==
/// "ja"`) or Hangul (any other `tag_lang`) runs out of `item`.
pub fn split_jako(tag_lang: &str, item: &LangText) -> Vec<LangText> {
  let pattern = if tag_lang == "ja" {
    &*JA_PATTERN
  } else {
    &*KO_PATTERN
  };
  let text = item.text.as_str();
  let mut lang_list = Vec::new();
  let mut tag = 0;
  for m in pattern.find_iter(text) {
    if m.start() > tag {
      lang_list.push(LangText::new(item.lang.clone(), &text[tag..m.start()]));
    }
    tag = m.end();
    lang_list.push(LangText::new(tag_lang, m.as_str()));
  }
  if tag < text.len() {
    lang_list.push(LangText::new(item.lang.clone(), &text[tag..]));
  }
  lang_list
}

fn split_jako_items(tag_lang: &str, it: &Item) -> Vec<Item> {
  split_jako(tag_lang, &it.borrow())
    .into_iter()
    .map(|l| Rc::new(RefCell::new(l)))
    .collect()
}

/// Port of `merge_lang(lang_list, item)` on aliased dicts.
fn merge_lang_items(lang_list: &mut Vec<Item>, it: Item) {
  if let Some(last) = lang_list.last()
    && it.borrow().lang == last.borrow().lang
  {
    // `lang_list[-1]['text'] += item['text']`; `item` may be `lang_list[-1]`.
    let add = it.borrow().text.clone();
    last.borrow_mut().text += &add;
    return;
  }
  lang_list.push(it);
}

/// Port of `merge_lang(lang_list, item)`.
pub fn merge_lang(mut lang_list: Vec<LangText>, item: LangText) -> Vec<LangText> {
  if let Some(last) = lang_list.last_mut()
    && item.lang == last.lang
  {
    last.text += &item.text;
  } else {
    lang_list.push(item);
  }
  lang_list
}

/// `IndexError('string index out of range')`.
fn string_index_error() -> SplitLangError {
  SplitLangError::IndexError("string index out of range".into())
}

/// `s[-1]` / `s[0]` of a Python string, or `IndexError`.
fn last_char(s: &str) -> Result<char, SplitLangError> {
  s.chars().next_back().ok_or_else(string_index_error)
}
fn first_char(s: &str) -> Result<char, SplitLangError> {
  s.chars().next().ok_or_else(string_index_error)
}

const SENTENCE_PUNCT: [char; 8] = [',', '.', '!', '?', '，', '。', '！', '？'];

/// Port of `class LangSegmenter`.
pub struct LangSegmenter;

impl LangSegmenter {
  /// Port of `LangSegmenter.DEFAULT_LANG_MAP` (默认过滤器, 基于gsv目前四种语言).
  pub const DEFAULT_LANG_MAP: &'static [(&'static str, &'static str)] = &[
    ("zh", "zh"),
    ("yue", "zh"),
    ("wuu", "zh"),
    ("zh-cn", "zh"),
    ("zh-tw", "x"),
    ("ko", "ko"),
    ("ja", "ja"),
    ("en", "en"),
  ];

  /// Port of `LangSegmenter.getTexts(text, default_lang="")`.
  ///
  /// Errors are the Python exceptions `getTexts` lets propagate (for example
  /// `IndexError` from split-lang when `text.strip()` is empty).
  #[allow(non_snake_case)]
  pub fn getTexts(text: &str, default_lang: &str) -> Result<Vec<LangText>, SplitLangError> {
    let mut lang_splitter = LangSplitter::new(LangSplitterOptions {
      lang_map: to_owned_map(Self::DEFAULT_LANG_MAP),
      ..LangSplitterOptions::default()
    });
    lang_splitter.merge_across_digit = false;
    let substr = lang_splitter.split_by_lang(text)?;

    let mut lang_list: Vec<Item> = Vec::new();
    let mut have_num = false;

    for sub in &substr {
      let mut dict_item = item(sub.lang.clone(), sub.text.clone());

      if dict_item.borrow().lang == "digit" {
        if !default_lang.is_empty() {
          dict_item.borrow_mut().lang = default_lang.to_string();
        } else {
          have_num = true;
        }
        merge_lang_items(&mut lang_list, dict_item);
        continue;
      }

      // 处理短英文被识别为其他语言的问题
      if full_en(&dict_item.borrow().text) {
        dict_item.borrow_mut().lang = "en".to_string();
        merge_lang_items(&mut lang_list, dict_item);
        continue;
      }

      if !default_lang.is_empty() {
        dict_item.borrow_mut().lang = default_lang.to_string();
        merge_lang_items(&mut lang_list, dict_item);
        continue;
      }

      // 处理非日语夹日文的问题(不包含CJK)
      let mut ja_list: Vec<Item> = Vec::new();
      if dict_item.borrow().lang != "ja" {
        ja_list = split_jako_items("ja", &dict_item);
      }
      if ja_list.is_empty() {
        ja_list.push(dict_item.clone());
      }

      // 处理非韩语夹韩语的问题(不包含CJK)
      // `ko_list` is not reset per item, as in Python.
      let mut ko_list: Vec<Item> = Vec::new();
      let mut temp_list: Vec<Item> = Vec::new();
      for ko_item in &ja_list {
        if ko_item.borrow().lang != "ko" {
          ko_list = split_jako_items("ko", ko_item);
        }
        if !ko_list.is_empty() {
          temp_list.extend(ko_list.iter().cloned());
        } else {
          temp_list.push(ko_item.clone());
        }
      }

      // 未存在非日韩文夹日韩文
      if temp_list.len() == 1 {
        // 未知语言检查是否为CJK
        if dict_item.borrow().lang == "x" {
          let cjk_text = full_cjk(&dict_item.borrow().text);
          if !cjk_text.is_empty() {
            dict_item = item("zh", cjk_text);
          }
        }
        merge_lang_items(&mut lang_list, dict_item);
        continue;
      }

      // 存在非日韩文夹日韩文
      for temp_item in temp_list {
        if temp_item.borrow().lang == "x" {
          let cjk_text = full_cjk(&temp_item.borrow().text);
          if !cjk_text.is_empty() {
            merge_lang_items(&mut lang_list, item("zh", cjk_text));
          } else {
            merge_lang_items(&mut lang_list, temp_item);
          }
        } else {
          merge_lang_items(&mut lang_list, temp_item);
        }
      }
    }

    // 有数字
    if have_num {
      let temp_list = std::mem::take(&mut lang_list);
      let n = temp_list.len();
      for (i, temp_item) in temp_list.iter().enumerate() {
        if temp_item.borrow().lang == "digit" {
          let new_lang = if !default_lang.is_empty() {
            default_lang.to_string()
          } else if !lang_list.is_empty() && i == n - 1 {
            lang_list.last().unwrap().borrow().lang.clone()
          } else if lang_list.is_empty() && i + 1 < n {
            temp_list[1].borrow().lang.clone()
          } else if !lang_list.is_empty() && i + 1 < n {
            let prev = lang_list.last().unwrap().borrow().clone();
            let next = temp_list[i + 1].borrow().clone();
            // One branch per Python `elif`, kept separate to mirror the source.
            #[allow(clippy::if_same_then_else)]
            if prev.lang == next.lang {
              prev.lang
            } else if SENTENCE_PUNCT.contains(&last_char(&prev.text)?) {
              next.lang
            } else if SENTENCE_PUNCT.contains(&first_char(&next.text)?) {
              prev.lang
            } else if matches!(last_char(&temp_item.borrow().text)?, '。' | '.') {
              prev.lang
            } else if prev.text.chars().count() >= next.text.chars().count() {
              prev.lang
            } else {
              next.lang
            }
          } else {
            "zh".to_string()
          };
          temp_item.borrow_mut().lang = new_lang;
        }
        merge_lang_items(&mut lang_list, temp_item.clone());
      }
    }

    // 筛X
    let temp_list = std::mem::take(&mut lang_list);
    let n = temp_list.len();
    for temp_item in &temp_list {
      if temp_item.borrow().lang == "x" {
        let new_lang = if let Some(last) = lang_list.last() {
          last.borrow().lang.clone()
        } else if n > 1 {
          temp_list[1].borrow().lang.clone()
        } else {
          "zh".to_string()
        };
        temp_item.borrow_mut().lang = new_lang;
      }
      merge_lang_items(&mut lang_list, temp_item.clone());
    }

    Ok(
      lang_list
        .into_iter()
        .map(|it| it.borrow().clone())
        .collect(),
    )
  }
}
