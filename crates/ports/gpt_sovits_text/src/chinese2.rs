//! Port of `GPT_SoVITS/text/chinese2.py` (GPT-SoVITS @ 04a44eb).
//!
//! # Module-level state
//!
//! `chinese2.py` builds its state when it is imported. The port keeps the
//! constant parts as Rust constants and statics and puts the parts that need
//! the G2PW model into the [`Chinese2`] struct:
//!
//! | Python (module level) | Rust |
//! |---|---|
//! | `normalizer = lambda x: cn2an.transform(x, "an2cn")` | [`normalizer`] |
//! | `pinyin_to_symbol_map` (read from `opencpop-strict.txt`) | [`PINYIN_TO_SYMBOL_MAP`] (`data/chinese2/opencpop-strict.txt`, embedded) |
//! | `jieba_fast.setLogLevel(logging.CRITICAL)`, `import jieba_fast.posseg as psg` | `jieba_fast::posseg::lcut` (the port does not log) |
//! | `is_g2pw = True` | [`Chinese2::is_g2pw`] (`true` from [`Chinese2::new`], `false` from [`Chinese2::without_g2pw`]) |
//! | `g2pw = G2PWPinyin(model_dir="GPT_SoVITS/text/G2PWModel", model_source=bert_path, v_to_u=False, neutral_tone_with_five=True)` | [`Chinese2::g2pw`], built by [`Chinese2::new`] from a model dir argument and [`G2PWPinyinOptions::chinese2`] |
//! | `rep_map` | [`REP_MAP`] |
//! | `tone_modifier = ToneSandhi()` | [`Chinese2::tone_modifier`] |
//! | `must_erhua`, `not_erhua` | [`MUST_ERHUA`], [`NOT_ERHUA`] |
//!
//! The functions that use module state (`g2p`, `_g2p`) are methods of
//! [`Chinese2`]; the others are free functions. `text_normalize` builds its
//! own `TextNormalizer()` on every call, as Python does.
//!
//! `model_source`: Python passes the `bert_path` environment variable
//! (default `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`);
//! the `g2pw` port embeds the tokenizer of that model, which
//! `model_source: None` selects (see the `g2pw` README).
//!
//! # Ported scope
//!
//! Everything in the file: `normalizer`, `pinyin_to_symbol_map`, `rep_map`,
//! `replace_punctuation`, `g2p`, `_get_initials_finals`, `must_erhua`,
//! `not_erhua`, `_merge_erhua`, `_g2p` (the G2PW branch and the pypinyin
//! branch, with `v_rep_map`, `pinyin_rep_map` and `single_rep_map`),
//! `replace_punctuation_with_en`, `replace_consecutive_punctuation` and
//! `text_normalize`. Left out: the `__main__` demo.
//!
//! # Differences in form (not in output)
//!
//! Where Python raises, the Rust functions return an [`Error`] that names the
//! Python exception (see [`Error`]). Strings are indexed by code point, as
//! Python `str` is. `pinyin[0].isalpha()` uses a table of the Python 3.11
//! `str.isalpha` code points (`data/chinese2/python_isalpha.txt`), and the
//! `\s` of the `g2p` split pattern and `str.strip` use the Python whitespace
//! set.
//!
//! `correct_pronunciation` of the `g2pw` port takes `Vec<String>`, but the
//! G2PW result can hold `None` (the tokenizer `{}` path of `__call__`). For a
//! word whose pinyins contain `None` the port runs the same steps on
//! `Option<String>` values ([`correct_pronunciation_opt`]); otherwise it calls
//! [`correct_pronunciation`].
//!
//! The pypinyin branch prints `pypinyin结果 [...] [...]` to stdout like
//! Python. The list repr quotes with `'` (with `"` when a string has a `'`
//! and no `"`) and escapes `\\`, the quote and ASCII control characters; it
//! does not reproduce Python's escaping of other non-printable characters.
//! Stdout is not part of the checked output.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::sync::LazyLock;

use pypinyin::contrib::tone_convert::{to_finals_tone3, to_initials};
use pypinyin::{LazyPinyinOptions, Style, lazy_pinyin};

use crate::g2pw::g2pw::{G2PWPinyinOptions, PP_DICT};
use crate::g2pw::{G2PWError, G2PWPinyin, correct_pronunciation};
use crate::symbols::PUNCTUATION;
use crate::tone_sandhi::{self, ToneSandhi};
use crate::zh_normalization::pyre::{py_isspace, py_strip};
use crate::zh_normalization::{self, TextNormalizer};

/// A Python exception raised by `chinese2.py` or the code it calls.
#[derive(Debug)]
pub enum Error {
  /// `AssertionError`, with its `args`: empty for the bare
  /// `assert c in punctuation` and `assert tone in "12345"`, and
  /// `[pinyin, seg, raw_pinyin]` for
  /// `assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)`.
  AssertionError(Vec<String>),
  /// `IndexError(message)`: `"string index out of range"` or
  /// `"list index out of range"` (also from `ToneSandhi`).
  IndexError(String),
  /// `TypeError(message)`: `pinyin[0]` on a `None` G2PW result.
  TypeError(String),
  /// `KeyError(key)` from `TextNormalizer.normalize`.
  KeyError(String),
  /// `ValueError(message)` from `TextNormalizer.normalize`.
  ValueError(String),
  /// An exception of the G2PW converter (`g2pw._g2pw(batch_inputs)`) or of
  /// `correct_pronunciation`.
  G2PW(G2PWError),
}

impl Error {
  /// The name of the Python exception (`type(e).__name__`); `None` for the
  /// port's own G2PW failures (ONNX Runtime, tokenizer).
  pub fn name(&self) -> Option<&'static str> {
    match self {
      Error::AssertionError(_) => Some("AssertionError"),
      Error::IndexError(_) => Some("IndexError"),
      Error::TypeError(_) => Some("TypeError"),
      Error::KeyError(_) => Some("KeyError"),
      Error::ValueError(_) => Some("ValueError"),
      Error::G2PW(e) => e.exception_name(),
    }
  }
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Error::AssertionError(args) => write!(f, "AssertionError: {args:?}"),
      Error::IndexError(m) => write!(f, "IndexError: {m}"),
      Error::TypeError(m) => write!(f, "TypeError: {m}"),
      Error::KeyError(k) => write!(f, "KeyError: {k:?}"),
      Error::ValueError(m) => write!(f, "ValueError: {m}"),
      Error::G2PW(e) => write!(f, "{e}"),
    }
  }
}

impl std::error::Error for Error {}

impl From<zh_normalization::Error> for Error {
  fn from(e: zh_normalization::Error) -> Self {
    match e {
      zh_normalization::Error::KeyError(k) => Error::KeyError(k),
      zh_normalization::Error::ValueError(m) => Error::ValueError(m),
    }
  }
}

impl From<tone_sandhi::IndexError> for Error {
  fn from(e: tone_sandhi::IndexError) -> Self {
    Error::IndexError(e.0.to_string())
  }
}

impl From<G2PWError> for Error {
  fn from(e: G2PWError) -> Self {
    Error::G2PW(e)
  }
}

const STRING_INDEX: &str = "string index out of range";
const LIST_INDEX: &str = "list index out of range";

/// Port of `normalizer = lambda x: cn2an.transform(x, "an2cn")`. Nothing in
/// `chinese2.py` calls it.
pub fn normalizer(x: &str) -> cn2an::Result<String> {
  cn2an::transform(x, "an2cn", false)
}

/// `opencpop-strict.txt` (`GPT_SoVITS/text/opencpop-strict.txt`), embedded.
pub const OPENCPOP_STRICT: &str = include_str!("../data/chinese2/opencpop-strict.txt");

/// Port of `pinyin_to_symbol_map`: `{line.split("\t")[0]:
/// line.strip().split("\t")[1] for line in open(...).readlines()}`. A later
/// line with the same key overwrites the earlier one, as in the dict
/// comprehension.
pub static PINYIN_TO_SYMBOL_MAP: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
  let mut map = HashMap::new();
  for line in OPENCPOP_STRICT.split_inclusive('\n') {
    let key = line.split('\t').next().unwrap_or("");
    let value = py_strip(line)
      .split('\t')
      .nth(1)
      .expect("opencpop-strict.txt has a tab on every line");
    map.insert(key.to_string(), value.to_string());
  }
  map
});

/// Port of `rep_map`, in dict order (the order of the regex alternation).
pub const REP_MAP: &[(&str, &str)] = &[
  ("：", ","),
  ("；", ","),
  ("，", ","),
  ("。", "."),
  ("！", "!"),
  ("？", "?"),
  ("\n", "."),
  ("·", ","),
  ("、", ","),
  ("...", "…"),
  ("$", "."),
  ("/", ","),
  ("—", "-"),
  ("~", "…"),
  ("～", "…"),
];

/// Port of `must_erhua`.
pub const MUST_ERHUA: &[&str] = &[
  "小院儿",
  "胡同儿",
  "范儿",
  "老汉儿",
  "撒欢儿",
  "寻老礼儿",
  "妥妥儿",
  "媳妇儿",
];

/// Port of `not_erhua`.
pub const NOT_ERHUA: &[&str] = &[
  "虐儿",
  "为儿",
  "护儿",
  "瞒儿",
  "救儿",
  "替儿",
  "有儿",
  "一儿",
  "我儿",
  "俺儿",
  "妻儿",
  "拐儿",
  "聋儿",
  "乞儿",
  "患儿",
  "幼儿",
  "孤儿",
  "婴儿",
  "婴幼儿",
  "连体儿",
  "脑瘫儿",
  "流浪儿",
  "体弱儿",
  "混血儿",
  "蜜雪儿",
  "舫儿",
  "祖儿",
  "美儿",
  "应采儿",
  "可儿",
  "侄儿",
  "孙儿",
  "侄孙儿",
  "女儿",
  "男儿",
  "红孩儿",
  "花儿",
  "虫儿",
  "马儿",
  "鸟儿",
  "猪儿",
  "猫儿",
  "狗儿",
  "少儿",
];

static MUST_ERHUA_SET: LazyLock<HashSet<&'static str>> =
  LazyLock::new(|| MUST_ERHUA.iter().copied().collect());
static NOT_ERHUA_SET: LazyLock<HashSet<&'static str>> =
  LazyLock::new(|| NOT_ERHUA.iter().copied().collect());

/// `python_isalpha.txt`: the code point ranges of Python 3.11
/// `str.isalpha()`, generated by `parity/chinese2/convert_data.py`.
const PYTHON_ISALPHA: &str = include_str!("../data/chinese2/python_isalpha.txt");

static ISALPHA_RANGES: LazyLock<Vec<(u32, u32)>> = LazyLock::new(|| {
  PYTHON_ISALPHA
    .lines()
    .filter(|l| !l.is_empty())
    .map(|l| {
      let (a, b) = l.split_once('-').expect("range line");
      (
        u32::from_str_radix(a, 16).expect("hex"),
        u32::from_str_radix(b, 16).expect("hex"),
      )
    })
    .collect()
});

/// Python 3.11 `str.isalpha()` for a one-character string.
pub fn py_isalpha(c: char) -> bool {
  let cp = c as u32;
  let ranges = &*ISALPHA_RANGES;
  let i = ranges.partition_point(|&(_, hi)| hi < cp);
  i < ranges.len() && ranges[i].0 <= cp
}

/// `c in punctuation` for a one-character string `c` (`punctuation` is a
/// list of one-character strings).
fn is_punctuation_char(c: char) -> bool {
  let mut buf = [0u8; 4];
  PUNCTUATION.contains(&&*c.encode_utf8(&mut buf))
}

/// `pattern.sub(lambda x: rep_map[x.group()], text)` with the alternation of
/// the escaped `rep_map` keys, tried in dict order at each position.
fn sub_rep_map(text: &str) -> String {
  let mut replaced = String::with_capacity(text.len());
  let mut rest = text;
  'outer: while !rest.is_empty() {
    for (k, v) in REP_MAP {
      if rest.starts_with(k) {
        replaced.push_str(v);
        rest = &rest[k.len()..];
        continue 'outer;
      }
    }
    let c = rest.chars().next().unwrap();
    replaced.push(c);
    rest = &rest[c.len_utf8()..];
  }
  replaced
}

/// Port of `replace_punctuation(text)`.
///
/// `嗯` becomes `恩` and `呣` becomes `母`, the `rep_map` keys are replaced,
/// then every run of characters outside `[一-龥!?…,.-]` is removed.
pub fn replace_punctuation(text: &str) -> String {
  let text = text.replace('嗯', "恩").replace('呣', "母");
  sub_rep_map(&text)
    .chars()
    .filter(|&c| ('\u{4e00}'..='\u{9fa5}').contains(&c) || is_punctuation_char(c))
    .collect()
}

/// Port of `replace_punctuation_with_en(text)`: [`replace_punctuation`] that
/// also keeps `A-Za-z`.
pub fn replace_punctuation_with_en(text: &str) -> String {
  let text = text.replace('嗯', "恩").replace('呣', "母");
  sub_rep_map(&text)
    .chars()
    .filter(|&c| {
      ('\u{4e00}'..='\u{9fa5}').contains(&c) || c.is_ascii_alphabetic() || is_punctuation_char(c)
    })
    .collect()
}

/// Port of `replace_consecutive_punctuation(text)`:
/// `re.sub(f"([{punctuations}])([{punctuations}])+", r"\1", text)`, which
/// keeps the first character of every run of two or more punctuation marks.
pub fn replace_consecutive_punctuation(text: &str) -> String {
  let mut result = String::with_capacity(text.len());
  let mut prev_punct = false;
  for c in text.chars() {
    let punct = is_punctuation_char(c);
    if !(punct && prev_punct) {
      result.push(c);
    }
    prev_punct = punct;
  }
  result
}

/// Port of `text_normalize(text)`: `TextNormalizer().normalize(text)`,
/// [`replace_punctuation`] of each sentence, concatenated, then
/// [`replace_consecutive_punctuation`].
pub fn text_normalize(text: &str) -> Result<String, Error> {
  let tx = TextNormalizer::new();
  let sentences = tx.normalize(text)?;
  let mut dest_text = String::new();
  for sentence in &sentences {
    dest_text.push_str(&replace_punctuation(sentence));
  }
  Ok(replace_consecutive_punctuation(&dest_text))
}

/// `re.split(r"(?<=[{punctuation}])\s*", text)`.
///
/// The pattern matches at every position right after a punctuation mark and
/// takes the whitespace that follows. A match always ends after whitespace
/// or is empty, so the next match starts at a later position; the pieces
/// are the text between the matches (the last one can be empty).
fn split_after_punctuation(text: &str) -> Vec<&str> {
  let chars: Vec<(usize, char)> = text.char_indices().collect();
  let n = chars.len();
  let byte = |k: usize| chars.get(k).map_or(text.len(), |&(i, _)| i);
  let mut pieces = Vec::new();
  let mut last = 0;
  let mut k = 0;
  while k <= n {
    if k > 0 && is_punctuation_char(chars[k - 1].1) {
      let mut e = k;
      while e < n && py_isspace(chars[e].1) {
        e += 1;
      }
      pieces.push(&text[last..byte(k)]);
      last = byte(e);
      // After a non-empty match the next position follows whitespace, so
      // it cannot match; after an empty match the search moves on by one.
      k = if e > k { e } else { k + 1 };
    } else {
      k += 1;
    }
  }
  pieces.push(&text[last..]);
  pieces
}

/// `re.sub("[a-zA-Z]+", "", seg)`.
fn remove_ascii_letters(seg: &str) -> String {
  seg.chars().filter(|c| !c.is_ascii_alphabetic()).collect()
}

/// Port of `_get_initials_finals(word)`: `lazy_pinyin(word,
/// neutral_tone_with_five=True, style=Style.INITIALS)` and the same with
/// `Style.FINALS_TONE3`, zipped (so cut to the shorter list).
pub fn _get_initials_finals(word: &str) -> (Vec<String>, Vec<String>) {
  let mut initials = Vec::new();
  let mut finals = Vec::new();

  let orig_initials = lazy_pinyin(
    word,
    &LazyPinyinOptions {
      style: Style::INITIALS,
      neutral_tone_with_five: true,
      ..LazyPinyinOptions::default()
    },
  );
  let orig_finals = lazy_pinyin(
    word,
    &LazyPinyinOptions {
      style: Style::FINALS_TONE3,
      neutral_tone_with_five: true,
      ..LazyPinyinOptions::default()
    },
  );

  for (c, v) in orig_initials.into_iter().zip(orig_finals) {
    initials.push(c);
    finals.push(v);
  }
  (initials, finals)
}

/// `word[-2:]`.
fn last_two(word: &[char]) -> String {
  word[word.len().saturating_sub(2)..].iter().collect()
}

/// Port of `_merge_erhua(initials, finals, word, pos)` ("Do erhub.").
///
/// A final `er1` of a word-final `儿` becomes `er2`. Words in
/// [`MUST_ERHUA`], or not in [`NOT_ERHUA`] with a `pos` other than `a`, `j`
/// and `nr`, whose final count equals the word length, then give a final
/// `er2` / `er5` of a word-final `儿` the tone of the previous final
/// (unless `word[-2:]` is in [`NOT_ERHUA`]).
///
/// `word[i]` past the end of the word (more finals than characters) gives
/// `IndexError`, as do an empty previous final (`new_finals[-1][-1]`) and an
/// initial list shorter than the final list.
pub fn _merge_erhua(
  initials: Vec<String>,
  mut finals: Vec<String>,
  word: &str,
  pos: &str,
) -> Result<(Vec<String>, Vec<String>), Error> {
  let wchars: Vec<char> = word.chars().collect();
  // fix er1: Python loops over every final, but the condition
  // `i == len(finals) - 1 and word[i] == "儿" and phn == "er1"` only holds
  // (and only reads `word[i]`) for the last one.
  let n = finals.len();
  if let Some(i) = n.checked_sub(1) {
    let wc = *wchars
      .get(i)
      .ok_or_else(|| Error::IndexError(STRING_INDEX.into()))?;
    if wc == '儿' && finals[i] == "er1" {
      finals[i] = "er2".to_string();
    }
  }

  // 发音
  if !MUST_ERHUA_SET.contains(word)
    && (NOT_ERHUA_SET.contains(word) || matches!(pos, "a" | "j" | "nr"))
  {
    return Ok((initials, finals));
  }

  // "……" 等情况直接返回
  if finals.len() != wchars.len() {
    return Ok((initials, finals));
  }

  // 与前一个字发同音
  let mut new_initials: Vec<String> = Vec::new();
  let mut new_finals: Vec<String> = Vec::new();
  for (i, phn) in finals.into_iter().enumerate() {
    let mut phn = phn;
    if i == n - 1
      && wchars[i] == '儿'
      && (phn == "er2" || phn == "er5")
      && !NOT_ERHUA_SET.contains(last_two(&wchars).as_str())
      && !new_finals.is_empty()
    {
      let last = new_finals
        .last()
        .unwrap()
        .chars()
        .last()
        .ok_or_else(|| Error::IndexError(STRING_INDEX.into()))?;
      phn = format!("er{last}");
    }

    new_initials.push(
      initials
        .get(i)
        .cloned()
        .ok_or_else(|| Error::IndexError(LIST_INDEX.into()))?,
    );
    new_finals.push(phn);
  }

  Ok((new_initials, new_finals))
}

/// [`correct_pronunciation`] on a G2PW result that can hold `None`: the same
/// steps on `Option<String>` values. A `pp_dict` entry for the whole word
/// replaces the list; otherwise each character with an entry replaces
/// `word_pinyins[idx]` with the first pinyin of its entry (`IndexError` past
/// the end).
pub fn correct_pronunciation_opt(
  word: &str,
  word_pinyins: Vec<Option<String>>,
) -> Result<Vec<Option<String>>, Error> {
  if word_pinyins.iter().all(Option::is_some) {
    let pinyins = word_pinyins.into_iter().map(Option::unwrap).collect();
    return Ok(
      correct_pronunciation(word, pinyins)?
        .into_iter()
        .map(Some)
        .collect(),
    );
  }
  match PP_DICT.get(word) {
    Some(new_pinyins) => Ok(new_pinyins.iter().cloned().map(Some).collect()),
    None => {
      let mut word_pinyins = word_pinyins;
      for (idx, w) in word.chars().enumerate() {
        if let Some(w_pinyin) = PP_DICT.get(w.to_string().as_str()) {
          let first = w_pinyin
            .first()
            .ok_or_else(|| Error::IndexError(LIST_INDEX.into()))?;
          let slot = word_pinyins
            .get_mut(idx)
            .ok_or_else(|| Error::IndexError("list assignment index out of range".into()))?;
          *slot = Some(first.clone());
        }
      }
      Ok(word_pinyins)
    }
  }
}

/// Python `repr` of a `list[str]`, for the debug print of the pypinyin
/// branch (see the module docs for what it leaves out).
fn py_repr_list(items: &[String]) -> String {
  let parts: Vec<String> = items
    .iter()
    .map(|s| {
      let quote = if s.contains('\'') && !s.contains('"') {
        '"'
      } else {
        '\''
      };
      let mut out = String::new();
      out.push(quote);
      for c in s.chars() {
        match c {
          '\\' => out.push_str("\\\\"),
          '\n' => out.push_str("\\n"),
          '\r' => out.push_str("\\r"),
          '\t' => out.push_str("\\t"),
          c if c == quote => {
            out.push('\\');
            out.push(c);
          }
          c if (c as u32) < 0x20 || c as u32 == 0x7f => {
            out.push_str(&format!("\\x{:02x}", c as u32))
          }
          c => out.push(c),
        }
      }
      out.push(quote);
      out
    })
    .collect();
  format!("[{}]", parts.join(", "))
}

/// The module-level singletons of `chinese2.py` that `g2p` uses (see the
/// module docs for the mapping).
pub struct Chinese2 {
  /// `is_g2pw` (`True` upstream).
  pub is_g2pw: bool,
  /// `g2pw` (only built when `is_g2pw`).
  pub g2pw: Option<G2PWPinyin>,
  /// `tone_modifier = ToneSandhi()`.
  pub tone_modifier: ToneSandhi,
}

impl fmt::Debug for Chinese2 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Chinese2")
      .field("is_g2pw", &self.is_g2pw)
      .field("tone_modifier", &self.tone_modifier)
      .finish_non_exhaustive()
  }
}

impl Chinese2 {
  /// The module as GPT-SoVITS imports it: `is_g2pw = True` and
  /// `g2pw = G2PWPinyin(model_dir=..., model_source=bert_path, v_to_u=False,
  /// neutral_tone_with_five=True)`, with `model_dir` the extracted
  /// `G2PWModel` directory (Python hardcodes `GPT_SoVITS/text/G2PWModel`).
  pub fn new(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
    let g2pw = G2PWPinyin::new(model_dir, &G2PWPinyinOptions::chinese2())?;
    Ok(Self {
      is_g2pw: true,
      g2pw: Some(g2pw),
      tone_modifier: ToneSandhi::new(),
    })
  }

  /// The module with `is_g2pw = False` (the commented-out setting upstream):
  /// `_g2p` takes the pypinyin branch and no G2PW model is loaded.
  pub fn without_g2pw() -> Self {
    Self {
      is_g2pw: false,
      g2pw: None,
      tone_modifier: ToneSandhi::new(),
    }
  }

  /// Port of `g2p(text)`: split after each punctuation mark (and the
  /// whitespace after it), drop the pieces that are empty after `strip()`,
  /// then [`Chinese2::_g2p`]. Returns `(phones, word2ph)`.
  pub fn g2p(&self, text: &str) -> Result<(Vec<String>, Vec<usize>), Error> {
    let sentences: Vec<&str> = split_after_punctuation(text)
      .into_iter()
      .filter(|i| !py_strip(i).is_empty())
      .collect();
    self._g2p(&sentences)
  }

  /// Port of `_g2p(segments)`.
  ///
  /// # Panics
  ///
  /// When `is_g2pw` is set and `g2pw` is `None` (Python would raise
  /// `NameError`); [`Chinese2::new`] always sets both.
  pub fn _g2p<S: AsRef<str>>(&self, segments: &[S]) -> Result<(Vec<String>, Vec<usize>), Error> {
    let mut phones_list: Vec<String> = Vec::new();
    let mut word2ph: Vec<usize> = Vec::new();
    let mut g2pw_batch_results: Vec<Vec<Option<String>>> = Vec::new();
    let mut g2pw_batch_cursor = 0;
    let processed_segments: Vec<String> = segments
      .iter()
      .map(|seg| remove_ascii_letters(seg.as_ref()))
      .collect();
    if self.is_g2pw {
      let batch_inputs: Vec<&str> = processed_segments
        .iter()
        .filter(|seg| !seg.is_empty())
        .map(String::as_str)
        .collect();
      if !batch_inputs.is_empty() {
        let g2pw = self.g2pw.as_ref().expect("is_g2pw requires g2pw");
        g2pw_batch_results = g2pw._g2pw.call(&batch_inputs)?;
      }
    }

    for seg in &processed_segments {
      let mut pinyins: Vec<Option<String>> = Vec::new();
      let seg_cut: Vec<(String, String)> = jieba_fast::posseg::lcut(seg, true)
        .into_iter()
        .map(|p| (p.word, p.flag))
        .collect();
      let seg_cut = self.tone_modifier.pre_merge_for_modify(&seg_cut);
      let mut initials: Vec<String> = Vec::new();
      let mut finals: Vec<String> = Vec::new();

      if !self.is_g2pw {
        for (word, pos) in &seg_cut {
          if pos == "eng" {
            continue;
          }
          let (sub_initials, sub_finals) = _get_initials_finals(word);
          let sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)?;
          // 儿化
          let (sub_initials, sub_finals) = _merge_erhua(sub_initials, sub_finals, word, pos)?;
          initials.extend(sub_initials);
          finals.extend(sub_finals);
        }
        println!(
          "pypinyin结果 {} {}",
          py_repr_list(&initials),
          py_repr_list(&finals)
        );
      } else {
        // g2pw采用整句推理（批量推理，逐句取结果）
        if !seg.is_empty() {
          pinyins = g2pw_batch_results[g2pw_batch_cursor].clone();
          g2pw_batch_cursor += 1;
        }

        let mut pre_word_length = 0;
        for (word, pos) in &seg_cut {
          let mut sub_initials = Vec::new();
          let mut sub_finals = Vec::new();
          let now_word_length = pre_word_length + word.chars().count();

          if pos == "eng" {
            pre_word_length = now_word_length;
            continue;
          }

          // pinyins[pre_word_length:now_word_length]
          let lo = pre_word_length.min(pinyins.len());
          let hi = now_word_length.min(pinyins.len()).max(lo);
          let word_pinyins = pinyins[lo..hi].to_vec();

          // 多音字消歧
          let word_pinyins = correct_pronunciation_opt(word, word_pinyins)?;

          for pinyin in word_pinyins {
            let Some(pinyin) = pinyin else {
              return Err(Error::TypeError(
                "'NoneType' object is not subscriptable".into(),
              ));
            };
            let first = pinyin
              .chars()
              .next()
              .ok_or_else(|| Error::IndexError(STRING_INDEX.into()))?;
            if py_isalpha(first) {
              sub_initials.push(to_initials(&pinyin, true));
              sub_finals.push(to_finals_tone3(&pinyin, true, false, true));
            } else {
              sub_initials.push(pinyin.clone());
              sub_finals.push(pinyin);
            }
          }

          pre_word_length = now_word_length;
          let sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)?;
          // 儿化
          let (sub_initials, sub_finals) = _merge_erhua(sub_initials, sub_finals, word, pos)?;
          initials.extend(sub_initials);
          finals.extend(sub_finals);
        }
      }

      for (c, v) in initials.iter().zip(finals.iter()) {
        let raw_pinyin = format!("{c}{v}");
        // NOTE: post process for pypinyin outputs
        // we discriminate i, ii and iii
        let phone: Vec<String> = if c == v {
          if !PUNCTUATION.contains(&c.as_str()) {
            return Err(Error::AssertionError(Vec::new()));
          }
          word2ph.push(1);
          vec![c.clone()]
        } else {
          let vchars: Vec<char> = v.chars().collect();
          let tone = *vchars
            .last()
            .ok_or_else(|| Error::IndexError(STRING_INDEX.into()))?;
          let v_without_tone: String = vchars[..vchars.len() - 1].iter().collect();

          let mut pinyin = format!("{c}{v_without_tone}");
          if !"12345".contains(tone) {
            return Err(Error::AssertionError(Vec::new()));
          }

          if !c.is_empty() {
            // 多音节
            let v_rep = match v_without_tone.as_str() {
              "uei" => Some("ui"),
              "iou" => Some("iu"),
              "uen" => Some("un"),
              _ => None,
            };
            if let Some(rep) = v_rep {
              pinyin = format!("{c}{rep}");
            }
          } else {
            // 单音节
            let pinyin_rep = match pinyin.as_str() {
              "ing" => Some("ying"),
              "i" => Some("yi"),
              "in" => Some("yin"),
              "u" => Some("wu"),
              _ => None,
            };
            if let Some(rep) = pinyin_rep {
              pinyin = rep.to_string();
            } else {
              let first = pinyin
                .chars()
                .next()
                .ok_or_else(|| Error::IndexError(STRING_INDEX.into()))?;
              let single_rep = match first {
                'v' => Some("yu"),
                'e' => Some("e"),
                'i' => Some("y"),
                'u' => Some("w"),
                _ => None,
              };
              if let Some(rep) = single_rep {
                pinyin = format!("{rep}{}", &pinyin[first.len_utf8()..]);
              }
            }
          }

          let Some(symbol) = PINYIN_TO_SYMBOL_MAP.get(&pinyin) else {
            return Err(Error::AssertionError(vec![pinyin, seg.clone(), raw_pinyin]));
          };
          let (new_c, new_v) = symbol
            .split_once(' ')
            .expect("every opencpop-strict.txt value is `initial final`");
          let phone = vec![new_c.to_string(), format!("{new_v}{tone}")];
          word2ph.push(phone.len());
          phone
        };

        phones_list.extend(phone);
      }
    }
    Ok((phones_list, word2ph))
  }
}
