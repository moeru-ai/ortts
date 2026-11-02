//! Port of `GPT_SoVITS/text/english.py` (GPT-SoVITS @ 04a44eb).
//!
//! # Ported scope
//!
//! Everything in the module except the pickle cache writer and the BERT-free
//! `__main__` demo: the path constants (as embedded file contents), `rep_map`,
//! `arpa`, `replace_phs`, `replace_consecutive_punctuation`, `read_dict`,
//! `read_dict_new`, `hot_reload_hot`, `get_dict`, `get_namedict`,
//! `text_normalize`, the `en_G2p` class (`__init__`, `__call__`, `qryword`),
//! the module-level `_g2p` instance and `g2p`.
//!
//! # Data
//!
//! `get_dict` loads `engdict_cache.pickle` when it exists (it does upstream),
//! and otherwise builds the dict with `read_dict_new`. The pickle is exactly
//! `read_dict_new()` (`parity/english/gen_data.py` checks
//! this), so the port always builds it from the embedded `cmudict.rep` and
//! `cmudict-fast.rep`, then applies `hot_reload_hot` with `engdict-hot.rep`.
//! `namedict_cache.pickle` is embedded as `data/english/namedict.tsv`,
//! converted by the same script.
//!
//! # Differences in form (not in output)
//!
//! - Where Python raises, the functions return an [`Error`] that names the
//!   Python exception: `KeyError` from `self.cmu[w]` for a single alphabetic
//!   character that is not in the dictionary (for example `é` in an
//!   unnormalized text), `IndexError` from `phones[-1]` when the stem of a
//!   possessive has no phones, and `TimeoutError` from the nltk tokenizer.
//!   `read_dict` raises `IndexError` on the embedded `cmudict.rep` (line 49
//!   has no two-space separator), as it does in Python.
//! - `replace_phs` prints `ph not in symbols:` for each dropped phone. The
//!   port drops the phone without printing.
//! - `cache_dict` (writes a pickle) is not ported.
//! - CPython string semantics are kept: `str.lower` is
//!   [`nltk::python::lower`], `str.strip` uses [`nltk::python::char_isspace`],
//!   and `str.isalpha` / `str.istitle` use tables generated from Python 3.11
//!   (`data/english/py_str.json`). `len()` and slicing count code points.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::LazyLock;

use g2p_en::G2p;
use nltk::python::{char_isspace, lower};
use nltk::{TweetTokenizer, pos_tag};
use serde::Deserialize;

use crate::en_normalization::{self, normalize, pyre::PyRegex};
use crate::{symbols, symbols2};

/// Contents of `CMU_DICT_PATH` (`GPT_SoVITS/text/cmudict.rep`).
pub const CMU_DICT: &str = include_str!("../data/english/cmudict.rep");
/// Contents of `CMU_DICT_FAST_PATH` (`GPT_SoVITS/text/cmudict-fast.rep`).
pub const CMU_DICT_FAST: &str = include_str!("../data/english/cmudict-fast.rep");
/// Contents of `CMU_DICT_HOT_PATH` (`GPT_SoVITS/text/engdict-hot.rep`).
pub const CMU_DICT_HOT: &str = include_str!("../data/english/engdict-hot.rep");
/// Contents of `NAMECACHE_PATH` (`GPT_SoVITS/text/namedict_cache.pickle`),
/// converted to text by `parity/english/gen_data.py`: one
/// line per entry, the key and then one tab-separated field per
/// pronunciation (phones joined by spaces).
pub const NAMECACHE: &str = include_str!("../data/english/namedict.tsv");

/// The pronunciation dictionaries: word to a list of pronunciations, each a
/// list of phones (`dict[str, list[list[str]]]`).
pub type G2pDict = HashMap<String, Vec<Vec<String>>>;

/// An exception raised by the ported code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
  /// `KeyError(key)` from a dictionary lookup (`self.cmu[w]`).
  KeyError(String),
  /// `IndexError(message)` from list indexing (`phones[-1]`,
  /// `word_split[1]`).
  IndexError(String),
  /// `TimeoutError` from the nltk tokenizer regexes. Holds the Rust message.
  TimeoutError(String),
}

impl Error {
  /// The Python exception class name.
  pub fn name(&self) -> &'static str {
    match self {
      Error::KeyError(_) => "KeyError",
      Error::IndexError(_) => "IndexError",
      Error::TimeoutError(_) => "TimeoutError",
    }
  }

  /// `str(exception.args[0])`.
  pub fn message(&self) -> &str {
    match self {
      Error::KeyError(m) | Error::IndexError(m) | Error::TimeoutError(m) => m,
    }
  }
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Error::KeyError(key) => write!(f, "KeyError: {key:?}"),
      Error::IndexError(m) => write!(f, "IndexError: {m}"),
      Error::TimeoutError(m) => write!(f, "TimeoutError: {m}"),
    }
  }
}

impl std::error::Error for Error {}

const LIST_INDEX_OUT_OF_RANGE: &str = "list index out of range";

// ---------------------------------------------------------------------------
// CPython string helpers (no Python counterpart)

#[derive(Deserialize)]
struct PyStrTables {
  isalpha: Vec<(u32, u32)>,
  upper_or_title: Vec<(u32, u32)>,
  lower: Vec<(u32, u32)>,
}

static PY_STR: LazyLock<PyStrTables> = LazyLock::new(|| {
  serde_json::from_str(include_str!("../data/english/py_str.json"))
    .expect("data/english/py_str.json is valid")
});

fn in_ranges(ranges: &[(u32, u32)], c: char) -> bool {
  let c = c as u32;
  ranges
    .binary_search_by(|&(lo, hi)| {
      if hi < c {
        std::cmp::Ordering::Less
      } else if lo > c {
        std::cmp::Ordering::Greater
      } else {
        std::cmp::Ordering::Equal
      }
    })
    .is_ok()
}

/// CPython `str.isalpha` for one character (Unicode 14.0.0).
fn py_isalpha(c: char) -> bool {
  in_ranges(&PY_STR.isalpha, c)
}

/// CPython `str.istitle` (`unicode_istitle_impl`, Unicode 14.0.0).
fn py_istitle(s: &str) -> bool {
  let t = &*PY_STR;
  let mut cased = false;
  let mut previous_is_cased = false;
  for c in s.chars() {
    if in_ranges(&t.upper_or_title, c) {
      if previous_is_cased {
        return false;
      }
      previous_is_cased = true;
      cased = true;
    } else if in_ranges(&t.lower, c) {
      if !previous_is_cased {
        return false;
      }
      previous_is_cased = true;
      cased = true;
    } else {
      previous_is_cased = false;
    }
  }
  cased
}

/// CPython `str.strip()` with no arguments.
fn py_strip(s: &str) -> &str {
  s.trim_matches(char_isspace)
}

/// CPython `str.split(sep)` with an explicit separator.
fn py_split<'a>(s: &'a str, sep: &str) -> Vec<&'a str> {
  s.split(sep).collect()
}

/// The lines that `f.readline()` returns for a file opened in text mode
/// (universal newlines: `\n`, `\r\n` and `\r` end a line).
fn readlines(text: &str) -> Vec<&str> {
  let bytes = text.as_bytes();
  let mut out = Vec::new();
  let mut start = 0;
  let mut i = 0;
  while i < bytes.len() {
    match bytes[i] {
      b'\n' => {
        out.push(&text[start..=i]);
        i += 1;
        start = i;
      }
      b'\r' => {
        let end = if bytes.get(i + 1) == Some(&b'\n') {
          i + 2
        } else {
          i + 1
        };
        out.push(&text[start..end]);
        i = end;
        start = i;
      }
      _ => i += 1,
    }
  }
  if start < bytes.len() {
    out.push(&text[start..]);
  }
  out
}

/// CPython 3.11 `re.escape`.
fn re_escape(pattern: &str) -> String {
  const SPECIAL: &str = "()[]{}?*+-|^$\\.&~# \t\n\r\x0b\x0c";
  let mut out = String::with_capacity(pattern.len());
  for c in pattern.chars() {
    if SPECIAL.contains(c) {
      out.push('\\');
    }
    out.push(c);
  }
  out
}

// ---------------------------------------------------------------------------
// Module data

/// Port of `rep_map` (适配中文及 g2p_en 标点), in dict order.
///
/// `text_normalize` joins the `re.escape`d keys, so the first two keys match
/// only the literal strings `[;:：，；]` and `["’]`, not the characters in
/// them.
pub const REP_MAP: &[(&str, &str)] = &[
  ("[;:：，；]", ","),
  ("[\"’]", "'"),
  ("。", "."),
  ("！", "!"),
  ("？", "?"),
];

/// Port of `arpa`. The module never reads it. Python keeps it as a `set`;
/// this is the source order.
pub const ARPA: &[&str] = &[
  "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG", "B", "G", "AY0", "M", "AA0", "F",
  "AO0", "ER2", "UH1", "IY1", "AH2", "DH", "IY0", "EY1", "IH0", "K", "N", "W", "IY2", "T", "AA1",
  "ER1", "EH2", "OY0", "UH2", "UW1", "Z", "AW2", "AW1", "V", "UW2", "AA2", "ER", "AW0", "UW0", "R",
  "OW1", "EH1", "ZH", "AE0", "IH2", "IH", "Y", "JH", "P", "AY1", "EY0", "OY2", "TH", "HH", "D",
  "ER0", "CH", "AO1", "AE1", "AO2", "OY1", "AY2", "IH1", "OW0", "L", "SH",
];

static SYMBOLS_SET: LazyLock<HashSet<&'static str>> =
  LazyLock::new(|| symbols2::SYMBOLS.iter().copied().collect());

/// Port of `replace_phs(phs)`: keeps the phones in `symbols2.symbols`, maps
/// `'` to `-` and drops the rest (Python also prints each dropped phone).
pub fn replace_phs<S: AsRef<str>>(phs: &[S]) -> Vec<String> {
  let rep_map: [(&str, &str); 1] = [("'", "-")];
  let mut phs_new = Vec::with_capacity(phs.len());
  for ph in phs {
    let ph = ph.as_ref();
    if SYMBOLS_SET.contains(ph) {
      phs_new.push(ph.to_string());
    } else if let Some((_, v)) = rep_map.iter().find(|(k, _)| *k == ph) {
      phs_new.push(v.to_string());
    }
    // else: print("ph not in symbols: ", ph)
  }
  phs_new
}

static CONSECUTIVE_PUNCTUATION_RE: LazyLock<PyRegex> = LazyLock::new(|| {
  let punctuations: String = symbols::PUNCTUATION.iter().map(|p| re_escape(p)).collect();
  PyRegex::compile(&format!("([{punctuations}\\s])([{punctuations}])+"))
});

/// Port of `replace_consecutive_punctuation(text)`: a punctuation mark or
/// whitespace followed by punctuation marks (`text.symbols.punctuation`)
/// keeps only the first character.
pub fn replace_consecutive_punctuation(text: &str) -> String {
  CONSECUTIVE_PUNCTUATION_RE
    .sub(text, |m| Ok(m.group_str(1).to_string()))
    .expect("the replacement never fails")
}

/// Port of `read_dict()`: the old `cmudict.rep` reader with syllables split on
/// ` - `. Upstream no longer calls it. On the shipped `cmudict.rep` it raises
/// `IndexError` (line 49, `;;; `, has no `"  "` separator), so this returns
/// that error.
pub fn read_dict() -> Result<G2pDict, Error> {
  let mut g2p_dict = G2pDict::new();
  let start_line = 49;
  for (i, line) in readlines(CMU_DICT).into_iter().enumerate() {
    let line_index = i + 1;
    if line_index >= start_line {
      let line = py_strip(line);
      let word_split = py_split(line, "  ");
      let word = lower(word_split[0]);

      let syllables = word_split
        .get(1)
        .ok_or_else(|| Error::IndexError(LIST_INDEX_OUT_OF_RANGE.to_string()))?;
      let syllable_split = py_split(syllables, " - ");
      let entry = g2p_dict.entry(word).or_default();
      entry.clear();
      for syllable in syllable_split {
        let phone_split = py_split(syllable, " ");
        entry.push(phone_split.into_iter().map(str::to_string).collect());
      }
    }
  }
  Ok(g2p_dict)
}

/// Port of `read_dict_new()`: `cmudict.rep` from line 57 (`"  "` between
/// word and phones), then the words of `cmudict-fast.rep` (`" "`
/// separated) that are not in it yet. Keys are lowercased.
pub fn read_dict_new() -> Result<G2pDict, Error> {
  let mut g2p_dict = G2pDict::new();
  for (i, line) in readlines(CMU_DICT).into_iter().enumerate() {
    let line_index = i + 1;
    if line_index >= 57 {
      let line = py_strip(line);
      let word_split = py_split(line, "  ");
      let word = lower(word_split[0]);
      let phones = word_split
        .get(1)
        .ok_or_else(|| Error::IndexError(LIST_INDEX_OUT_OF_RANGE.to_string()))?;
      g2p_dict.insert(
        word,
        vec![
          py_split(phones, " ")
            .into_iter()
            .map(str::to_string)
            .collect(),
        ],
      );
    }
  }

  for line in readlines(CMU_DICT_FAST) {
    let line = py_strip(line);
    let word_split = py_split(line, " ");
    let word = lower(word_split[0]);
    g2p_dict
      .entry(word)
      .or_insert_with(|| vec![word_split[1..].iter().map(|s| s.to_string()).collect()]);
  }

  Ok(g2p_dict)
}

/// Port of `hot_reload_hot(g2p_dict)`: the words of `engdict-hot.rep`
/// overwrite the dictionary entries (自定义发音词直接覆盖字典).
pub fn hot_reload_hot(mut g2p_dict: G2pDict) -> G2pDict {
  for line in readlines(CMU_DICT_HOT) {
    let line = py_strip(line);
    let word_split = py_split(line, " ");
    let word = lower(word_split[0]);
    g2p_dict.insert(
      word,
      vec![word_split[1..].iter().map(|s| s.to_string()).collect()],
    );
  }
  g2p_dict
}

/// Port of `get_dict()`: `engdict_cache.pickle` (which equals
/// `read_dict_new()`) with `hot_reload_hot` applied.
pub fn get_dict() -> Result<G2pDict, Error> {
  Ok(hot_reload_hot(read_dict_new()?))
}

/// Port of `get_namedict()`: the contents of `namedict_cache.pickle`.
pub fn get_namedict() -> G2pDict {
  let mut name_dict = G2pDict::new();
  for line in NAMECACHE.split_terminator('\n') {
    let mut fields = line.split('\t');
    let key = fields.next().expect("split yields at least one field");
    let prons = fields
      .map(|f| {
        if f.is_empty() {
          Vec::new()
        } else {
          f.split(' ').map(str::to_string).collect()
        }
      })
      .collect();
    name_dict.insert(key.to_string(), prons);
  }
  name_dict
}

static REP_MAP_RE: LazyLock<PyRegex> = LazyLock::new(|| {
  let pattern: Vec<String> = REP_MAP.iter().map(|(k, _)| re_escape(k)).collect();
  PyRegex::compile(&pattern.join("|"))
});

/// Port of `text_normalize(text)`: `rep_map` replacement,
/// `en_normalization.expend.normalize`, then
/// `replace_consecutive_punctuation`. Returns the exception that
/// `normalize` raises.
pub fn text_normalize(text: &str) -> Result<String, en_normalization::Error> {
  // 效果相同，和 chinese.py 保持一致
  let text = REP_MAP_RE.sub(text, |m| {
    let g = m.group0();
    Ok(
      REP_MAP
        .iter()
        .find(|(k, _)| *k == g)
        .map(|(_, v)| v.to_string())
        .expect("every match is a rep_map key"),
    )
  })?;

  // text = unicode(text) is a no-op on str.
  let text = normalize(&text)?;

  // 避免重复标点引起的参考泄露
  Ok(replace_consecutive_punctuation(&text))
}

// ---------------------------------------------------------------------------
// en_G2p

fn strings(phones: &[&str]) -> Vec<String> {
  phones.iter().map(|s| s.to_string()).collect()
}

/// Port of `class en_G2p(G2p)`. Derefs to the parent [`G2p`] for
/// `homograph2features` and `predict`.
#[allow(non_camel_case_types)]
pub struct en_G2p {
  /// The parent `G2p` (its `__init__` state).
  pub base: G2p,
  /// `self.cmu`: `get_dict()` without the deleted abbreviations.
  pub cmu: G2pDict,
  /// `self.namedict`: `get_namedict()`.
  pub namedict: G2pDict,
}

impl Deref for en_G2p {
  type Target = G2p;

  fn deref(&self) -> &G2p {
    &self.base
  }
}

impl DerefMut for en_G2p {
  fn deref_mut(&mut self) -> &mut G2p {
    &mut self.base
  }
}

impl Default for en_G2p {
  fn default() -> Self {
    Self::new()
  }
}

impl en_G2p {
  /// Port of `en_G2p.__init__`: the parent `G2p.__init__`,
  /// `wordsegment.load()`, `get_dict()` minus `ae ai ar ios hud os`,
  /// `get_namedict()`, and the `read` / `complex` homograph overrides.
  pub fn new() -> Self {
    let mut base = G2p::new();
    // 分词初始化
    wordsegment::load();

    // 扩展过时字典, 添加姓名字典
    let mut cmu = get_dict().expect("the embedded dictionaries are well formed");
    let namedict = get_namedict();

    // 剔除读音错误的几个缩写
    for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"] {
      let key = lower(word);
      if cmu.remove(&key).is_none() {
        panic!("KeyError: {key:?}");
      }
    }

    // 修正多音字
    base.homograph2features.insert(
      "read".to_string(),
      (
        strings(&["R", "IY1", "D"]),
        strings(&["R", "EH1", "D"]),
        "VBP".to_string(),
      ),
    );
    base.homograph2features.insert(
      "complex".to_string(),
      (
        strings(&["K", "AH0", "M", "P", "L", "EH1", "K", "S"]),
        strings(&["K", "AA1", "M", "P", "L", "EH0", "K", "S"]),
        "JJ".to_string(),
      ),
    );

    Self {
      base,
      cmu,
      namedict,
    }
  }

  fn cmu_first(&self, key: &str) -> Result<Vec<String>, Error> {
    match self.cmu.get(key) {
      Some(prons) => prons
        .first()
        .cloned()
        .ok_or_else(|| Error::IndexError(LIST_INDEX_OUT_OF_RANGE.to_string())),
      None => Err(Error::KeyError(key.to_string())),
    }
  }

  /// Port of `en_G2p.__call__(text)`: `TweetTokenizer().tokenize`, `pos_tag`,
  /// then one pronunciation per token (non-letter tokens as is, single
  /// letters, homographs by POS tag, everything else through `qryword`),
  /// joined by `" "` tokens.
  pub fn __call__(&self, text: &str) -> Result<Vec<String>, Error> {
    // tokenization
    let words = TweetTokenizer::new()
      .tokenize(text)
      .map_err(|e| Error::TimeoutError(e.to_string()))?;
    let tokens = pos_tag(&words); // tuples of (word, tag)

    // steps
    let mut prons: Vec<String> = Vec::new();
    for (o_word, pos) in &tokens {
      // 还原 g2p_en 小写操作逻辑
      let word = lower(o_word);

      let pron: Vec<String> = if !word.chars().any(|c| c.is_ascii_lowercase()) {
        vec![word]
      }
      // 先把单字母推出去
      else if word.chars().count() == 1 {
        // 单读 A 发音修正, 这里需要原格式 o_word 判断大写
        if o_word == "A" {
          strings(&["EY1"])
        } else {
          self.cmu_first(&word)?
        }
      }
      // g2p_en 原版多音字处理
      else if let Some((pron1, pron2, pos1)) = self.homograph2features.get(&word) {
        // Python has two `if` branches that both pick pron1; they are merged
        // with `||` here (same order, same result).
        if pos.starts_with(pos1.as_str())
          // pos1比pos长仅出现在read
          || (pos.chars().count() < pos1.chars().count()
            && pos.chars().eq(pos1.chars().take(pos.chars().count())))
        {
          pron1.clone()
        } else {
          pron2.clone()
        }
      } else {
        // 递归查找预测
        self.qryword(o_word)?
      };

      prons.extend(pron);
      prons.push(" ".to_string());
    }

    prons.pop();
    Ok(prons)
  }

  /// Port of `en_G2p.qryword(o_word)`: CMU dict (except single letters),
  /// the name dict for title-case words, letter-by-letter reading for words
  /// of at most 3 characters, possessive `'s`, `wordsegment` compounds, and
  /// `predict` for the rest.
  pub fn qryword(&self, o_word: &str) -> Result<Vec<String>, Error> {
    let word = lower(o_word);
    let len = word.chars().count();

    // 查字典, 单字母除外
    if len > 1
      && let Some(prons) = self.cmu.get(&word)
    {
      // lookup CMU dict
      return prons
        .first()
        .cloned()
        .ok_or_else(|| Error::IndexError(LIST_INDEX_OUT_OF_RANGE.to_string()));
    }

    // 单词仅首字母大写时查找姓名字典
    if py_istitle(o_word)
      && let Some(prons) = self.namedict.get(&word)
    {
      return prons
        .first()
        .cloned()
        .ok_or_else(|| Error::IndexError(LIST_INDEX_OUT_OF_RANGE.to_string()));
    }

    // oov 长度小于等于 3 直接读字母
    if len <= 3 {
      let mut phones = Vec::new();
      for w in word.chars() {
        // 单读 A 发音修正, 此处不存在大写的情况
        if w == 'a' {
          phones.push("EY1".to_string());
        } else if !py_isalpha(w) {
          phones.push(w.to_string());
        } else {
          phones.extend(self.cmu_first(w.encode_utf8(&mut [0; 4]))?);
        }
      }
      return Ok(phones);
    }

    // 尝试分离所有格
    if is_possessive(&word) {
      let stem: String = word.chars().take(len - 2).collect();
      let mut phones = self.qryword(&stem)?;
      let last = phones
        .last()
        .ok_or_else(|| Error::IndexError(LIST_INDEX_OUT_OF_RANGE.to_string()))?;
      // P T K F TH HH 无声辅音结尾 's 发 ['S']
      if ["P", "T", "K", "F", "TH", "HH"].contains(&last.as_str()) {
        phones.push("S".to_string());
      }
      // S Z SH ZH CH JH 擦声结尾 's 发 ['IH1', 'Z'] 或 ['AH0', 'Z']
      else if ["S", "Z", "SH", "ZH", "CH", "JH"].contains(&last.as_str()) {
        phones.extend(strings(&["AH0", "Z"]));
      }
      // 有声辅音和元音结尾 's 发 ['Z']
      else {
        phones.push("Z".to_string());
      }
      return Ok(phones);
    }

    // 尝试进行分词，应对复合词
    let comps = wordsegment::segment(&lower(&word));

    // 无法分词的送回去预测
    if comps.len() == 1 {
      return Ok(self.predict(&word));
    }

    // 可以分词的递归处理
    let mut phones = Vec::new();
    for comp in &comps {
      phones.extend(self.qryword(comp)?);
    }
    Ok(phones)
  }
}

/// `re.match(r"^([a-z]+)('s)$", word)`. `$` also matches before a trailing
/// newline.
fn is_possessive(word: &str) -> bool {
  let w = word.strip_suffix('\n').unwrap_or(word);
  match w.strip_suffix("'s") {
    Some(stem) => !stem.is_empty() && stem.bytes().all(|b| b.is_ascii_lowercase()),
    None => false,
  }
}

static _G2P: LazyLock<en_G2p> = LazyLock::new(en_G2p::new);

/// Port of the module-level `_g2p = en_G2p()`.
pub fn _g2p() -> &'static en_G2p {
  &_G2P
}

/// Port of `g2p(text)`: `_g2p(text)` without the `" "`, `<pad>`, `UW`,
/// `</s>` and `<s>` phones, `<unk>` as `UNK`, then `replace_phs`.
pub fn g2p(text: &str) -> Result<Vec<String>, Error> {
  // g2p_en 整段推理，剔除不存在的arpa返回
  let phone_list = _g2p().__call__(text)?;
  let phones: Vec<&str> = phone_list
    .iter()
    .map(String::as_str)
    .filter(|ph| !matches!(*ph, " " | "<pad>" | "UW" | "</s>" | "<s>"))
    .map(|ph| if ph != "<unk>" { ph } else { "UNK" })
    .collect();

  Ok(replace_phs(&phones))
}
