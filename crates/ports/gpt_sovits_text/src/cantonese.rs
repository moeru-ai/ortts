//! Port of `GPT_SoVITS/text/cantonese.py` (GPT-SoVITS @ 04a44eb).
//!
//! # Ported scope
//!
//! `INITIALS` (with the `sp`/`spl`/`spn`/`sil` additions), `rep_map`,
//! `replace_punctuation`, `text_normalize`, `punctuation_set`,
//! `jyuping_to_initials_finals_tones`, `get_jyutping` and `g2p`. Left out:
//! `normalizer` (the `cn2an.transform(x, "an2cn")` lambda, which nothing in
//! the module calls), `get_bert_feature` (BERT, not part of the text
//! frontend) and the `__main__` demo.
//!
//! # Differences in form (not in output)
//!
//! Where Python raises, the Rust functions return an [`Error`] that names the
//! Python exception: `text_normalize` passes on the `KeyError` / `ValueError`
//! of `TextNormalizer.normalize`; `get_jyutping` (and so `g2p`) gives
//! `TypeError` for a character that ToJyutping has no reading for (Python
//! calls `re.search` on `None`) and `ValueError` for a reading that does not
//! match the jyutping pattern; `jyuping_to_initials_finals_tones` gives
//! `IndexError` for an empty syllable (`syllable[-1]`).
//!
//! Python string indexing (`syllable[-1]`, `[:2]`, ...) counts code points, so
//! the port slices by `char`. `int(syllable[-1])` accepts every Unicode decimal
//! digit (`str.isdecimal`), as Python does.
//!
//! The `ValueError` branch of `get_jyutping` is ported but cannot be reached:
//! every reading in the embedded ToJyutping 3.2.0 trie matches
//! `^([a-z]+[1-6]+[ ]?)+$` (the fixture generator checks all of them).
//!
//! # Parity
//!
//! `tests/cantonese.rs` checks the fixtures from
//! `parity/cantonese/gen_fixtures.py` for exact equality,
//! including exception types and messages:
//!
//! | Fixture | Cases | Exact matches |
//! |---|---|---|
//! | `sentences.jsonl`: `replace_punctuation`, `text_normalize`, `get_jyutping(norm)`, `g2p(norm)`, `g2p(raw text)` | 3357 x 5 | 3357 x 5 |
//! | `syllables.jsonl`: `jyuping_to_initials_finals_tones` | 2190 | 2190 |
//!
//! There are no known mismatches, so there is no exception list.

use std::sync::LazyLock;

use regex::Regex;

use crate::symbols::PUNCTUATION;
use crate::tts_infer_pack::text_segmentation_method::py_isdecimal;
use crate::zh_normalization::{self, TextNormalizer};

/// A Python exception raised by `cantonese.py` or the code it calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
  /// `KeyError(key)` from `TextNormalizer.normalize`.
  KeyError(String),
  /// `ValueError(message)`, from `TextNormalizer.normalize` or from
  /// `get_jyutping` (`Failed to convert {word} to jyutping: {syllable}`).
  ValueError(String),
  /// `TypeError` from `re.search` on a `None` reading in `get_jyutping`.
  TypeError(String),
  /// `IndexError` from `syllable[-1]` on an empty syllable in
  /// `jyuping_to_initials_finals_tones`.
  IndexError(String),
}

impl Error {
  /// The Python exception class name.
  pub fn name(&self) -> &'static str {
    match self {
      Error::KeyError(_) => "KeyError",
      Error::ValueError(_) => "ValueError",
      Error::TypeError(_) => "TypeError",
      Error::IndexError(_) => "IndexError",
    }
  }
}

impl std::fmt::Display for Error {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Error::KeyError(key) => write!(f, "KeyError: {key:?}"),
      Error::ValueError(msg) => write!(f, "ValueError: {msg}"),
      Error::TypeError(msg) => write!(f, "TypeError: {msg}"),
      Error::IndexError(msg) => write!(f, "IndexError: {msg}"),
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

/// Port of `INITIALS` (after `INITIALS += ["sp", "spl", "spn", "sil"]`), in
/// list order. The order matters: the first prefix that matches wins.
pub const INITIALS: &[&str] = &[
  "aa", "aai", "aak", "aap", "aat", "aau", "ai", "au", "ap", "at", "ak", "a", "p", "b", "e", "ts",
  "t", "dz", "d", "kw", "k", "gw", "g", "f", "h", "l", "m", "ng", "n", "s", "y", "w", "c", "z",
  "j", "ong", "on", "ou", "oi", "ok", "o", "uk", "ung", "sp", "spl", "spn", "sil",
];

/// Port of `rep_map`, in dict order.
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
  ("“", "'"),
  ("”", "'"),
  ("\"", "'"),
  ("‘", "'"),
  ("’", "'"),
  ("（", "'"),
  ("）", "'"),
  ("(", "'"),
  (")", "'"),
  ("《", "'"),
  ("》", "'"),
  ("【", "'"),
  ("】", "'"),
  ("[", "'"),
  ("]", "'"),
  ("—", "-"),
  ("～", "-"),
  ("~", "-"),
  ("「", "'"),
  ("」", "'"),
];

/// `c in punctuation` for `text.symbols.punctuation` (a list of
/// one-character strings), which is also the `[...]` class that
/// `replace_punctuation` and `get_jyutping` build from it.
fn is_punctuation_char(c: char) -> bool {
  let mut buf = [0u8; 4];
  let s: &str = c.encode_utf8(&mut buf);
  PUNCTUATION.contains(&s)
}

/// Port of `replace_punctuation`.
///
/// First `re.sub` with the alternation of the escaped `rep_map` keys (tried in
/// dict order at each position), then removal of every run of characters
/// outside `[\u4e00-\u9fa5!?…,.-]`.
pub fn replace_punctuation(text: &str) -> String {
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
    .chars()
    .filter(|&c| ('\u{4e00}'..='\u{9fa5}').contains(&c) || is_punctuation_char(c))
    .collect()
}

/// Port of `text_normalize`: `TextNormalizer().normalize(text)`, then
/// [`replace_punctuation`] on each sentence, concatenated.
pub fn text_normalize(text: &str) -> Result<String, Error> {
  let tx = TextNormalizer::new();
  let sentences = tx.normalize(text)?;
  let mut dest_text = String::new();
  for sentence in &sentences {
    dest_text.push_str(&replace_punctuation(sentence));
  }
  Ok(dest_text)
}

/// Port of `punctuation_set` (`set(punctuation)`): membership of a whole
/// string.
fn in_punctuation_set(s: &str) -> bool {
  PUNCTUATION.contains(&s)
}

/// Python `int(c)` for a one-character string: the value of a Unicode decimal
/// digit, `None` (`ValueError`) for anything else. Decimal digits come in
/// runs of ten starting at zero, so the value is the offset in the run.
fn py_int_char(c: char) -> Option<u32> {
  if !py_isdecimal(c) {
    return None;
  }
  let mut start = c as u32;
  while start > 0 {
    match char::from_u32(start - 1) {
      Some(p) if py_isdecimal(p) => start -= 1,
      _ => break,
    }
  }
  Some((c as u32 - start) % 10)
}

/// Python `s[a:b]` by code points, for non-negative `a` and `b`.
fn char_slice(s: &str, a: usize, b: Option<usize>) -> String {
  let it = s.chars().skip(a);
  match b {
    Some(b) => it.take(b.saturating_sub(a)).collect(),
    None => it.collect(),
  }
}

/// Port of `jyuping_to_initials_finals_tones(jyuping_syllables)`.
///
/// Returns `(phones, word2ph)`. Quirks kept from Python: a syllable that
/// matches no entry of [`INITIALS`] adds nothing (not even to `word2ph`); a
/// multi-syllable reading such as `"jing1 cyun3"` is treated as one syllable
/// (tone from the last character, the rest after the initial as the final).
pub fn jyuping_to_initials_finals_tones<S: AsRef<str>>(
  jyuping_syllables: &[S],
) -> Result<(Vec<String>, Vec<usize>), Error> {
  let mut initials_finals: Vec<String> = Vec::new();
  let mut tones: Vec<i64> = Vec::new();
  let mut word2ph: Vec<usize> = Vec::new();

  for syllable in jyuping_syllables {
    let syllable = syllable.as_ref();
    // Python has two branches with the same body: `syllable in punctuation`
    // and `syllable == "_"`.
    if in_punctuation_set(syllable) || syllable == "_" {
      initials_finals.push(syllable.to_string());
      tones.push(0);
      word2ph.push(1);
    } else {
      let Some(last) = syllable.chars().last() else {
        return Err(Error::IndexError("string index out of range".to_string()));
      };
      let (tone, syllable_without_tone) = match py_int_char(last) {
        Some(t) => (t as i64, &syllable[..syllable.len() - last.len_utf8()]),
        None => (0, syllable),
      };

      for initial in INITIALS {
        if let Some(rest) = syllable_without_tone.strip_prefix(initial) {
          if syllable_without_tone.starts_with("nga") {
            let rest = char_slice(syllable_without_tone, 2, None);
            let final_ = if rest.is_empty() {
              syllable_without_tone.chars().last().unwrap().to_string()
            } else {
              rest
            };
            initials_finals.push(char_slice(syllable_without_tone, 0, Some(2)));
            initials_finals.push(final_);
            tones.extend([-1, tone]);
            word2ph.push(2);
          } else {
            let final_ = if rest.is_empty() {
              initial.chars().last().unwrap().to_string()
            } else {
              rest.to_string()
            };
            initials_finals.push(initial.to_string());
            initials_finals.push(final_);
            tones.extend([-1, tone]);
            word2ph.push(2);
          }
          break;
        }
      }
    }
  }
  assert_eq!(initials_finals.len(), tones.len());

  let mut phones = Vec::with_capacity(initials_finals.len());
  for (a, b) in initials_finals.iter().zip(&tones) {
    let mut todo = if *b != -1 && *b != 0 {
      format!("{a}{b}")
    } else {
      a.clone()
    };
    if !in_punctuation_set(&todo) {
      todo = format!("Y{todo}");
    }
    phones.push(todo);
  }

  Ok((phones, word2ph))
}

/// `re.search(r"^([a-z]+[1-6]+[ ]?)+$", syllable)`. Python `$` also matches
/// before a final newline.
static JYUTPING_RE: LazyLock<Regex> =
  LazyLock::new(|| Regex::new(r"^(?:[a-z]+[1-6]+ ?)+\n?\z").unwrap());

/// `punct_pattern.match(word)`: `^[{punctuation}]+$`, where `$` also matches
/// before a final newline.
fn punct_pattern_match(word: &str) -> bool {
  let body = word.strip_suffix('\n').unwrap_or(word);
  !body.is_empty() && body.chars().all(is_punctuation_char)
}

/// Port of `get_jyutping(text)`.
pub fn get_jyutping(text: &str) -> Result<Vec<String>, Error> {
  let mut jyutping_array = Vec::new();

  let syllables = tojyutping::get_jyutping_list(text);

  for (word, syllable) in syllables {
    if punct_pattern_match(&word) {
      // re.split(r"([{punctuation}])", word), keeping the non-empty parts.
      let mut piece = String::new();
      for c in word.chars() {
        if is_punctuation_char(c) {
          if !piece.is_empty() {
            jyutping_array.push(std::mem::take(&mut piece));
          }
          jyutping_array.push(c.to_string());
        } else {
          piece.push(c);
        }
      }
      if !piece.is_empty() {
        jyutping_array.push(piece);
      }
    } else {
      let Some(syllable) = syllable else {
        return Err(Error::TypeError(
          "expected string or bytes-like object, got 'NoneType'".to_string(),
        ));
      };
      if !JYUTPING_RE.is_match(&syllable) {
        return Err(Error::ValueError(format!(
          "Failed to convert {word} to jyutping: {syllable}"
        )));
      }
      jyutping_array.push(syllable);
    }
  }

  Ok(jyutping_array)
}

/// Port of `g2p(text)`: [`get_jyutping`], then
/// [`jyuping_to_initials_finals_tones`]. Returns `(phones, word2ph)`.
pub fn g2p(text: &str) -> Result<(Vec<String>, Vec<usize>), Error> {
  let jyuping = get_jyutping(text)?;
  jyuping_to_initials_finals_tones(&jyuping)
}
