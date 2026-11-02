//! Subset port of the Python `jieba_fast` 0.53 package used by GPT-SoVITS.
//!
//! Ported: `jieba_fast/__init__.py` (`Tokenizer` with the default dictionary,
//! accurate mode `cut` with and without HMM, `cut_for_search`), the C
//! extension `_jieba_fast_functions_py3` (`_get_DAG_and_calc`, `_viterbi`),
//! `jieba_fast/finalseg` and `jieba_fast/posseg`. See the crate README for
//! the left-out parts.

#![allow(non_snake_case)]

use std::sync::LazyLock;

mod fxhash;

pub mod _jieba_fast_functions;
pub mod finalseg;
pub mod posseg;

pub use fxhash::FxHashMap;

/// Port of `jieba_fast.DEFAULT_DICT_NAME` (`dict.txt`): the file contents,
/// copied unmodified from the 0.53 package.
pub static DEFAULT_DICT: &str = include_str!("../data/dict.txt");

/// Port of `re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)`
/// as a per-character predicate.
pub fn re_han_default(c: char) -> bool {
  matches!(c, '\u{4E00}'..='\u{9FD5}' | 'a'..='z' | 'A'..='Z' | '0'..='9' | '+' | '#' | '&' | '.' | '_' | '%')
}

/// Python `\s` for `str` patterns (`Py_UNICODE_ISSPACE`, the same set as
/// `str.isspace()` in Python 3.11). It is not Rust's `char::is_whitespace`:
/// Python includes U+001C..U+001F and excludes nothing that Rust has here.
pub fn is_python_space(c: char) -> bool {
  matches!(
    c,
    '\u{9}'..='\u{d}'
      | '\u{1c}'..='\u{20}'
      | '\u{85}'
      | '\u{a0}'
      | '\u{1680}'
      | '\u{2000}'..='\u{200a}'
      | '\u{2028}'
      | '\u{2029}'
      | '\u{202f}'
      | '\u{205f}'
      | '\u{3000}'
  )
}

/// `re_eng = re.compile('[a-zA-Z0-9]', re.U)` as a character predicate.
pub fn re_eng(c: char) -> bool {
  c.is_ascii_alphanumeric()
}

/// Byte offsets of every character of `s`, plus `s.len()` at the end, so
/// `&s[offs[a]..offs[b]]` is the Python slice `s[a:b]`.
pub(crate) fn char_offsets(s: &str) -> Vec<usize> {
  let mut offs: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
  offs.push(s.len());
  offs
}

/// `re.split("(X+)", s)` for a character class `X`, without the empty strings:
/// the maximal runs of `s` with the flag telling whether the run matched.
pub(crate) fn split_runs(s: &str, class: impl Fn(char) -> bool) -> Vec<(&str, bool)> {
  let mut out = Vec::new();
  let mut start = 0;
  let mut cur: Option<bool> = None;
  for (i, c) in s.char_indices() {
    let m = class(c);
    match cur {
      Some(prev) if prev == m => {}
      Some(prev) => {
        out.push((&s[start..i], prev));
        start = i;
        cur = Some(m);
      }
      None => cur = Some(m),
    }
  }
  if let Some(prev) = cur {
    out.push((&s[start..], prev));
  }
  out
}

/// `re_skip_default.split(blk)` (`(\r\n|\s)`) followed by the
/// `for xx in x: yield xx` loop of `Tokenizer.cut` for accurate mode: every
/// character becomes a token, except that `"\r\n"` stays one token.
pub(crate) fn skip_split_chars(blk: &str) -> Vec<&str> {
  let mut out = Vec::new();
  let mut it = blk.char_indices().peekable();
  while let Some((i, c)) = it.next() {
    if c == '\r' && matches!(it.peek(), Some((_, '\n'))) {
      it.next();
      out.push(&blk[i..i + 2]);
    } else {
      out.push(&blk[i..i + c.len_utf8()]);
    }
  }
  out
}

/// Port of `jieba_fast.Tokenizer`.
pub struct Tokenizer {
  /// `Tokenizer.FREQ`: word and prefix frequencies (prefixes that are not
  /// words have frequency 0).
  pub FREQ: FxHashMap<Box<str>, i64>,
  /// `Tokenizer.total`: sum of the frequencies of all dictionary lines.
  pub total: i64,
}

impl Default for Tokenizer {
  fn default() -> Self {
    Self::new()
  }
}

impl Tokenizer {
  /// Port of `Tokenizer()` followed by `initialize()` with the default
  /// dictionary.
  pub fn new() -> Self {
    Self::from_dict(DEFAULT_DICT)
  }

  /// Port of `Tokenizer(dictionary)` + `initialize()`, taking the dictionary
  /// file contents instead of a path.
  pub fn from_dict(dict: &str) -> Self {
    let (FREQ, total) = Self::gen_pfdict(dict);
    Self { FREQ, total }
  }

  /// Port of `Tokenizer.gen_pfdict`. Panics with the Python `ValueError`
  /// message for an invalid line.
  pub fn gen_pfdict(f: &str) -> (FxHashMap<Box<str>, i64>, i64) {
    let mut lfreq: FxHashMap<Box<str>, i64> = FxHashMap::default();
    lfreq.reserve(600_000);
    let mut ltotal: i64 = 0;
    for (idx, raw) in python_lines(f).enumerate() {
      let lineno = idx + 1;
      let line =
        raw.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'));
      let mut parts = line.split(' ');
      let (Some(word), Some(freq)) = (parts.next(), parts.next()) else {
        panic!("ValueError: invalid dictionary entry in <dict> at Line {lineno}: {line}");
      };
      let Ok(freq) = freq.parse::<i64>() else {
        panic!("ValueError: invalid dictionary entry in <dict> at Line {lineno}: {line}");
      };
      lfreq.insert(word.into(), freq);
      ltotal += freq;
      for (i, c) in word.char_indices() {
        let wfrag = &word[..i + c.len_utf8()];
        if !lfreq.contains_key(wfrag) {
          lfreq.insert(wfrag.into(), 0);
        }
      }
    }
    (lfreq, ltotal)
  }

  /// `self.FREQ.get(word)` as a Python truth value: present and non-zero.
  pub(crate) fn freq_truthy(&self, word: &str) -> bool {
    self.FREQ.get(word).is_some_and(|&f| f != 0)
  }

  /// Port of `Tokenizer.get_DAG` (the pure Python version, used by
  /// `posseg`). `DAG[k]` lists the end indices (inclusive, in characters) of
  /// the words that start at `k`.
  pub fn get_DAG(&self, sentence: &str) -> Vec<Vec<usize>> {
    let offs = char_offsets(sentence);
    let n = offs.len() - 1;
    let mut dag = Vec::with_capacity(n);
    for k in 0..n {
      let mut tmplist = Vec::new();
      let mut i = k;
      while i < n {
        match self.FREQ.get(&sentence[offs[k]..offs[i + 1]]) {
          Some(&f) => {
            if f != 0 {
              tmplist.push(i);
            }
          }
          None => break,
        }
        i += 1;
      }
      if tmplist.is_empty() {
        tmplist.push(k);
      }
      dag.push(tmplist);
    }
    dag
  }

  /// Port of `Tokenizer.calc` (the pure Python version, used by `posseg`):
  /// `route[idx] = max((log(FREQ.get(frag) or 1) - logtotal + route[x + 1][0], x) for x in DAG[idx])`.
  /// Ties pick the larger `x`, like Python's tuple `max`.
  pub fn calc(&self, sentence: &str, DAG: &[Vec<usize>]) -> Vec<(f64, usize)> {
    let offs = char_offsets(sentence);
    let n = offs.len() - 1;
    let mut route = vec![(0.0f64, 0usize); n + 1];
    let logtotal = (self.total as f64).ln();
    for idx in (0..n).rev() {
      let mut best: Option<(f64, usize)> = None;
      for &x in &DAG[idx] {
        let f = match self.FREQ.get(&sentence[offs[idx]..offs[x + 1]]) {
          Some(&f) if f != 0 => f,
          _ => 1,
        };
        let cand = ((f as f64).ln() - logtotal + route[x + 1].0, x);
        best = match best {
          None => Some(cand),
          Some(b) if cand.0 > b.0 || (cand.0 == b.0 && cand.1 > b.1) => Some(cand),
          Some(b) => Some(b),
        };
      }
      route[idx] = best.expect("ValueError: max() arg is an empty sequence");
    }
    route
  }

  /// Flushes `buf` the way the end of `Tokenizer.__cut_DAG` does.
  fn flush_hmm_buf(&self, buf: &str, out: &mut Vec<String>) {
    if buf.chars().nth(1).is_none() {
      out.push(buf.to_string());
    } else if !self.freq_truthy(buf) {
      finalseg::cut_into(buf, out);
    } else {
      for (i, c) in buf.char_indices() {
        out.push(buf[i..i + c.len_utf8()].to_string());
      }
    }
  }

  /// Port of `Tokenizer.__cut_DAG` (HMM=True), which uses the C
  /// `_get_DAG_and_calc` route.
  fn cut_DAG(&self, sentence: &str, out: &mut Vec<String>) {
    let offs = char_offsets(sentence);
    let n = offs.len() - 1;
    let route = _jieba_fast_functions::_get_DAG_and_calc(&self.FREQ, sentence, self.total as f64);
    let mut x = 0;
    let mut buf_start: Option<usize> = None;
    while x < n {
      let y = route[x] + 1;
      if y - x == 1 {
        buf_start.get_or_insert(x);
      } else {
        if let Some(bs) = buf_start.take() {
          self.flush_hmm_buf(&sentence[offs[bs]..offs[x]], out);
        }
        out.push(sentence[offs[x]..offs[y]].to_string());
      }
      x = y;
    }
    if let Some(bs) = buf_start {
      self.flush_hmm_buf(&sentence[offs[bs]..], out);
    }
  }

  /// Port of `Tokenizer.__cut_DAG_NO_HMM`, which uses the C
  /// `_get_DAG_and_calc` route.
  fn cut_DAG_NO_HMM(&self, sentence: &str, out: &mut Vec<String>) {
    let offs = char_offsets(sentence);
    let n = offs.len() - 1;
    let route = _jieba_fast_functions::_get_DAG_and_calc(&self.FREQ, sentence, self.total as f64);
    let mut x = 0;
    let mut buf = String::new();
    while x < n {
      let y = route[x] + 1;
      let l_word = &sentence[offs[x]..offs[y]];
      let mut cs = l_word.chars();
      let single_eng = matches!((cs.next(), cs.next()), (Some(c), None) if re_eng(c));
      if single_eng {
        buf.push_str(l_word);
      } else {
        if !buf.is_empty() {
          out.push(std::mem::take(&mut buf));
        }
        out.push(l_word.to_string());
      }
      x = y;
    }
    if !buf.is_empty() {
      out.push(buf);
    }
  }

  /// Port of `Tokenizer.cut(sentence, cut_all=False, HMM=hmm)`.
  /// `cut_all=True` is not ported.
  pub fn cut(&self, sentence: &str, hmm: bool) -> Vec<String> {
    let mut out = Vec::new();
    for (blk, is_han) in split_runs(sentence, re_han_default) {
      if is_han {
        if hmm {
          self.cut_DAG(blk, &mut out);
        } else {
          self.cut_DAG_NO_HMM(blk, &mut out);
        }
      } else {
        out.extend(skip_split_chars(blk).into_iter().map(String::from));
      }
    }
    out
  }

  /// Port of `Tokenizer.lcut(sentence, cut_all=False, HMM=hmm)`.
  pub fn lcut(&self, sentence: &str, hmm: bool) -> Vec<String> {
    self.cut(sentence, hmm)
  }

  /// Port of `Tokenizer.cut_for_search(sentence, HMM=hmm)`.
  pub fn cut_for_search(&self, sentence: &str, hmm: bool) -> Vec<String> {
    let mut out = Vec::new();
    for w in self.cut(sentence, hmm) {
      let offs = char_offsets(&w);
      let n = offs.len() - 1;
      if n > 2 {
        for i in 0..n - 1 {
          let gram2 = &w[offs[i]..offs[i + 2]];
          if self.freq_truthy(gram2) {
            out.push(gram2.to_string());
          }
        }
      }
      if n > 3 {
        for i in 0..n - 2 {
          let gram3 = &w[offs[i]..offs[i + 3]];
          if self.freq_truthy(gram3) {
            out.push(gram3.to_string());
          }
        }
      }
      out.push(w);
    }
    out
  }

  /// Port of `Tokenizer.lcut_for_search(sentence, HMM=hmm)`.
  pub fn lcut_for_search(&self, sentence: &str, hmm: bool) -> Vec<String> {
    self.cut_for_search(sentence, hmm)
  }
}

/// Lines of a file read in binary mode by `for line in f`: split after each
/// `\n`, with a final line without `\n` when the file does not end with one.
pub(crate) fn python_lines(f: &str) -> impl Iterator<Item = &str> {
  f.split_inclusive('\n')
}

/// Port of the default tokenizer `jieba_fast.dt`.
#[allow(non_upper_case_globals)]
pub static dt: LazyLock<Tokenizer> = LazyLock::new(Tokenizer::new);

/// Port of `jieba_fast.get_FREQ(k, d=None)`.
pub fn get_FREQ(k: &str) -> Option<i64> {
  dt.FREQ.get(k).copied()
}

/// Port of `jieba_fast.cut(sentence, cut_all=False, HMM=hmm)`.
pub fn cut(sentence: &str, hmm: bool) -> Vec<String> {
  dt.cut(sentence, hmm)
}

/// Port of `jieba_fast.lcut(sentence, cut_all=False, HMM=True)`.
pub fn lcut(sentence: &str, hmm: bool) -> Vec<String> {
  dt.lcut(sentence, hmm)
}

/// Port of `jieba_fast.cut_for_search(sentence, HMM=True)`.
pub fn cut_for_search(sentence: &str, hmm: bool) -> Vec<String> {
  dt.cut_for_search(sentence, hmm)
}

/// Port of `jieba_fast.lcut_for_search(sentence, HMM=True)`.
pub fn lcut_for_search(sentence: &str, hmm: bool) -> Vec<String> {
  dt.lcut_for_search(sentence, hmm)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn split_runs_matches_re_split() {
    assert_eq!(
      split_runs("ab，c d", |c| c.is_ascii_alphabetic()),
      vec![
        ("ab", true),
        ("，", false),
        ("c", true),
        (" ", false),
        ("d", true)
      ]
    );
    assert!(split_runs("", |_| true).is_empty());
  }

  #[test]
  fn skip_split_keeps_crlf() {
    assert_eq!(
      skip_split_chars("\r\r\n，\n"),
      vec!["\r", "\r\n", "，", "\n"]
    );
  }
}
