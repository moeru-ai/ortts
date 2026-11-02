//! Port of `jieba_fast.finalseg`: the BMES HMM that splits runs of single
//! characters into new words.

use std::sync::LazyLock;

use unicode_general_category::{GeneralCategory, get_general_category};

use crate::_jieba_fast_functions::{_viterbi, STATES, ViterbiModel};
use crate::fxhash::FxHashMap;
use crate::split_runs;

/// `finalseg/prob_start.py`, converted by `parity/export_data.py`.
pub static PROB_START_P: &str = include_str!("../data/finalseg/prob_start.txt");
/// `finalseg/prob_trans.py`, converted by `parity/export_data.py`.
pub static PROB_TRANS_P: &str = include_str!("../data/finalseg/prob_trans.txt");
/// `finalseg/prob_emit.py`, converted by `parity/export_data.py`.
pub static PROB_EMIT_P: &str = include_str!("../data/finalseg/prob_emit.txt");

/// Port of `MIN_FLOAT`.
pub const MIN_FLOAT: f64 = -3.14e100;

fn state_pos(s: &str) -> usize {
  STATES
    .iter()
    .position(|c| s.len() == 1 && s.starts_with(*c))
    .unwrap_or_else(|| panic!("unknown finalseg state {s:?}"))
}

fn parse_f64(s: &str) -> f64 {
  s.parse()
    .unwrap_or_else(|_| panic!("invalid float {s:?} in finalseg data"))
}

/// Port of `load_model()` / the `start_P`, `trans_P`, `emit_P` imports,
/// reading the converted tables.
pub fn load_model() -> ViterbiModel {
  let mut start_p = [f64::NAN; 4];
  for line in PROB_START_P.lines() {
    let (s, p) = line.split_once('\t').unwrap();
    start_p[state_pos(s)] = parse_f64(p);
  }
  let mut trans_p = [[None; 4]; 4];
  for line in PROB_TRANS_P.lines() {
    let mut fields = line.split('\t');
    let from = state_pos(fields.next().unwrap());
    for f in fields {
      let (to, p) = f.split_once(':').unwrap();
      trans_p[from][state_pos(to)] = Some(parse_f64(p));
    }
  }
  let mut emit_p: [FxHashMap<char, f64>; 4] = Default::default();
  let mut cur = None;
  for line in PROB_EMIT_P.lines() {
    if let Some(s) = line.strip_prefix('@') {
      cur = Some(state_pos(s));
      continue;
    }
    let (c, p) = line.split_once('\t').unwrap();
    let mut cs = c.chars();
    let ch = cs.next().unwrap();
    assert!(cs.next().is_none());
    emit_p[cur.unwrap()].insert(ch, parse_f64(p));
  }
  assert!(start_p.iter().all(|p| !p.is_nan()));
  ViterbiModel {
    start_p,
    trans_p,
    emit_p,
  }
}

/// The finalseg model (`start_P`, `trans_P`, `emit_P`).
pub static MODEL: LazyLock<ViterbiModel> = LazyLock::new(load_model);

/// Port of `viterbi(obs, states, start_p, trans_p, emit_p)`: `finalseg.__cut`
/// calls the C `_viterbi` instead, which this delegates to.
pub fn viterbi(obs: &[char]) -> (f64, Vec<char>) {
  _viterbi(obs, &MODEL)
}

/// Port of `finalseg.__cut(sentence)` for a run of Han characters.
fn cut_han(sentence: &str, out: &mut Vec<String>) {
  let chars: Vec<char> = sentence.chars().collect();
  let (_, pos_list) = viterbi(&chars);
  let offs = crate::char_offsets(sentence);
  let (mut begin, mut nexti) = (0, 0);
  for (i, &c) in chars.iter().enumerate() {
    match pos_list[i] {
      'B' => begin = i,
      'E' => {
        out.push(sentence[offs[begin]..offs[i + 1]].to_string());
        nexti = i + 1;
      }
      'S' => {
        out.push(c.to_string());
        nexti = i + 1;
      }
      _ => {}
    }
  }
  if nexti < chars.len() {
    out.push(sentence[offs[nexti]..].to_string());
  }
}

/// `re_han = re.compile("([\u4E00-\u9FD5]+)")` as a character predicate.
pub fn re_han(c: char) -> bool {
  ('\u{4E00}'..='\u{9FD5}').contains(&c)
}

/// Python `\d` for `str` patterns: Unicode decimal digits (category Nd).
fn is_python_digit(c: char) -> bool {
  c.is_ascii_digit() || (!c.is_ascii() && get_general_category(c) == GeneralCategory::DecimalNumber)
}

/// `re_skip = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")`: the length in bytes
/// of a match at the start of `s`, if any.
fn re_skip_match(s: &str) -> Option<usize> {
  let b = s.as_bytes();
  let mut i = 0;
  while i < b.len() && b[i].is_ascii_alphanumeric() {
    i += 1;
  }
  if i == 0 {
    return None;
  }
  if i < b.len() && b[i] == b'.' {
    let rest = &s[i + 1..];
    let digits: usize = rest
      .chars()
      .take_while(|&c| is_python_digit(c))
      .map(char::len_utf8)
      .sum();
    if digits > 0 {
      i += 1 + digits;
    }
  }
  if i < b.len() && b[i] == b'%' {
    i += 1;
  }
  Some(i)
}

/// `re_skip.split(blk)` with the empty strings dropped.
fn re_skip_split(blk: &str) -> Vec<&str> {
  let mut out = Vec::new();
  let mut start = 0;
  let mut i = 0;
  while i < blk.len() {
    if let Some(len) = re_skip_match(&blk[i..]) {
      if start < i {
        out.push(&blk[start..i]);
      }
      out.push(&blk[i..i + len]);
      i += len;
      start = i;
    } else {
      i += blk[i..].chars().next().unwrap().len_utf8();
    }
  }
  if start < blk.len() {
    out.push(&blk[start..]);
  }
  out
}

/// Port of `finalseg.cut(sentence)`, appending to `out`.
pub(crate) fn cut_into(sentence: &str, out: &mut Vec<String>) {
  for (blk, is_han) in split_runs(sentence, re_han) {
    if is_han {
      cut_han(blk, out);
    } else {
      out.extend(re_skip_split(blk).into_iter().map(String::from));
    }
  }
}

/// Port of `finalseg.cut(sentence)`. `Force_Split_Words` is always empty here
/// because `add_word` is not ported.
pub fn cut(sentence: &str) -> Vec<String> {
  let mut out = Vec::new();
  cut_into(sentence, &mut out);
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn re_skip_split_matches_python() {
    assert_eq!(
      re_skip_split("ab3.5%+x.y"),
      vec!["ab3.5%", "+", "x", ".", "y"]
    );
    assert_eq!(re_skip_split("a.%"), vec!["a", ".%"]);
    assert_eq!(re_skip_split("1.٣"), vec!["1.٣"]);
  }
}
