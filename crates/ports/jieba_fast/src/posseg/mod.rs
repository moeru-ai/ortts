//! Port of `jieba_fast.posseg`: part-of-speech tagging.

use std::sync::LazyLock;

use crate::fxhash::FxHashMap;
use crate::{Tokenizer, char_offsets, is_python_space, python_lines, split_runs};

pub mod viterbi;

use viterbi::viterbi;

/// `posseg/char_state_tab.py`, converted by `parity/export_data.py`.
pub static CHAR_STATE_TAB_P: &str = include_str!("../../data/posseg/char_state_tab.txt");
/// `posseg/prob_start.py`, converted by `parity/export_data.py`.
pub static PROB_START_P: &str = include_str!("../../data/posseg/prob_start.txt");
/// `posseg/prob_trans.py`, converted by `parity/export_data.py`.
pub static PROB_TRANS_P: &str = include_str!("../../data/posseg/prob_trans.txt");
/// `posseg/prob_emit.py`, converted by `parity/export_data.py`.
pub static PROB_EMIT_P: &str = include_str!("../../data/posseg/prob_emit.txt");

/// The posseg HMM tables (`char_state_tab_P`, `start_P`, `trans_P`,
/// `emit_P`) with the `(position, tag)` states numbered in `trans_P` key
/// order (`all_states = trans_p.keys()`).
pub struct Model {
  /// The states, e.g. `('B', "n")`.
  pub states: Vec<(char, String)>,
  /// Rank of each state in Python tuple order, for `max()` tie-breaking.
  pub rank: Vec<u16>,
  /// `char_state_tab_P`.
  pub char_state_tab: FxHashMap<char, Vec<u16>>,
  /// `start_P`.
  pub start_p: Vec<f64>,
  /// `trans_P` as a dense `from * len + to` matrix; missing keys hold
  /// `MIN_INF`, the default of `trans_p[y0].get(y, MIN_INF)`.
  pub trans_p: Vec<f64>,
  /// The keys of `trans_P[state]`.
  pub trans_keys: Vec<Vec<u16>>,
  /// `emit_P`.
  pub emit_p: Vec<FxHashMap<char, f64>>,
}

fn parse_state(s: &str) -> (char, String) {
  let (pos, tag) = s
    .split_once(',')
    .unwrap_or_else(|| panic!("invalid posseg state {s:?}"));
  let mut cs = pos.chars();
  let p = cs.next().unwrap();
  assert!(cs.next().is_none());
  (p, tag.to_string())
}

fn parse_f64(s: &str) -> f64 {
  s.parse()
    .unwrap_or_else(|_| panic!("invalid float {s:?} in posseg data"))
}

fn single_char(s: &str) -> char {
  let mut cs = s.chars();
  let c = cs.next().unwrap();
  assert!(cs.next().is_none(), "expected one character, got {s:?}");
  c
}

/// Port of `posseg.load_model()` / the table imports, reading the converted
/// tables.
pub fn load_model() -> Model {
  let mut states = Vec::new();
  let mut index: FxHashMap<String, u16> = FxHashMap::default();
  let trans_lines: Vec<&str> = PROB_TRANS_P.lines().collect();
  for line in &trans_lines {
    let name = line.split('\t').next().unwrap();
    index.insert(name.to_string(), states.len() as u16);
    states.push(parse_state(name));
  }
  let n = states.len();
  let idx = |name: &str| -> u16 {
    *index
      .get(name)
      .unwrap_or_else(|| panic!("KeyError: {name:?}"))
  };

  let mut trans_p = vec![viterbi::MIN_INF; n * n];
  let mut trans_keys = vec![Vec::new(); n];
  for line in &trans_lines {
    let mut fields = line.split('\t');
    let from = idx(fields.next().unwrap()) as usize;
    for f in fields {
      let (to, p) = f.rsplit_once(':').unwrap();
      let to = idx(to);
      trans_p[from * n + to as usize] = parse_f64(p);
      trans_keys[from].push(to);
    }
  }

  let mut start_p = vec![f64::NAN; n];
  for line in PROB_START_P.lines() {
    let (s, p) = line.split_once('\t').unwrap();
    start_p[idx(s) as usize] = parse_f64(p);
  }
  assert!(
    start_p.iter().all(|p| !p.is_nan()),
    "KeyError: state missing from start_P"
  );

  let mut emit_p = vec![FxHashMap::default(); n];
  let mut cur = None;
  for line in PROB_EMIT_P.lines() {
    if let Some(s) = line.strip_prefix('@') {
      cur = Some(idx(s) as usize);
      continue;
    }
    let (c, p) = line.split_once('\t').unwrap();
    emit_p[cur.unwrap()].insert(single_char(c), parse_f64(p));
  }

  let mut char_state_tab = FxHashMap::default();
  for line in CHAR_STATE_TAB_P.lines() {
    let mut fields = line.split('\t');
    let c = single_char(fields.next().unwrap());
    char_state_tab.insert(c, fields.map(idx).collect());
  }

  let mut order: Vec<u16> = (0..n as u16).collect();
  order.sort_by(|&a, &b| states[a as usize].cmp(&states[b as usize]));
  let mut rank = vec![0u16; n];
  for (r, &s) in order.iter().enumerate() {
    rank[s as usize] = r as u16;
  }

  Model {
    states,
    rank,
    char_state_tab,
    start_p,
    trans_p,
    trans_keys,
    emit_p,
  }
}

/// The posseg model.
pub static MODEL: LazyLock<Model> = LazyLock::new(load_model);

/// `re_han_detail = re.compile("([\u4E00-\u9FD5]+)")`.
pub fn re_han_detail(c: char) -> bool {
  ('\u{4E00}'..='\u{9FD5}').contains(&c)
}

/// `re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")`.
pub fn re_han_internal(c: char) -> bool {
  matches!(c, '\u{4E00}'..='\u{9FD5}' | 'a'..='z' | 'A'..='Z' | '0'..='9' | '+' | '#' | '&' | '.' | '_')
}

/// `re_num.match(x)` with `re_num = re.compile("[\.0-9]+")`.
fn re_num_match(x: &str) -> bool {
  x.starts_with(|c: char| c == '.' || c.is_ascii_digit())
}

/// `re_eng.match(x)` with `re_eng = re.compile("[a-zA-Z0-9]+")`.
fn re_eng_match(x: &str) -> bool {
  x.starts_with(|c: char| c.is_ascii_alphanumeric())
}

/// `re_eng1.match(x)` with `re_eng1 = re.compile('^[a-zA-Z0-9]$', re.U)`.
/// `$` also matches before a trailing `"\n"`, which a DAG word cannot have.
fn re_eng1_match(x: &str) -> bool {
  let mut cs = x.chars();
  match (cs.next(), cs.next(), cs.next()) {
    (Some(c), None, _) => c.is_ascii_alphanumeric(),
    (Some(c), Some('\n'), None) => c.is_ascii_alphanumeric(),
    _ => false,
  }
}

/// `re_skip_detail.split(blk)` with `([\.0-9]+|[a-zA-Z0-9]+)`, without the
/// empty strings.
fn re_skip_detail_split(blk: &str) -> Vec<&str> {
  let b = blk.as_bytes();
  let mut out = Vec::new();
  let mut start = 0;
  let mut i = 0;
  while i < b.len() {
    let c = b[i];
    let end = if c == b'.' || c.is_ascii_digit() {
      let mut j = i;
      while j < b.len() && (b[j] == b'.' || b[j].is_ascii_digit()) {
        j += 1;
      }
      Some(j)
    } else if c.is_ascii_alphabetic() {
      let mut j = i;
      while j < b.len() && b[j].is_ascii_alphanumeric() {
        j += 1;
      }
      Some(j)
    } else {
      None
    };
    match end {
      Some(j) => {
        if start < i {
          out.push(&blk[start..i]);
        }
        out.push(&blk[i..j]);
        i = j;
        start = j;
      }
      None => i += blk[i..].chars().next().unwrap().len_utf8(),
    }
  }
  if start < blk.len() {
    out.push(&blk[start..]);
  }
  out
}

/// Port of `jieba_fast.posseg.pair`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
  pub word: String,
  pub flag: String,
}

/// Port of `pair(word, flag)`.
fn pair(word: &str, flag: &str) -> Pair {
  Pair {
    word: word.to_string(),
    flag: flag.to_string(),
  }
}

/// Port of `jieba_fast.posseg.POSTokenizer`.
pub struct POSTokenizer<'a> {
  /// `POSTokenizer.tokenizer`.
  pub tokenizer: &'a Tokenizer,
  /// `POSTokenizer.word_tag_tab`.
  pub word_tag_tab: FxHashMap<Box<str>, Box<str>>,
}

impl<'a> POSTokenizer<'a> {
  /// Port of `POSTokenizer(tokenizer)` with the default dictionary file for
  /// `load_word_tag`.
  pub fn new(tokenizer: &'a Tokenizer) -> Self {
    Self::with_dict(tokenizer, crate::DEFAULT_DICT)
  }

  /// Port of `POSTokenizer(tokenizer)` where `tokenizer.get_dict_file()`
  /// returns `dict` (file contents).
  pub fn with_dict(tokenizer: &'a Tokenizer, dict: &str) -> Self {
    let mut t = Self {
      tokenizer,
      word_tag_tab: FxHashMap::default(),
    };
    t.load_word_tag(dict);
    t
  }

  /// Port of `POSTokenizer.load_word_tag(f)`, taking the file contents.
  pub fn load_word_tag(&mut self, f: &str) {
    self.word_tag_tab = FxHashMap::default();
    self.word_tag_tab.reserve(360_000);
    for (idx, raw) in python_lines(f).enumerate() {
      let line =
        raw.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'));
      if line.is_empty() {
        continue;
      }
      let parts: Vec<&str> = line.split(' ').collect();
      let [word, _, tag] = parts[..] else {
        panic!(
          "ValueError: invalid POS dictionary entry in <dict> at Line {}: {line}",
          idx + 1
        );
      };
      self.word_tag_tab.insert(word.into(), tag.into());
    }
  }

  /// `self.word_tag_tab.get(word, 'x')`.
  fn tag(&self, word: &str) -> &str {
    self.word_tag_tab.get(word).map_or("x", |t| t)
  }

  /// Port of `POSTokenizer.__cut(sentence)` (HMM tagging of a Han run).
  fn cut_hmm(&self, sentence: &str, out: &mut Vec<Pair>) {
    let model = &*MODEL;
    let chars: Vec<char> = sentence.chars().collect();
    let offs = char_offsets(sentence);
    let (_, pos_list) = viterbi(&chars, model);
    let (mut begin, mut nexti) = (0, 0);
    for i in 0..chars.len() {
      let (pos, ref tag) = model.states[pos_list[i] as usize];
      match pos {
        'B' => begin = i,
        'E' => {
          out.push(pair(&sentence[offs[begin]..offs[i + 1]], tag));
          nexti = i + 1;
        }
        'S' => {
          out.push(pair(&sentence[offs[i]..offs[i + 1]], tag));
          nexti = i + 1;
        }
        _ => {}
      }
    }
    if nexti < chars.len() {
      out.push(pair(
        &sentence[offs[nexti]..],
        &model.states[pos_list[nexti] as usize].1,
      ));
    }
  }

  /// Port of `POSTokenizer.__cut_detail(sentence)`.
  fn cut_detail(&self, sentence: &str, out: &mut Vec<Pair>) {
    for (blk, is_han) in split_runs(sentence, re_han_detail) {
      if is_han {
        self.cut_hmm(blk, out);
      } else {
        for x in re_skip_detail_split(blk) {
          if re_num_match(x) {
            out.push(pair(x, "m"));
          } else if re_eng_match(x) {
            out.push(pair(x, "eng"));
          } else {
            out.push(pair(x, "x"));
          }
        }
      }
    }
  }

  /// Port of `POSTokenizer.__cut_DAG_NO_HMM(sentence)`.
  fn cut_DAG_NO_HMM(&self, sentence: &str, out: &mut Vec<Pair>) {
    let dag = self.tokenizer.get_DAG(sentence);
    let route = self.tokenizer.calc(sentence, &dag);
    let offs = char_offsets(sentence);
    let n = offs.len() - 1;
    let mut x = 0;
    let mut buf = String::new();
    while x < n {
      let y = route[x].1 + 1;
      let l_word = &sentence[offs[x]..offs[y]];
      if re_eng1_match(l_word) {
        buf.push_str(l_word);
      } else {
        if !buf.is_empty() {
          out.push(pair(&buf, "eng"));
          buf.clear();
        }
        out.push(pair(l_word, self.tag(l_word)));
      }
      x = y;
    }
    if !buf.is_empty() {
      out.push(pair(&buf, "eng"));
    }
  }

  /// Flushes `buf` the way `POSTokenizer.__cut_DAG` does.
  fn flush_buf(&self, buf: &str, out: &mut Vec<Pair>) {
    if buf.chars().nth(1).is_none() {
      out.push(pair(buf, self.tag(buf)));
    } else if !self.tokenizer.freq_truthy(buf) {
      self.cut_detail(buf, out);
    } else {
      for (i, c) in buf.char_indices() {
        let elem = &buf[i..i + c.len_utf8()];
        out.push(pair(elem, self.tag(elem)));
      }
    }
  }

  /// Port of `POSTokenizer.__cut_DAG(sentence)`. It uses the pure Python
  /// `Tokenizer.get_DAG` and `Tokenizer.calc`, not the C route.
  fn cut_DAG(&self, sentence: &str, out: &mut Vec<Pair>) {
    let dag = self.tokenizer.get_DAG(sentence);
    let route = self.tokenizer.calc(sentence, &dag);
    let offs = char_offsets(sentence);
    let n = offs.len() - 1;
    let mut x = 0;
    let mut buf_start: Option<usize> = None;
    while x < n {
      let y = route[x].1 + 1;
      if y - x == 1 {
        buf_start.get_or_insert(x);
      } else {
        if let Some(bs) = buf_start.take() {
          self.flush_buf(&sentence[offs[bs]..offs[x]], out);
        }
        let l_word = &sentence[offs[x]..offs[y]];
        out.push(pair(l_word, self.tag(l_word)));
      }
      x = y;
    }
    if let Some(bs) = buf_start {
      self.flush_buf(&sentence[offs[bs]..], out);
    }
  }

  /// Port of `POSTokenizer.__cut_internal(sentence, HMM)`.
  /// `makesure_userdict_loaded` is a no-op without `add_word`.
  fn cut_internal(&self, sentence: &str, hmm: bool) -> Vec<Pair> {
    let mut out = Vec::new();
    for (blk, is_han) in split_runs(sentence, re_han_internal) {
      if is_han {
        if hmm {
          self.cut_DAG(blk, &mut out);
        } else {
          self.cut_DAG_NO_HMM(blk, &mut out);
        }
      } else {
        // re_skip_internal.split(blk), then per piece x.
        for x in skip_internal_split(blk) {
          if x.chars().next().is_some_and(is_python_space) {
            out.push(pair(x, "x"));
          } else {
            for (i, c) in x.char_indices() {
              let xx = &x[i..i + c.len_utf8()];
              if re_num_match(xx) {
                out.push(pair(xx, "m"));
              } else if re_eng_match(x) {
                out.push(pair(xx, "eng"));
              } else {
                out.push(pair(xx, "x"));
              }
            }
          }
        }
      }
    }
    out
  }

  /// Port of `POSTokenizer.cut(sentence, HMM=hmm)`.
  pub fn cut(&self, sentence: &str, hmm: bool) -> Vec<Pair> {
    self.cut_internal(sentence, hmm)
  }

  /// Port of `POSTokenizer.lcut(sentence, HMM=hmm)`.
  pub fn lcut(&self, sentence: &str, hmm: bool) -> Vec<Pair> {
    self.cut(sentence, hmm)
  }
}

/// `re_skip_internal.split(blk)` with `(\r\n|\s)`, without the empty
/// strings: whitespace tokens (`"\r\n"` as one) and the runs between them.
fn skip_internal_split(blk: &str) -> Vec<&str> {
  let mut out = Vec::new();
  let mut start = 0;
  let mut it = blk.char_indices().peekable();
  while let Some((i, c)) = it.next() {
    if is_python_space(c) {
      if start < i {
        out.push(&blk[start..i]);
      }
      let end = if c == '\r' && matches!(it.peek(), Some((_, '\n'))) {
        it.next();
        i + 2
      } else {
        i + c.len_utf8()
      };
      out.push(&blk[i..end]);
      start = end;
    }
  }
  if start < blk.len() {
    out.push(&blk[start..]);
  }
  out
}

/// Port of the default `jieba_fast.posseg.dt = POSTokenizer(jieba_fast.dt)`.
#[allow(non_upper_case_globals)]
pub static dt: LazyLock<POSTokenizer<'static>> = LazyLock::new(|| POSTokenizer::new(&crate::dt));

/// Port of `jieba_fast.posseg.cut(sentence, HMM=hmm)` (without the parallel
/// pool).
pub fn cut(sentence: &str, hmm: bool) -> Vec<Pair> {
  dt.cut(sentence, hmm)
}

/// Port of `jieba_fast.posseg.lcut(sentence, HMM=True)`.
pub fn lcut(sentence: &str, hmm: bool) -> Vec<Pair> {
  dt.lcut(sentence, hmm)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn re_skip_detail_split_matches_python() {
    assert_eq!(
      re_skip_detail_split("3abc.5x+y"),
      vec!["3", "abc", ".5", "x", "+", "y"]
    );
  }

  #[test]
  fn skip_internal_split_matches_python() {
    assert_eq!(
      skip_internal_split("，\r\n。 ！"),
      vec!["，", "\r\n", "。", " ", "！"]
    );
  }
}
