//! Port of the Python wordsegment 1.3.1 package used by GPT-SoVITS.
//!
//! Python: `wordsegment/__init__.py`. English word segmentation based on the
//! unigram and bigram counts of a trillion-word corpus, from the chapter
//! "Natural Language Corpus Data" of the book "Beautiful Data" (Peter Norvig).
//!
//! ```
//! wordsegment::load();
//! assert_eq!(wordsegment::segment("thisisatest"), ["this", "is", "a", "test"]);
//! ```

#![allow(non_snake_case)]

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{LazyLock, RwLock, RwLockReadGuard};

/// A fast non-cryptographic hasher (the rustc `FxHasher` algorithm) for the
/// count tables. Hashing does not affect results, only lookup speed.
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
  hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
  #[inline]
  fn add(&mut self, word: u64) {
    self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
  }
}

impl Hasher for FxHasher {
  #[inline]
  fn write(&mut self, mut bytes: &[u8]) {
    while bytes.len() >= 8 {
      self.add(u64::from_le_bytes(bytes[..8].try_into().unwrap()));
      bytes = &bytes[8..];
    }
    if bytes.len() >= 4 {
      self.add(u64::from(u32::from_le_bytes(
        bytes[..4].try_into().unwrap(),
      )));
      bytes = &bytes[4..];
    }
    for &b in bytes {
      self.add(u64::from(b));
    }
  }

  #[inline]
  fn write_u8(&mut self, i: u8) {
    self.add(u64::from(i));
  }

  #[inline]
  fn write_usize(&mut self, i: usize) {
    self.add(i as u64);
  }

  #[inline]
  fn finish(&self) -> u64 {
    self.hash
  }
}

/// Python: the `dict` of word (or `"previous word"`) to float count used for
/// `Segmenter.unigrams` and `Segmenter.bigrams`.
pub type Counts = HashMap<String, f64, BuildHasherDefault<FxHasher>>;

thread_local! {
  /// Reused buffer for the `'{0} {1}'.format(previous, word)` bigram key.
  static BIGRAM_KEY: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Python: `Segmenter.UNIGRAMS_FILENAME` (the file contents, embedded).
pub const UNIGRAMS_DATA: &str = include_str!("../data/unigrams.txt");
/// Python: `Segmenter.BIGRAMS_FILENAME` (the file contents, embedded).
pub const BIGRAMS_DATA: &str = include_str!("../data/bigrams.txt");
/// Python: `Segmenter.WORDS_FILENAME` (the file contents, embedded).
pub const WORDS_DATA: &str = include_str!("../data/words.txt");

/// Python: `wordsegment.Segmenter`.
///
/// Support for object-oriented programming and customization.
#[derive(Debug, Clone, Default)]
pub struct Segmenter {
  /// Python: `Segmenter.unigrams`.
  pub unigrams: Counts,
  /// Python: `Segmenter.bigrams`.
  pub bigrams: Counts,
  /// Python: `Segmenter.total`.
  pub total: f64,
  /// Python: `Segmenter.limit`.
  pub limit: usize,
  /// Python: `Segmenter.words`.
  pub words: Vec<String>,
}

impl Segmenter {
  /// Python: `Segmenter.ALPHABET`.
  pub const ALPHABET: &'static str = "abcdefghijklmnopqrstuvwxyz0123456789";
  /// Python: `Segmenter.TOTAL`.
  pub const TOTAL: f64 = 1024908267229.0;
  /// Python: `Segmenter.LIMIT`.
  pub const LIMIT: usize = 24;

  /// Python: `Segmenter.__init__`.
  pub fn new() -> Self {
    Self::default()
  }

  /// Python: `Segmenter.load`.
  ///
  /// Load unigram and bigram counts from the embedded data files.
  pub fn load(&mut self) {
    self.unigrams.extend(Self::parse(UNIGRAMS_DATA));
    self.bigrams.extend(Self::parse(BIGRAMS_DATA));
    self.total = Self::TOTAL;
    self.limit = Self::LIMIT;
    self.words.extend(splitlines(WORDS_DATA));
  }

  /// Python: `Segmenter.parse`.
  ///
  /// Parse tab-separated file contents of word and count pairs. Takes the
  /// file contents instead of a file name. Later duplicate keys overwrite
  /// earlier ones, like `dict()`.
  ///
  /// # Panics
  ///
  /// Panics where Python raises: a line without exactly one tab
  /// (`ValueError` on unpacking) or a count that is not a number.
  pub fn parse(text: &str) -> Counts {
    let mut map = Counts::default();
    for line in text_lines(text) {
      let mut parts = line.split('\t');
      let (Some(word), Some(number), None) = (parts.next(), parts.next(), parts.next()) else {
        panic!("ValueError: line {line:?} is not a word and count pair");
      };
      map.insert(word.to_owned(), py_float(number));
    }
    map
  }

  /// Python: `Segmenter.score`.
  ///
  /// Score `word` in the context of `previous` word.
  ///
  /// # Panics
  ///
  /// Panics where Python raises: `ZeroDivisionError` when `total` is zero
  /// (the segmenter was not loaded) and `OverflowError` when `word` has more
  /// than 308 characters.
  pub fn score(&self, word: &str, previous: Option<&str>) -> f64 {
    let unigrams = &self.unigrams;
    let bigrams = &self.bigrams;
    let total = self.total;

    let Some(previous) = previous else {
      if let Some(&count) = unigrams.get(word) {
        // Probability of the given word.
        return py_div(count, total);
      }

      // Penalize words not found in the unigrams according
      // to their length, a crucial heuristic.
      return py_div(10.0, total * pow10(word.chars().count()));
    };

    // Python: `bigram = '{0} {1}'.format(previous, word)` and
    // `bigram in bigrams and previous in unigrams`. Both tests are pure
    // lookups, so testing `previous in unigrams` first (to skip building the
    // key) gives the same result.
    let bigram_count = if unigrams.contains_key(previous) {
      BIGRAM_KEY.with_borrow_mut(|bigram| {
        bigram.clear();
        bigram.push_str(previous);
        bigram.push(' ');
        bigram.push_str(word);
        bigrams.get(bigram.as_str()).copied()
      })
    } else {
      None
    };

    if let Some(count) = bigram_count {
      // Conditional probability of the word given the previous
      // word. The technical name is *stupid backoff* and it's
      // not a probability distribution but it works well in
      // practice.
      return py_div(py_div(count, total), self.score(previous, None));
    }

    // Fall back to using the unigram probability.
    self.score(word, None)
  }

  /// Python: `Segmenter.isegment`.
  ///
  /// Return iterator of words that is the best segmenation of `text`.
  /// Unlike the Python generator, the whole segmentation is computed before
  /// the iterator is returned.
  pub fn isegment(&self, text: &str) -> std::vec::IntoIter<String> {
    // Avoid recursion limit issues by dividing text into chunks, segmenting
    // those chunks and combining the results together. Chunks may divide
    // words in the middle so prefix chunks with the last five words of the
    // previous result.

    // `clean` only keeps ASCII characters, so byte offsets are character
    // offsets below.
    let clean_text = Self::clean(text);
    let size = 250;
    let mut prefix = String::new();
    let mut out = Vec::new();

    for offset in (0..clean_text.len()).step_by(size) {
      let end = (offset + size).min(clean_text.len());
      let chunk = &clean_text[offset..end];
      let (_, mut chunk_words) = Search::new(self, &format!("{prefix}{chunk}")).run();
      let keep = chunk_words.len().saturating_sub(5);
      prefix = chunk_words[keep..].concat();
      chunk_words.truncate(keep);
      out.extend(chunk_words);
    }

    let (_, prefix_words) = Search::new(self, &prefix).run();
    out.extend(prefix_words);

    out.into_iter()
  }

  /// Python: `Segmenter.segment`.
  ///
  /// Return list of words that is the best segmenation of `text`.
  pub fn segment(&self, text: &str) -> Vec<String> {
    self.isegment(text).collect()
  }

  /// Python: `Segmenter.divide`.
  ///
  /// Yield `(prefix, suffix)` pairs from `text`. Positions count characters
  /// (Python code points), not bytes.
  pub fn divide<'a>(&self, text: &'a str) -> impl Iterator<Item = (&'a str, &'a str)> + 'a {
    let mut bounds: Vec<usize> = text.char_indices().map(|(i, _)| i).skip(1).collect();
    bounds.push(text.len());
    let n = bounds.len().min(self.limit);
    let n = if text.is_empty() { 0 } else { n };
    bounds
      .into_iter()
      .take(n)
      .map(move |pos| (&text[..pos], &text[pos..]))
  }

  /// Python: `Segmenter.clean`.
  ///
  /// Return `text` lower-cased with non-alphanumeric characters removed.
  pub fn clean(text: &str) -> String {
    let alphabet = Self::ALPHABET;
    let text_lower = text.to_lowercase();
    text_lower
      .chars()
      .filter(|&letter| alphabet.contains(letter))
      .collect()
  }
}

/// Python: the word list of a `(score, words)` candidate in `search`, kept as
/// a shared linked list so memoized suffixes are not copied.
struct Words<'a> {
  word: &'a str,
  next: Option<Rc<Words<'a>>>,
}

/// Python: list comparison of two candidate word lists, element by element;
/// a list that is a prefix of the other compares smaller.
fn cmp_words(mut a: Option<&Rc<Words<'_>>>, mut b: Option<&Rc<Words<'_>>>) -> Ordering {
  loop {
    match (a, b) {
      (None, None) => return Ordering::Equal,
      (None, Some(_)) => return Ordering::Less,
      (Some(_), None) => return Ordering::Greater,
      (Some(x), Some(y)) => {
        let ord = x.word.cmp(y.word);
        if ord != Ordering::Equal {
          return ord;
        }
        a = x.next.as_ref();
        b = y.next.as_ref();
      }
    }
  }
}

/// Python: the `(score, words)` tuple returned by `search`.
type Candidate<'a> = (f64, Option<Rc<Words<'a>>>);

/// Python: tuple comparison `a > b` for `(float, list[str])` tuples, as used
/// by `max()` in `search`.
fn tuple_gt(a: &Candidate<'_>, b: &Candidate<'_>) -> bool {
  #[allow(clippy::float_cmp)]
  if a.0 != b.0 {
    return a.0 > b.0;
  }
  cmp_words(a.1.as_ref(), b.1.as_ref()) == Ordering::Greater
}

/// Python: the nested `search` function and its shared `memo` in
/// `Segmenter.isegment`.
///
/// Python keys the memo by `(suffix, prefix)` strings. The memo value is a
/// pure function of those strings, so keying it by byte offsets into the
/// searched text gives the same results.
struct Search<'a> {
  segmenter: &'a Segmenter,
  text: &'a str,
  memo: HashMap<(usize, usize), Candidate<'a>, BuildHasherDefault<FxHasher>>,
}

impl<'a> Search<'a> {
  fn new(segmenter: &'a Segmenter, text: &'a str) -> Self {
    Self {
      segmenter,
      text,
      memo: HashMap::default(),
    }
  }

  /// Python: `search(text)` with the default `previous='<s>'`.
  fn run(mut self) -> (f64, Vec<String>) {
    let (score, mut node) = self.search(0, "<s>");
    let mut words = Vec::new();
    while let Some(n) = node {
      words.push(n.word.to_owned());
      node = n.next.clone();
    }
    (score, words)
  }

  /// Python: `search(text[start:], previous)`.
  ///
  /// Return max of candidates matching `text` given `previous` word.
  fn search(&mut self, start: usize, previous: &str) -> Candidate<'a> {
    let text: &'a str = &self.text[start..];
    if text.is_empty() {
      return (0.0, None);
    }

    // Python: `max(candidates())`, which keeps the first maximum and only
    // replaces it with a strictly greater tuple.
    let mut best: Option<Candidate<'a>> = None;
    let divisions: Vec<(&'a str, &'a str)> = self.segmenter.divide(text).collect();
    for (prefix, _suffix) in divisions {
      let prefix_score = self.segmenter.score(prefix, Some(previous)).log10();

      let suffix_start = start + prefix.len();
      let pair = (start, suffix_start);
      let (suffix_score, suffix_words) = match self.memo.get(&pair) {
        Some(found) => found.clone(),
        None => {
          let found = self.search(suffix_start, prefix);
          self.memo.insert(pair, found.clone());
          found
        }
      };

      let words = Rc::new(Words {
        word: prefix,
        next: suffix_words,
      });
      let candidate = (prefix_score + suffix_score, Some(words));
      match &best {
        Some(current) if !tuple_gt(&candidate, current) => {}
        _ => best = Some(candidate),
      }
    }
    // Python raises `ValueError` from `max()` on an empty sequence; this
    // only happens when `limit` is 0.
    best.expect("ValueError: max() arg is an empty sequence")
  }
}

/// Python: `float(number)` for the non-negative integer counts in the data
/// files (`float()` strips surrounding whitespace such as the trailing
/// newline). Rust's parser rounds correctly, like CPython's.
fn py_float(number: &str) -> f64 {
  let trimmed = number.trim();
  trimmed
    .parse::<f64>()
    .unwrap_or_else(|_| panic!("ValueError: could not convert string to float: {number:?}"))
}

/// Python: float `a / b`, which raises `ZeroDivisionError` when `b == 0`.
fn py_div(a: f64, b: f64) -> f64 {
  assert!(b != 0.0, "ZeroDivisionError: float division by zero");
  a / b
}

/// Python: `float(10 ** n)`, correctly rounded, as done by `total * 10 ** n`.
fn pow10(n: usize) -> f64 {
  static TABLE: LazyLock<Vec<f64>> = LazyLock::new(|| {
    (0..=308)
      .map(|n| format!("1e{n}").parse().expect("valid float literal"))
      .collect()
  });
  assert!(n <= 308, "OverflowError: int too large to convert to float");
  TABLE[n]
}

/// Python: iterating a file opened with `io.open(..., encoding='utf-8')`
/// (universal newlines: `\r\n` and `\r` become `\n`), with the line ending
/// left on each line, as `Segmenter.parse` sees it.
fn text_lines(text: &str) -> impl Iterator<Item = String> + '_ {
  let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
  let mut lines: Vec<String> = normalized
    .split_inclusive('\n')
    .map(str::to_owned)
    .collect();
  lines.retain(|l| !l.is_empty());
  lines.into_iter()
}

/// Python: `str.splitlines()` on the universal-newline decoded text.
fn splitlines(text: &str) -> Vec<String> {
  let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
  let mut out = Vec::new();
  let mut current = String::new();
  for c in normalized.chars() {
    match c {
      '\n' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}' => {
        out.push(std::mem::take(&mut current))
      }
      _ => current.push(c),
    }
  }
  if !current.is_empty() {
    out.push(current);
  }
  out
}

/// Python: `wordsegment._segmenter`, the module-level segmenter.
static _SEGMENTER: LazyLock<RwLock<Segmenter>> = LazyLock::new(|| RwLock::new(Segmenter::new()));

fn read_segmenter() -> RwLockReadGuard<'static, Segmenter> {
  _SEGMENTER.read().unwrap_or_else(|e| e.into_inner())
}

/// Python: `wordsegment.load` (`_segmenter.load`).
pub fn load() {
  _SEGMENTER.write().unwrap_or_else(|e| e.into_inner()).load();
}

/// Python: `wordsegment.clean` (`_segmenter.clean`).
pub fn clean(text: &str) -> String {
  Segmenter::clean(text)
}

/// Python: `wordsegment.isegment` (`_segmenter.isegment`).
pub fn isegment(text: &str) -> std::vec::IntoIter<String> {
  read_segmenter().isegment(text)
}

/// Python: `wordsegment.segment` (`_segmenter.segment`).
pub fn segment(text: &str) -> Vec<String> {
  read_segmenter().segment(text)
}

/// Read access to one field of the module-level segmenter.
pub struct FieldGuard<T: 'static> {
  guard: RwLockReadGuard<'static, Segmenter>,
  get: fn(&Segmenter) -> &T,
}

impl<T> Deref for FieldGuard<T> {
  type Target = T;

  fn deref(&self) -> &T {
    (self.get)(&self.guard)
  }
}

/// Python: `wordsegment.UNIGRAMS` (`_segmenter.unigrams`).
pub fn UNIGRAMS() -> FieldGuard<Counts> {
  FieldGuard {
    guard: read_segmenter(),
    get: |s| &s.unigrams,
  }
}

/// Python: `wordsegment.BIGRAMS` (`_segmenter.bigrams`).
pub fn BIGRAMS() -> FieldGuard<Counts> {
  FieldGuard {
    guard: read_segmenter(),
    get: |s| &s.bigrams,
  }
}

/// Python: `wordsegment.WORDS` (`_segmenter.words`).
pub fn WORDS() -> FieldGuard<Vec<String>> {
  FieldGuard {
    guard: read_segmenter(),
    get: |s| &s.words,
  }
}

/// Python: `wordsegment._segmenter`, read access to the whole module-level
/// segmenter (for `score`, `total`, `limit`).
pub fn _segmenter() -> RwLockReadGuard<'static, Segmenter> {
  read_segmenter()
}
