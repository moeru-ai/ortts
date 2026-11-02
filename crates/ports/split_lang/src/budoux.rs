//! Port of the part of budoux 0.10.2 (`budoux/parser.py`) that split-lang
//! uses: `Parser`, `Parser.parse`, `load_default_japanese_parser` and
//! `load_default_simplified_chinese_parser`.
//!
//! budoux is licensed under the Apache License 2.0 (Copyright 2021 Google
//! LLC); see `LICENSE-APACHE` and `NOTICE` in this crate. The models in
//! `data/budoux/` are verbatim copies of `budoux/models/*.json`.

use std::collections::HashMap;
use std::sync::LazyLock;

/// The feature groups that `Parser.parse` reads, in the order it reads them.
const GROUPS: [&str; 13] = [
  "UW1", "UW2", "UW3", "UW4", "UW5", "UW6", "BW1", "BW2", "BW3", "TW1", "TW2", "TW3", "TW4",
];

/// Port of `class Parser`.
#[derive(Debug, Clone)]
pub struct Parser {
  /// `_model`: feature group -> feature -> score.
  pub _model: HashMap<String, HashMap<String, i64>>,
  /// `_base_score`: `-sum(sum(g.values()) for g in model.values()) * 0.5`.
  pub _base_score: f64,
}

impl Parser {
  /// Port of `Parser.__init__(model)`.
  pub fn new(model: HashMap<String, HashMap<String, i64>>) -> Self {
    let total: i64 = model.values().map(|g| g.values().sum::<i64>()).sum();
    Self {
      _model: model,
      _base_score: -(total as f64) * 0.5,
    }
  }

  /// Parses a budoux model JSON (`json.load`).
  pub fn from_json(json: &str) -> Self {
    Self::new(serde_json::from_str(json).expect("invalid budoux model"))
  }

  /// Port of `Parser.parse(sentence)`.
  ///
  /// Indices are code points, as in Python string indexing.
  pub fn parse(&self, sentence: &str) -> Vec<String> {
    if sentence.is_empty() {
      return Vec::new();
    }
    let empty = HashMap::new();
    let g: Vec<&HashMap<String, i64>> = GROUPS
      .iter()
      .map(|k| self._model.get(*k).unwrap_or(&empty))
      .collect();
    let [
      uw1,
      uw2,
      uw3,
      uw4,
      uw5,
      uw6,
      bw1,
      bw2,
      bw3,
      tw1,
      tw2,
      tw3,
      tw4,
    ] = g[..]
    else {
      unreachable!()
    };
    let chars: Vec<char> = sentence.chars().collect();
    let n = chars.len();
    let s = |a: usize, b: usize| -> String { chars[a..b].iter().collect() };
    let get =
      |m: &HashMap<String, i64>, key: String| -> f64 { m.get(&key).copied().unwrap_or(0) as f64 };
    let mut chunks = vec![chars[0].to_string()];
    // `for i in range(1, len(sentence))`, indexing neighbours as Python does.
    #[allow(clippy::needless_range_loop)]
    for i in 1..n {
      let mut score = self._base_score;
      if i > 2 {
        score += get(uw1, s(i - 3, i - 2));
      }
      if i > 1 {
        score += get(uw2, s(i - 2, i - 1));
      }
      score += get(uw3, s(i - 1, i));
      score += get(uw4, s(i, i + 1));
      if i + 1 < n {
        score += get(uw5, s(i + 1, i + 2));
      }
      if i + 2 < n {
        score += get(uw6, s(i + 2, i + 3));
      }

      if i > 1 {
        score += get(bw1, s(i - 2, i));
      }
      score += get(bw2, s(i - 1, i + 1));
      if i + 1 < n {
        score += get(bw3, s(i, i + 2));
      }

      if i > 2 {
        score += get(tw1, s(i - 3, i));
      }
      if i > 1 {
        score += get(tw2, s(i - 2, i + 1));
      }
      if i + 1 < n {
        score += get(tw3, s(i - 1, i + 2));
      }
      if i + 2 < n {
        score += get(tw4, s(i, i + 3));
      }

      if score > 0.0 {
        chunks.push(chars[i].to_string());
      } else {
        chunks.last_mut().expect("non-empty").push(chars[i]);
      }
    }
    chunks
  }
}

static JA_PARSER: LazyLock<Parser> =
  LazyLock::new(|| Parser::from_json(include_str!("../data/budoux/ja.json")));
static ZH_HANS_PARSER: LazyLock<Parser> =
  LazyLock::new(|| Parser::from_json(include_str!("../data/budoux/zh-hans.json")));

/// Port of `load_default_japanese_parser()` (the model is parsed once and
/// shared).
pub fn load_default_japanese_parser() -> &'static Parser {
  &JA_PARSER
}

/// Port of `load_default_simplified_chinese_parser()` (the model is parsed
/// once and shared).
pub fn load_default_simplified_chinese_parser() -> &'static Parser {
  &ZH_HANS_PARSER
}
