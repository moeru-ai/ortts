//! Port of `GPT_SoVITS/text/g2pw/utils.py` (GPT-SoVITS @ 04a44eb).
//!
//! Credits: this code is modified from <https://github.com/GitYCC/g2pW>
//! (PaddleSpeech, Apache-2.0).

use std::collections::BTreeMap;
use std::path::Path;

use tokenizers::Tokenizer;

use super::G2PWError;

/// Port of `wordize_and_map(text)`.
///
/// `text` is a list of code points (Python indexes `str` by code point).
/// Returns `(words, index_map_from_text_to_word, index_map_from_word_to_text)`;
/// `None` in the text-to-word map marks a space.
#[allow(clippy::type_complexity)]
pub fn wordize_and_map(text: &[char]) -> (Vec<String>, Vec<Option<usize>>, Vec<(usize, usize)>) {
  let mut words: Vec<String> = Vec::new();
  let mut index_map_from_text_to_word: Vec<Option<usize>> = Vec::new();
  let mut index_map_from_word_to_text: Vec<(usize, usize)> = Vec::new();
  let mut text = text;
  while !text.is_empty() {
    // re.match(r"^ +", text)
    let space_len = text.iter().take_while(|&&c| c == ' ').count();
    if space_len > 0 {
      index_map_from_text_to_word.extend(std::iter::repeat_n(None, space_len));
      text = &text[space_len..];
      continue;
    }

    // re.match(r"^[a-zA-Z0-9]+", text)
    let en_len = text
      .iter()
      .take_while(|c| c.is_ascii_alphanumeric())
      .count();
    if en_len > 0 {
      let en_word: String = text[..en_len].iter().collect();

      let word_start_pos = index_map_from_text_to_word.len();
      let word_end_pos = word_start_pos + en_len;
      index_map_from_word_to_text.push((word_start_pos, word_end_pos));

      index_map_from_text_to_word.extend(std::iter::repeat_n(Some(words.len()), en_len));

      words.push(en_word);
      text = &text[en_len..];
    } else {
      let word_start_pos = index_map_from_text_to_word.len();
      let word_end_pos = word_start_pos + 1;
      index_map_from_word_to_text.push((word_start_pos, word_end_pos));

      index_map_from_text_to_word.push(Some(words.len()));

      words.push(text[0].to_string());
      text = &text[1..];
    }
  }
  (
    words,
    index_map_from_text_to_word,
    index_map_from_word_to_text,
  )
}

/// `tokenizer.tokenize(word)` of the transformers fast tokenizer:
/// `encode_plus(word, add_special_tokens=False).tokens()`.
pub fn tokenize(tokenizer: &Tokenizer, word: &str) -> Result<Vec<String>, G2PWError> {
  let encoding = tokenizer
    .encode(word, false)
    .map_err(|e| G2PWError::Tokenizer(e.to_string()))?;
  Ok(encoding.get_tokens().to_vec())
}

/// Port of `tokenize_and_map(tokenizer, text)`.
///
/// Returns `(tokens, index_map_from_text_to_token, index_map_from_token_to_text)`.
/// As in Python, the text-to-token map starts as the text-to-word map and
/// each position covered by a token is overwritten with the token index. A
/// token whose length (without `##`) runs past the end of the text gives
/// `IndexError` (for example a Hangul syllable, which the BERT normalizer
/// decomposes into several jamo tokens).
#[allow(clippy::type_complexity)]
pub fn tokenize_and_map(
  tokenizer: &Tokenizer,
  text: &[char],
) -> Result<(Vec<String>, Vec<Option<usize>>, Vec<(usize, usize)>), G2PWError> {
  let (words, text2word, word2text) = wordize_and_map(text);

  let mut tokens: Vec<String> = Vec::new();
  let mut index_map_from_token_to_text: Vec<(usize, usize)> = Vec::new();
  for (word, &(word_start, word_end)) in words.iter().zip(word2text.iter()) {
    let word_tokens = tokenize(tokenizer, word)?;

    if word_tokens.is_empty() || word_tokens == ["[UNK]"] {
      index_map_from_token_to_text.push((word_start, word_end));
      tokens.push("[UNK]".to_string());
    } else {
      let mut current_word_start = word_start;
      for word_token in word_tokens {
        // len(re.sub(r"^##", "", word_token))
        let word_token_len = word_token
          .strip_prefix("##")
          .unwrap_or(&word_token)
          .chars()
          .count();
        index_map_from_token_to_text
          .push((current_word_start, current_word_start + word_token_len));
        current_word_start += word_token_len;
        tokens.push(word_token);
      }
    }
  }

  let mut index_map_from_text_to_token = text2word;
  for (i, &(token_start, token_end)) in index_map_from_token_to_text.iter().enumerate() {
    for token_pos in token_start..token_end {
      match index_map_from_text_to_token.get_mut(token_pos) {
        Some(slot) => *slot = Some(i),
        None => {
          return Err(G2PWError::IndexError(
            "list assignment index out of range".to_string(),
          ));
        }
      }
    }
  }

  Ok((
    tokens,
    index_map_from_text_to_token,
    index_map_from_token_to_text,
  ))
}

/// A Python literal value of a `config.py` assignment (the subset that
/// `G2PWModel/config.py` and `default_config_dict` use).
#[derive(Clone, Debug, PartialEq)]
pub enum PyValue {
  None,
  Bool(bool),
  Int(i64),
  Float(f64),
  Str(String),
  List(Vec<PyValue>),
  /// Keys in insertion order.
  Dict(Vec<(String, PyValue)>),
}

impl PyValue {
  /// Python truthiness (`bool(value)`).
  pub fn truthy(&self) -> bool {
    match self {
      PyValue::None => false,
      PyValue::Bool(b) => *b,
      PyValue::Int(i) => *i != 0,
      PyValue::Float(f) => *f != 0.0,
      PyValue::Str(s) => !s.is_empty(),
      PyValue::List(l) => !l.is_empty(),
      PyValue::Dict(d) => !d.is_empty(),
    }
  }

  /// The string, if this is a `str`.
  pub fn as_str(&self) -> Option<&str> {
    match self {
      PyValue::Str(s) => Some(s),
      _ => None,
    }
  }
}

/// The module object that `load_config` returns: the module-level names of
/// `config.py` in assignment order (a later assignment replaces the value).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Config {
  /// `vars(config)` restricted to the assigned names.
  pub attrs: BTreeMap<String, PyValue>,
}

impl Config {
  /// `getattr(config, name)`.
  pub fn get(&self, name: &str) -> Option<&PyValue> {
    self.attrs.get(name)
  }

  /// `config.model_source`.
  pub fn model_source(&self) -> Option<&str> {
    self.get("model_source").and_then(PyValue::as_str)
  }

  /// `bool(config.use_mask)`.
  pub fn use_mask(&self) -> bool {
    self.get("use_mask").is_some_and(PyValue::truthy)
  }

  /// `bool(config.use_char_phoneme)`.
  pub fn use_char_phoneme(&self) -> bool {
    self.get("use_char_phoneme").is_some_and(PyValue::truthy)
  }
}

/// Port of `_load_config(config_path)`.
///
/// Python executes `config.py` as a module. The port parses it instead: it
/// accepts module-level assignments `name = <literal>` where the literal is
/// `None`, `True`, `False`, an int, a float, a string, a list or a dict
/// (possibly over several lines), plus comments and blank lines. Anything
/// else gives `G2PWError::ValueError`. `G2PWModel/config.py` of
/// `G2PWModel_1.1.zip` has only such assignments.
pub fn _load_config(config_path: &Path) -> Result<Config, G2PWError> {
  let source = std::fs::read_to_string(config_path)
    .map_err(|e| G2PWError::FileNotFoundError(format!("{}: {e}", config_path.display())))?;
  parse_config(&source)
}

/// Parses the text of a `config.py` (see [`_load_config`]).
pub fn parse_config(source: &str) -> Result<Config, G2PWError> {
  let chars: Vec<char> = source.chars().collect();
  let mut p = LiteralParser { s: &chars, pos: 0 };
  let mut config = Config::default();
  loop {
    p.skip_blank_lines();
    if p.pos >= p.s.len() {
      break;
    }
    let name = p.identifier()?;
    p.skip_inline_ws();
    if !p.eat('=') {
      return Err(p.error("expected '='"));
    }
    let value = p.value()?;
    p.skip_inline_ws();
    if p.peek() == Some('#') {
      p.skip_comment();
    }
    match p.peek() {
      None | Some('\n') => {}
      Some(c) => return Err(p.error(&format!("unexpected {c:?} after value"))),
    }
    config.attrs.insert(name, value);
  }
  Ok(config)
}

struct LiteralParser<'a> {
  s: &'a [char],
  pos: usize,
}

impl LiteralParser<'_> {
  fn error(&self, msg: &str) -> G2PWError {
    G2PWError::ValueError(format!("config.py: {msg} at offset {}", self.pos))
  }

  fn peek(&self) -> Option<char> {
    self.s.get(self.pos).copied()
  }

  fn eat(&mut self, c: char) -> bool {
    if self.peek() == Some(c) {
      self.pos += 1;
      true
    } else {
      false
    }
  }

  fn skip_comment(&mut self) {
    while let Some(c) = self.peek() {
      if c == '\n' {
        break;
      }
      self.pos += 1;
    }
  }

  fn skip_inline_ws(&mut self) {
    while matches!(self.peek(), Some(' ' | '\t' | '\r')) {
      self.pos += 1;
    }
  }

  /// Whitespace, newlines and comments (inside brackets and between
  /// statements).
  fn skip_blank_lines(&mut self) {
    loop {
      match self.peek() {
        Some(' ' | '\t' | '\r' | '\n') => self.pos += 1,
        Some('#') => self.skip_comment(),
        _ => break,
      }
    }
  }

  fn identifier(&mut self) -> Result<String, G2PWError> {
    let start = self.pos;
    while let Some(c) = self.peek() {
      if c == '_' || c.is_alphanumeric() {
        self.pos += 1;
      } else {
        break;
      }
    }
    if start == self.pos || self.s[start].is_ascii_digit() {
      return Err(self.error("expected a name"));
    }
    Ok(self.s[start..self.pos].iter().collect())
  }

  fn value(&mut self) -> Result<PyValue, G2PWError> {
    self.skip_inline_ws();
    match self.peek() {
      Some('\'' | '"') => self.string().map(PyValue::Str),
      Some('[') => {
        self.pos += 1;
        let mut items = Vec::new();
        loop {
          self.skip_blank_lines();
          if self.eat(']') {
            break;
          }
          items.push(self.value()?);
          self.skip_blank_lines();
          if !self.eat(',') {
            self.skip_blank_lines();
            if !self.eat(']') {
              return Err(self.error("expected ',' or ']'"));
            }
            break;
          }
        }
        Ok(PyValue::List(items))
      }
      Some('{') => {
        self.pos += 1;
        let mut items: Vec<(String, PyValue)> = Vec::new();
        loop {
          self.skip_blank_lines();
          if self.eat('}') {
            break;
          }
          let key = match self.value()? {
            PyValue::Str(s) => s,
            _ => return Err(self.error("only str dict keys are supported")),
          };
          self.skip_blank_lines();
          if !self.eat(':') {
            return Err(self.error("expected ':'"));
          }
          self.skip_blank_lines();
          let value = self.value()?;
          // A repeated key keeps its first position and takes the new value.
          match items.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => items.push((key, value)),
          }
          self.skip_blank_lines();
          if !self.eat(',') {
            self.skip_blank_lines();
            if !self.eat('}') {
              return Err(self.error("expected ',' or '}'"));
            }
            break;
          }
        }
        Ok(PyValue::Dict(items))
      }
      Some(c) if c.is_ascii_digit() || c == '-' || c == '+' || c == '.' => self.number(),
      Some(_) => {
        let name = self.identifier()?;
        match name.as_str() {
          "None" => Ok(PyValue::None),
          "True" => Ok(PyValue::Bool(true)),
          "False" => Ok(PyValue::Bool(false)),
          _ => Err(self.error(&format!("unsupported expression {name:?}"))),
        }
      }
      None => Err(self.error("expected a value")),
    }
  }

  fn string(&mut self) -> Result<String, G2PWError> {
    let quote = self.peek().unwrap();
    self.pos += 1;
    let mut out = String::new();
    loop {
      match self.peek() {
        None | Some('\n') => return Err(self.error("unterminated string")),
        Some(c) if c == quote => {
          self.pos += 1;
          return Ok(out);
        }
        Some('\\') => {
          self.pos += 1;
          let c = self
            .peek()
            .ok_or_else(|| self.error("unterminated string"))?;
          self.pos += 1;
          out.push(match c {
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            '0' => '\0',
            '\\' | '\'' | '"' => c,
            _ => return Err(self.error("unsupported string escape")),
          });
        }
        Some(c) => {
          out.push(c);
          self.pos += 1;
        }
      }
    }
  }

  fn number(&mut self) -> Result<PyValue, G2PWError> {
    let start = self.pos;
    while let Some(c) = self.peek() {
      if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+' | '_') {
        self.pos += 1;
      } else {
        break;
      }
    }
    let text: String = self.s[start..self.pos]
      .iter()
      .filter(|&&c| c != '_')
      .collect();
    if let Ok(i) = text.parse::<i64>() {
      return Ok(PyValue::Int(i));
    }
    text
      .parse::<f64>()
      .map(PyValue::Float)
      .map_err(|_| self.error(&format!("unsupported number {text:?}")))
  }
}

/// Port of `default_config_dict`.
pub fn default_config_dict() -> Vec<(String, PyValue)> {
  use PyValue::*;
  let s = |v: &str| Str(v.to_string());
  let d = |items: Vec<(&str, PyValue)>| {
    Dict(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
  };
  vec![
    ("manual_seed", Int(1313)),
    ("model_source", s("bert-base-chinese")),
    ("window_size", Int(32)),
    ("num_workers", Int(2)),
    ("use_mask", Bool(true)),
    ("use_char_phoneme", Bool(false)),
    ("use_conditional", Bool(true)),
    (
      "param_conditional",
      d(vec![
        ("affect_location", s("softmax")),
        ("bias", Bool(true)),
        ("char-linear", Bool(true)),
        ("pos-linear", Bool(false)),
        ("char+pos-second", Bool(true)),
        ("char+pos-second_lowrank", Bool(false)),
        ("lowrank_size", Int(0)),
        ("char+pos-second_fm", Bool(false)),
        ("fm_size", Int(0)),
        ("fix_mode", None),
        ("count_json", s("train.count.json")),
      ]),
    ),
    ("lr", Float(5e-5)),
    ("val_interval", Int(200)),
    ("num_iter", Int(10000)),
    ("use_focal", Bool(false)),
    (
      "param_focal",
      d(vec![("alpha", Float(0.0)), ("gamma", Float(0.7))]),
    ),
    ("use_pos", Bool(true)),
    (
      // The key has a trailing space upstream.
      "param_pos ",
      d(vec![
        ("weight", Float(0.1)),
        ("pos_joint_training", Bool(true)),
        ("train_pos_path", s("train.pos")),
        ("valid_pos_path", s("dev.pos")),
        ("test_pos_path", s("test.pos")),
      ]),
    ),
  ]
  .into_iter()
  .map(|(k, v)| (k.to_string(), v))
  .collect()
}

/// Port of `load_config(config_path, use_default=False)`.
///
/// With `use_default`, a missing attribute takes the default value and a
/// dict attribute gets the missing keys of the default dict. A non-dict
/// attribute whose default is a dict gives `TypeError`, as `k not in d`
/// would for most types in Python.
pub fn load_config(config_path: &Path, use_default: bool) -> Result<Config, G2PWError> {
  let mut config = _load_config(config_path)?;
  if use_default {
    apply_default_config(&mut config)?;
  }
  Ok(config)
}

/// The `use_default` merge of [`load_config`].
pub fn apply_default_config(config: &mut Config) -> Result<(), G2PWError> {
  for (attr, val) in default_config_dict() {
    match config.attrs.get_mut(&attr) {
      None => {
        config.attrs.insert(attr, val);
      }
      Some(existing) => {
        if let PyValue::Dict(default_items) = val {
          let PyValue::Dict(d) = existing else {
            return Err(G2PWError::TypeError(format!("config.{attr} is not a dict")));
          };
          for (dict_k, dict_v) in default_items {
            if !d.iter().any(|(k, _)| *k == dict_k) {
              d.push((dict_k, dict_v));
            }
          }
        }
      }
    }
  }
  Ok(())
}
