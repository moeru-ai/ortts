//! Port of `nltk.tokenize.casual` (Twitter-aware tokenizer).
//!
//! The regular expression sources are copied verbatim from nltk 3.10.4. nltk
//! compiles them with the third-party `regex` module (through `nltk.redos`);
//! here they go through [`compile`], which applies the Python `VERBOSE` rules
//! itself and hands the result to `fancy-regex` (a backtracking engine, needed
//! for the lookarounds and backreferences). See the crate README for the
//! remaining differences between the two engines.

use std::collections::HashMap;
use std::sync::OnceLock;

use fancy_regex::{Captures, Regex, RegexBuilder};
use serde::Deserialize;

use crate::python;

/// Port of `nltk.tokenize.casual.EMOTICONS` (ASCII emoticons).
pub const EMOTICONS: &str = r#"
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      </?3                       # heart
    )"#;

/// Port of `nltk.tokenize.casual.URLS` (John Gruber's URL pattern, modified by Tom Winzig).
pub const URLS: &str = r#"			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]{1,255}[.]              # bounded to avoid catastrophic backtracking (ReDoS)
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]{0,255}?\([^\s()]{1,255}\)[^\s()]{0,255}?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]{1,255}?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]{0,255}?\([^\s()]{1,255}\)[^\s()]{0,255}?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]{1,255}?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+){0,126}  # bounded to avoid catastrophic backtracking (ReDoS); DNS allows at most 127 labels
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"#;

/// Port of `nltk.tokenize.casual.FLAGS` (emoji flag sequences).
pub const FLAGS: &str = r#"
  (?:
    [\U0001F1E6-\U0001F1FF]{2}  # all enclosed letter pairs
    |
    # English flag
    \U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006e\U000E0067\U000E007F
    |
    # Scottish flag
    \U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F
    |
    # For Wales? Why Richard, it profit a man nothing to give his soul for the whole world … but for Wales!
    \U0001F3F4\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F
  )
"#;

/// Port of `nltk.tokenize.casual.PHONE_REGEX`.
pub const PHONE_REGEX: &str = r#"
    (?:
      (?:            # (international)
        \+?[01]
        [ *\-.\)]*
      )?
      (?:            # (area code)
        [\(]?
        \d{3}
        [ *\-.\)]*
      )?
      \d{3}          # exchange
      [ *\-.\)]*
      \d{4}          # base
    )"#;

/// Port of `nltk.tokenize.casual.REGEXPS` (the components of the tokenizer).
pub const REGEXPS: [&str; 10] = [
  URLS,
  EMOTICONS,
  r#"<[^>\s]+>"#,
  r#"[\-]+>|<[\-]+"#,
  r#"(?:@[\w_]+)"#,
  r#"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"#,
  r#"[\w.+-]{1,64}@[\w-]{1,63}\.(?:[\w-]\.?){1,251}[\w-]"#,
  r#"\S(?:
        [\U0001f3fb-\U0001f3ff]?(?:\u200d\S[\U0001f3fb-\U0001f3ff]?)+
        |
        [\U0001f3fb-\U0001f3ff]
    )"#,
  FLAGS,
  r#"
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:[^\S\r\n]*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    "#,
];

/// Source of `nltk.tokenize.casual.HANG_RE`.
pub const HANG_RE_SRC: &str = r#"([^\p{L}\p{N}])\1{3,}"#;

/// Source of `nltk.tokenize.casual.ENT_RE`.
pub const ENT_RE_SRC: &str = r#"&(#?(x?))([^&;\s]+);"#;

/// Source of `nltk.tokenize.casual.HANDLES_RE`.
pub const HANDLES_RE_SRC: &str =
  r#"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))"#;

/// Port of `nltk.tokenize.casual.REGEXPS_PHONE`: [`REGEXPS`] with
/// [`PHONE_REGEX`] as the second element.
pub const REGEXPS_PHONE: [&str; 11] = [
  REGEXPS[0],
  PHONE_REGEX,
  REGEXPS[1],
  REGEXPS[2],
  REGEXPS[3],
  REGEXPS[4],
  REGEXPS[5],
  REGEXPS[6],
  REGEXPS[7],
  REGEXPS[8],
  REGEXPS[9],
];

/// Python `regex` flags used by `nltk.tokenize.casual`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Flags {
  /// `regex.VERBOSE`
  pub verbose: bool,
  /// `regex.I`
  pub ignorecase: bool,
}

/// `regex.VERBOSE | regex.I | regex.UNICODE`, the flags of `WORD_RE`, `PHONE_WORD_RE`
/// and `EMOTICON_RE`. The other regexes use the default flags.
pub const VERBOSE_I: Flags = Flags {
  verbose: true,
  ignorecase: true,
};

/// Python `regex` limits a match by wall-clock time (`nltk.redos`,
/// 5 seconds); `fancy-regex` limits it by backtracking steps instead.
const BACKTRACK_LIMIT: usize = 100_000_000;

/// Removes whitespace and `#` comments outside character sets, as the
/// `regex` module parser does for `regex.VERBOSE` (`Source.get` with
/// `ignore_space`, which `parse_set` and `parse_escape` turn off).
///
/// Whitespace is what Python `str.isspace` accepts. The output is a
/// non-verbose pattern, because `fancy-regex` / `regex` ignore whitespace
/// inside character sets in verbose mode while Python keeps it.
pub fn strip_verbose(src: &str) -> String {
  let mut out = String::with_capacity(src.len());
  let mut chars = src.chars().peekable();
  while let Some(c) = chars.next() {
    match c {
      '\\' => {
        out.push(c);
        if let Some(n) = chars.next() {
          out.push(n);
        }
      }
      '[' => {
        // A set is copied verbatim up to its closing `]`. A `]` right
        // after `[` or `[^` is literal.
        out.push(c);
        if let Some(n) = chars.next_if_eq(&'^') {
          out.push(n);
        }
        if let Some(n) = chars.next_if_eq(&']') {
          out.push(n);
        }
        while let Some(n) = chars.next() {
          out.push(n);
          match n {
            '\\' => {
              if let Some(e) = chars.next() {
                out.push(e);
              }
            }
            ']' => break,
            _ => {}
          }
        }
      }
      '#' => {
        for n in chars.by_ref() {
          if n == '\n' {
            break;
          }
        }
      }
      c if python::char_isspace(c) => {}
      _ => out.push(c),
    }
  }
  out
}

#[derive(Deserialize)]
struct RegexSets {
  default: HashMap<String, Vec<(u32, u32)>>,
  ignorecase: HashMap<String, Vec<(u32, u32)>>,
}

/// The code points that each character-class atom of the nltk regexes
/// matches in the Python `regex` module, generated by
/// `parity/gen_data.py`.
fn regex_sets(ignorecase: bool) -> &'static HashMap<String, Vec<(u32, u32)>> {
  static SETS: OnceLock<RegexSets> = OnceLock::new();
  let sets = SETS.get_or_init(|| {
    serde_json::from_str(include_str!("../../data/regex_sets.json"))
      .expect("data/regex_sets.json is valid")
  });
  if ignorecase {
    &sets.ignorecase
  } else {
    &sets.default
  }
}

/// Index just past the character set that starts at `src[i] == '['`.
fn set_end(src: &[char], i: usize) -> usize {
  let mut j = i + 1;
  if src.get(j) == Some(&'^') {
    j += 1;
  }
  if src.get(j) == Some(&']') {
    j += 1;
  }
  while j < src.len() {
    let c = src[j];
    j += 1;
    if c == '\\' {
      j += 1;
    } else if c == ']' {
      break;
    }
  }
  j.min(src.len())
}

fn render_set(atom: &str, ignorecase: bool) -> String {
  let ranges = regex_sets(ignorecase).get(atom).unwrap_or_else(|| {
        panic!("no code point table for regex atom {atom:?} (ignorecase={ignorecase}); rerun `pixi run --manifest-path crates/ports/nltk/pixi.toml data`")
    });
  if ranges.is_empty() {
    return r"[^\x{0}-\x{10FFFF}]".to_owned();
  }
  let mut out = String::from("[");
  for &(lo, hi) in ranges {
    if lo == hi {
      out.push_str(&format!(r"\x{{{lo:X}}}"));
    } else {
      out.push_str(&format!(r"\x{{{lo:X}}}-\x{{{hi:X}}}"));
    }
  }
  out.push(']');
  out
}

/// A piece of a non-verbose pattern, see [`scan`].
enum Piece {
  /// A character-class atom, looked up in [`regex_sets`].
  Atom(String),
  /// `\b`
  WordBoundary,
  /// Anything else, copied as is.
  Text(String),
}

/// Splits a non-verbose pattern into the atoms whose meaning depends on the
/// Unicode database or on case folding, `\b`, and the rest.
///
/// Atoms are character sets, `\w \W \d \D \s \S`, `\p{..}`, `\P{..}` and `.`,
/// plus, with `ignorecase`, literal letters, non-ASCII literals and
/// `\U \u \x` escapes. `parity/gen_data.py` has the same scanner.
fn scan(src: &str, ignorecase: bool) -> Vec<Piece> {
  let src: Vec<char> = src.chars().collect();
  let text = |a: &[char]| a.iter().collect::<String>();
  let mut out = Vec::new();
  let mut i = 0;
  while i < src.len() {
    let c = src[i];
    if c == '\\' {
      let e = src.get(i + 1).copied().unwrap_or('\\');
      let atom_len = match e {
        'w' | 'W' | 'd' | 'D' | 's' | 'S' => Some(2),
        'p' | 'P' if src.get(i + 2) == Some(&'{') => {
          let close = src[i..].iter().position(|&c| c == '}').expect("closing }");
          Some(close + 1)
        }
        'U' if ignorecase => Some(10),
        'u' if ignorecase => Some(6),
        'x' if ignorecase => Some(4),
        _ => None,
      };
      let len = atom_len.unwrap_or(2).min(src.len() - i);
      let piece = text(&src[i..i + len]);
      out.push(if atom_len.is_some() {
        Piece::Atom(piece)
      } else if e == 'b' {
        Piece::WordBoundary
      } else {
        Piece::Text(piece)
      });
      i += len;
    } else if c == '[' {
      let j = set_end(&src, i);
      out.push(Piece::Atom(text(&src[i..j])));
      i = j;
    } else if c == '.' || (ignorecase && (c.is_alphabetic() || (c as u32) > 127)) {
      out.push(Piece::Atom(c.to_string()));
      i += 1;
    } else {
      out.push(Piece::Text(c.to_string()));
      i += 1;
    }
  }
  out
}

/// The atoms of a non-verbose pattern: character sets, `\w \W \d \D \s \S`,
/// `\p{..}`, `\P{..}` and `.`, plus, with `ignorecase`, literal letters,
/// non-ASCII literals and `\U \u \x` escapes. `\b` counts as `\w`.
pub fn atoms(src: &str, ignorecase: bool) -> Vec<String> {
  scan(src, ignorecase)
    .into_iter()
    .filter_map(|p| match p {
      Piece::Atom(a) => Some(a),
      Piece::WordBoundary => Some(r"\w".to_owned()),
      Piece::Text(_) => None,
    })
    .collect()
}

/// Replaces every atom (see [`scan`]) with the explicit set of code points
/// that the Python `regex` module matches for it, and `\b` with a lookaround
/// on the `\w` set. The result needs no `(?i)`, so the Rust engine's own
/// Unicode tables and case folding are never used.
fn substitute_atoms(src: &str, ignorecase: bool) -> String {
  let mut out = String::with_capacity(src.len() * 4);
  for piece in scan(src, ignorecase) {
    match piece {
      Piece::Atom(a) => out.push_str(&render_set(&a, ignorecase)),
      Piece::WordBoundary => {
        let w = render_set(r"\w", ignorecase);
        out.push_str(&format!("(?:(?<={w})(?!{w})|(?<!{w})(?={w}))"));
      }
      Piece::Text(t) => out.push_str(&t),
    }
  }
  out
}

/// Port of `nltk.redos.compile(pattern, flags)`.
///
/// The pattern is translated for `fancy-regex`: [`strip_verbose`] applies
/// `regex.VERBOSE`, and every Unicode- or case-dependent atom is replaced by
/// the code points the Python `regex` module matches for it, which also
/// applies `regex.I`.
///
/// Panics if the pattern does not compile (all patterns are constants).
pub fn compile(pattern: &str, flags: Flags) -> Regex {
  let src = if flags.verbose {
    strip_verbose(pattern)
  } else {
    pattern.to_owned()
  };
  let src = substitute_atoms(&src, flags.ignorecase);
  RegexBuilder::new(&src)
    .backtrack_limit(BACKTRACK_LIMIT)
    .build()
    .unwrap_or_else(|e| panic!("pattern {src:?} does not compile: {e}"))
}

/// Port of `nltk.tokenize.casual.HANG_RE`.
pub fn hang_re() -> &'static Regex {
  static RE: OnceLock<Regex> = OnceLock::new();
  RE.get_or_init(|| compile(HANG_RE_SRC, Flags::default()))
}

/// Port of `nltk.tokenize.casual.EMOTICON_RE`.
pub fn emoticon_re() -> &'static Regex {
  static RE: OnceLock<Regex> = OnceLock::new();
  RE.get_or_init(|| compile(EMOTICONS, VERBOSE_I))
}

/// Port of `nltk.tokenize.casual.ENT_RE`.
pub fn ent_re() -> &'static Regex {
  static RE: OnceLock<Regex> = OnceLock::new();
  RE.get_or_init(|| compile(ENT_RE_SRC, Flags::default()))
}

/// Port of `nltk.tokenize.casual.HANDLES_RE`.
pub fn handles_re() -> &'static Regex {
  static RE: OnceLock<Regex> = OnceLock::new();
  RE.get_or_init(|| compile(HANDLES_RE_SRC, Flags::default()))
}

/// Error raised when a match exceeds the backtracking limit (Python raises
/// `TimeoutError` when the `nltk.redos` time limit is exceeded).
#[derive(Debug)]
pub struct TimeoutError(pub Box<fancy_regex::Error>);

impl From<fancy_regex::Error> for TimeoutError {
  fn from(e: fancy_regex::Error) -> Self {
    Self(Box::new(e))
  }
}

impl std::fmt::Display for TimeoutError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "TweetTokenizer exceeded its backtracking limit; the input may be adversarial: {}",
      self.0
    )
  }
}

impl std::error::Error for TimeoutError {}

/// Port of `nltk.tokenize.casual._replace_html_entities`.
///
/// Removes entities from text by converting them to their corresponding
/// unicode character. `keep` lists entity names that are not replaced. When
/// `remove_illegal` is true, entities that cannot be converted are removed,
/// otherwise they are kept as is.
///
/// Python returns a lone surrogate for `&#xD800;` to `&#xDFFF;`; a Rust
/// string cannot hold one, so this returns U+FFFD instead.
pub fn _replace_html_entities(
  text: &str,
  keep: &[&str],
  remove_illegal: bool,
) -> Result<String, TimeoutError> {
  let convert_entity = |caps: &Captures| -> String {
    let whole = caps.get(0).map_or("", |m| m.as_str());
    let group = |i| caps.get(i).map_or("", |m| m.as_str());
    let entity_body = group(3);
    let number: Option<i64>;
    if !group(1).is_empty() {
      let base = if group(2).is_empty() { 10 } else { 16 };
      match python::int(entity_body, base) {
        Ok(n) => {
          if (0x80..=0x9F).contains(&n) {
            // bytes((number,)).decode("cp1252"); UnicodeDecodeError
            // is a ValueError, which sets number to None.
            match python::cp1252_c1(n as u8) {
              Some(c) => return c.to_string(),
              None => number = None,
            }
          } else {
            number = Some(n);
          }
        }
        Err(python::ValueError) => number = None,
      }
    } else {
      if keep.contains(&entity_body) {
        return whole.to_owned();
      }
      number = python::name2codepoint()
        .get(entity_body)
        .map(|&n| i64::from(n));
    }
    if let Some(n) = number {
      // chr(number) raises ValueError / OverflowError outside the
      // Unicode range.
      if (0..=0x10FFFF).contains(&n) {
        return char::from_u32(n as u32)
          .unwrap_or(char::REPLACEMENT_CHARACTER)
          .to_string();
      }
    }
    if remove_illegal {
      String::new()
    } else {
      whole.to_owned()
    }
  };
  Ok(
    ent_re()
      .try_replacen(text, 0, convert_entity)
      .map_err(TimeoutError::from)?
      .into_owned(),
  )
}

/// Port of `nltk.tokenize.casual.TweetTokenizer`.
#[derive(Debug, Clone, Copy)]
pub struct TweetTokenizer {
  /// Whether to preserve the casing of the text. Defaults to true.
  pub preserve_case: bool,
  /// Whether to replace repeated character sequences of length 3 or greater
  /// with sequences of length 3. Defaults to false.
  pub reduce_len: bool,
  /// Whether to remove Twitter handles. Defaults to false.
  pub strip_handles: bool,
  /// Whether to look for phone numbers. Defaults to true.
  pub match_phone_numbers: bool,
}

impl Default for TweetTokenizer {
  /// Port of `TweetTokenizer.__init__` with the default arguments.
  fn default() -> Self {
    Self {
      preserve_case: true,
      reduce_len: false,
      strip_handles: false,
      match_phone_numbers: true,
    }
  }
}

impl TweetTokenizer {
  /// Port of `TweetTokenizer.__init__` with the default arguments.
  pub fn new() -> Self {
    Self::default()
  }

  /// Port of `TweetTokenizer.WORD_RE`, the core tokenizer regex.
  #[allow(non_snake_case)]
  pub fn WORD_RE() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile(&format!("({})", REGEXPS.join("|")), VERBOSE_I))
  }

  /// Port of `TweetTokenizer.PHONE_WORD_RE`, the core tokenizer regex with
  /// phone numbers.
  #[allow(non_snake_case)]
  pub fn PHONE_WORD_RE() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile(&format!("({})", REGEXPS_PHONE.join("|")), VERBOSE_I))
  }

  /// Port of `TweetTokenizer.tokenize`.
  pub fn tokenize(&self, text: &str) -> Result<Vec<String>, TimeoutError> {
    // Fix HTML character entities:
    let mut text = _replace_html_entities(text, &[], true)?;
    // Remove username handles
    if self.strip_handles {
      text = remove_handles(&text)?;
    }
    // Normalize word lengthening
    if self.reduce_len {
      text = reduce_lengthening(&text)?;
    }
    // Shorten problematic sequences of characters
    let safe_text = hang_re()
      .try_replacen(&text, 0, "${1}${1}${1}")
      .map_err(TimeoutError::from)?;
    let re = if self.match_phone_numbers {
      Self::PHONE_WORD_RE()
    } else {
      Self::WORD_RE()
    };
    let mut words = Vec::new();
    for m in re.find_iter(&safe_text) {
      words.push(m.map_err(TimeoutError::from)?.as_str().to_owned());
    }
    // Possibly alter the case, but avoid changing emoticons like :D into :d:
    if !self.preserve_case {
      for w in &mut words {
        if !emoticon_re().is_match(w).map_err(TimeoutError::from)? {
          *w = python::lower(w);
        }
      }
    }
    Ok(words)
  }
}

/// Port of `nltk.tokenize.casual.reduce_lengthening`.
///
/// Replaces repeated character sequences of length 3 or greater with
/// sequences of length 3.
pub fn reduce_lengthening(text: &str) -> Result<String, TimeoutError> {
  static RE: OnceLock<Regex> = OnceLock::new();
  let pattern = RE.get_or_init(|| compile(r"(.)\1{2,}", Flags::default()));
  Ok(
    pattern
      .try_replacen(text, 0, "${1}${1}${1}")
      .map_err(TimeoutError::from)?
      .into_owned(),
  )
}

/// Port of `nltk.tokenize.casual.remove_handles`.
///
/// Substitutes handles with `' '` so that text on either side of removed
/// handles is tokenized correctly.
pub fn remove_handles(text: &str) -> Result<String, TimeoutError> {
  Ok(
    handles_re()
      .try_replacen(text, 0, " ")
      .map_err(TimeoutError::from)?
      .into_owned(),
  )
}

/// Port of `nltk.tokenize.casual.casual_tokenize`.
pub fn casual_tokenize(text: &str, options: TweetTokenizer) -> Result<Vec<String>, TimeoutError> {
  options.tokenize(text)
}
