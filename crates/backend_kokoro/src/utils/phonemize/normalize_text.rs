use regex::{Captures, Regex};
use std::sync::OnceLock;

use super::{flip_money, point_num, split_num};

static REPLACEMENTS: OnceLock<Vec<(Regex, Box<dyn Fn(&str) -> String + Send + Sync>)>> =
  OnceLock::new();

fn init_replacements() -> Vec<(Regex, Box<dyn Fn(&str) -> String + Send + Sync>)> {
  vec![
    // 1. Handle quotes and brackets
    (Regex::new(r"[‘’]").unwrap(), Box::new(|_| "'".to_string())),
    (Regex::new(r"«").unwrap(), Box::new(|_| "“".to_string())),
    (Regex::new(r"»").unwrap(), Box::new(|_| "”".to_string())),
    (Regex::new(r"[“”]").unwrap(), Box::new(|_| "\"".to_string())),
    (Regex::new(r"\(").unwrap(), Box::new(|_| "«".to_string())),
    (Regex::new(r"\)").unwrap(), Box::new(|_| "»".to_string())),
    // 2. Replace uncommon punctuation marks
    (
      Regex::new(r"[、，]").unwrap(),
      Box::new(|_| ", ".to_string()),
    ),
    (Regex::new(r"。").unwrap(), Box::new(|_| ". ".to_string())),
    (Regex::new(r"！").unwrap(), Box::new(|_| "! ".to_string())),
    (Regex::new(r"：").unwrap(), Box::new(|_| ": ".to_string())),
    (Regex::new(r"；").unwrap(), Box::new(|_| "; ".to_string())),
    (Regex::new(r"？").unwrap(), Box::new(|_| "? ".to_string())),
    // 3. Whitespace normalization
    (
      Regex::new(r"[^\S \n]").unwrap(),
      Box::new(|_| " ".to_string()),
    ),
    (Regex::new(r"  +").unwrap(), Box::new(|_| " ".to_string())),
    (
      Regex::new(r"\n +\n").unwrap(),
      Box::new(|_| "\n\n".to_string()),
    ),
    // 4. Abbreviations
    (
      Regex::new(r"\bD[Rr]\.(?= [A-Z])").unwrap(),
      Box::new(|_| "Doctor".to_string()),
    ),
    (
      Regex::new(r"\b(?:Mr\.|MR\.(?= [A-Z]))").unwrap(),
      Box::new(|_| "Mister".to_string()),
    ),
    (
      Regex::new(r"\b(?:Ms\.|MS\.(?= [A-Z]))").unwrap(),
      Box::new(|_| "Miss".to_string()),
    ),
    (
      Regex::new(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))").unwrap(),
      Box::new(|_| "Mrs".to_string()),
    ),
    (
      Regex::new(r"(?i)\betc\.(?! [A-Z])").unwrap(),
      Box::new(|_| "etc".to_string()),
    ),
    // 5. Normalize casual words
    (
      Regex::new(r"(?i)\b(y)eah?\b").unwrap(),
      Box::new(|m| format!("{}e'a", &m[1..2])),
    ),
    // 5. Handle numbers and currencies
    (
      Regex::new(r"\d*\.\d+|\b\d{4}s?\b|(?::)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?::)").unwrap(),
      Box::new(|m| split_num(&m)),
    ),
    (
      Regex::new(r"(\d),(\d)").unwrap(),
      Box::new(|m| format!("{}{}", &m[1..2], &m[3..4])), // 捕获并连接数字
    ),
    (
      Regex::new(
        r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b",
      )
      .unwrap(),
      Box::new(|m| flip_money(&m)),
    ),
    (
      Regex::new(r"\d*\.\d+").unwrap(),
      Box::new(|m| point_num(&m)),
    ),
    (
      Regex::new(r"(\d)-(\d)").unwrap(),
      Box::new(|m| format!("{0} to {1}", &m[1..2], &m[3..4])),
    ),
    (
      Regex::new(r"(\d)S").unwrap(),
      Box::new(|m| format!("{0} S", &m[1..2])),
    ),
    // 6. Handle possessives
    (
      Regex::new(r"([BCDFGHJ-NP-TV-Z])'?s\b").unwrap(),
      Box::new(|m| format!("{}'S", &m[1..2])),
    ),
    (
      Regex::new(r"(X')S\b").unwrap(),
      Box::new(|m| format!("{}s", &m[1..2])),
    ),
    // 7. Handle hyphenated words/letters
    (
      Regex::new(r"([A-Za-z]\.){2,} [a-z]").unwrap(),
      Box::new(|m| m.replace('.', "-")),
    ),
    (
      Regex::new(r"(?i)([A-Z])\.([A-Z])").unwrap(),
      Box::new(|m| format!("{}-{}", &m[1..2], &m[3..4])),
    ),
  ]
}

/// Normalize text for phonemization
pub fn normalize_text(text: &str) -> String {
  let replacements = REPLACEMENTS.get_or_init(init_replacements);

  let mut normalized = text.to_string();

  for (regex, replacer) in replacements.iter() {
    normalized = regex
      .replace_all(&normalized, |caps: &Captures| {
        let whole_match = caps.get(0).map_or("", |m| m.as_str());
        replacer(whole_match)
      })
      .to_string();
  }

  normalized.trim().to_string()
}
