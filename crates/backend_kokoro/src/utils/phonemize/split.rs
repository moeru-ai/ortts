use regex::Regex;

/// Helper function to split a string on a regex, but keep the delimiters.
pub fn split(text: &str, regex: &Regex) -> Vec<(bool, String)> {
  let mut result = Vec::new();
  let mut prev = 0;

  for mat in regex.find_iter(text) {
    let full_match = mat.as_str();
    let match_start = mat.start();
    let match_end = mat.end();

    if prev < match_start {
      result.push((false, text[prev..match_start].to_string()));
    }

    if !full_match.is_empty() {
      result.push((true, full_match.to_string()));
    }

    prev = match_end;
  }

  if prev < text.len() {
    result.push((false, text[prev..].to_string()));
  }

  result
}
