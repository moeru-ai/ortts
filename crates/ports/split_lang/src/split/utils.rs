//! Port of `split_lang/split/utils.py`.

/// Port of `PUNCTUATION`.
pub const PUNCTUATION: &str =
  r#"〜~,.;:!?，。！？；：、·([{<（【《〈「『“‘)]}>）】》〉」』”’"-_——#$%&……￥'*+<=>?@[\]^_`{|}~"#;

/// `chinese_char_pattern`: `[\u4e00-\u9fff]`.
fn is_chinese_char(c: char) -> bool {
  ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// `hangul_pattern`: `[\uac00-\ud7af]`.
fn is_hangul(c: char) -> bool {
  ('\u{ac00}'..='\u{d7af}').contains(&c)
}

/// `hiragana_katakana_pattern`: `[\u3040-\u30ff々]`.
fn is_hiragana_katakana(c: char) -> bool {
  ('\u{3040}'..='\u{30ff}').contains(&c) || c == '々'
}

/// `zh_ja_pattern`: `[\u4e00-\u9fff\u3040-\u30ff々]`.
fn is_zh_ja(c: char) -> bool {
  is_chinese_char(c) || is_hiragana_katakana(c)
}

/// Port of `contains_chinese_char(text)`.
pub fn contains_chinese_char(text: &str) -> bool {
  text.chars().any(is_chinese_char)
}

/// Port of `contains_hangul(text)`.
pub fn contains_hangul(text: &str) -> bool {
  text.chars().any(is_hangul)
}

/// Port of `contains_ja_kana(text)`.
pub fn contains_ja_kana(text: &str) -> bool {
  text.chars().any(is_hiragana_katakana)
}

/// Port of `contains_zh_ja(text)`.
pub fn contains_zh_ja(text: &str) -> bool {
  text.chars().any(is_zh_ja)
}

/// Port of `contains_only_kana(text)` (`pattern.match(char)` for every char;
/// true for the empty string).
pub fn contains_only_kana(text: &str) -> bool {
  text.chars().all(is_hiragana_katakana)
}
