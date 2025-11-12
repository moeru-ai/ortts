use unicode_normalization::UnicodeNormalization;

pub fn hiragana_normalize(text: &str) -> String {
  let text = kakasi::convert(text).hiragana;

  text.nfkd().collect::<String>()
}
