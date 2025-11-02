pub fn hiragana_normalize(text: &str) -> String {
  kakasi::convert(text).hiragana
}
