const SENTENCE_ENDS: &str = ".!?…。？！\n";
const TRAILING_CHARS: &str = "\"')]}」』”";

pub fn split_sentences(text: &str) -> Vec<String> {
  let text = text.trim();
  if text.is_empty() {
    return Vec::new();
  }

  let mut sentences = Vec::new();
  let mut start = 0;
  let mut characters = text.char_indices().peekable();

  while let Some((index, character)) = characters.next() {
    if !SENTENCE_ENDS.contains(character) {
      continue;
    }

    let mut end = index + character.len_utf8();
    while let Some(&(next_index, next_character)) = characters.peek()
      && SENTENCE_ENDS.contains(next_character)
      && next_character != '\n'
    {
      characters.next();
      end = next_index + next_character.len_utf8();
    }
    while let Some(&(next_index, next_character)) = characters.peek()
      && TRAILING_CHARS.contains(next_character)
    {
      characters.next();
      end = next_index + next_character.len_utf8();
    }

    if character.is_ascii()
      && let Some(next_character) = text[end..].chars().next()
      && next_character.is_ascii()
      && !next_character.is_whitespace()
    {
      continue;
    }

    if let Some(sentence) = text.get(start..end).map(str::trim)
      && !sentence.is_empty()
    {
      sentences.push(sentence.to_owned());
    }
    start = end;
  }

  if let Some(sentence) = text.get(start..).map(str::trim)
    && !sentence.is_empty()
  {
    sentences.push(sentence.to_owned());
  }

  sentences
}

#[cfg(test)]
mod tests {
  use super::split_sentences;

  #[test]
  fn splits_sentence_punctuation_and_keeps_closing_quotes() {
    assert_eq!(
      split_sentences("First sentence. Second sentence!"),
      vec![
        String::from("First sentence."),
        String::from("Second sentence!")
      ]
    );
    assert_eq!(
      split_sentences(r#"She said, "Wait?!" Then left."#),
      vec![
        String::from(r#"She said, "Wait?!""#),
        String::from("Then left.")
      ]
    );
  }

  #[test]
  fn ignores_inline_ascii_punctuation() {
    assert_eq!(
      split_sentences("Hello!World. Visit https://example.com/a. Done."),
      vec![
        String::from("Hello!World."),
        String::from("Visit https://example.com/a."),
        String::from("Done."),
      ]
    );
  }

  #[test]
  fn empty_text_has_no_sentences() {
    assert!(split_sentences(" \n ").is_empty());
  }
}
