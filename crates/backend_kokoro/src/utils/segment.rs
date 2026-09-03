use ortts_shared::AppError;

use super::{Tokenizer, phonemize};
use crate::utils::sentence::split_sentences;

pub const MAX_PHONEME_TOKENS: usize = 510;

pub async fn prepare_segments(
  text: &str,
  tokenizer: &Tokenizer,
) -> Result<Vec<Vec<i64>>, AppError> {
  let mut segments = Vec::new();
  for sentence in split_sentences(text) {
    let tokens = tokenize(&sentence, tokenizer).await?;
    if tokens.is_empty() {
      continue;
    }
    if tokens.len() <= MAX_PHONEME_TOKENS {
      segments.push(with_padding(&tokens));
      continue;
    }

    segments.extend(split_long_sentence(&sentence, tokens, tokenizer).await?);
  }
  Ok(segments)
}

async fn split_long_sentence(
  sentence: &str,
  sentence_tokens: Vec<i64>,
  tokenizer: &Tokenizer,
) -> Result<Vec<Vec<i64>>, AppError> {
  let words: Vec<_> = sentence.split_whitespace().collect();
  if words.len() <= 1 {
    return Ok(split_token_ids(&sentence_tokens));
  }

  let mut segments = Vec::new();
  let mut start = 0;
  while start < words.len() {
    let fitting_end = find_fitting_end(&words, start, tokenizer).await?;
    if fitting_end == start {
      segments.extend(split_token_ids(&tokenize(words[start], tokenizer).await?));
      start += 1;
      continue;
    }

    let boundary = preferred_boundary(&words[start..fitting_end]);
    let end = start + boundary;
    segments.extend(split_token_ids(
      &tokenize(&words[start..end].join(" "), tokenizer).await?,
    ));
    start = end;
  }
  Ok(segments)
}

async fn find_fitting_end(
  words: &[&str],
  start: usize,
  tokenizer: &Tokenizer,
) -> Result<usize, AppError> {
  let mut low = start;
  let mut high = words.len();
  while low < high {
    let middle = low + (high - low).div_ceil(2);
    let candidate = words[start..middle].join(" ");
    if tokenize(&candidate, tokenizer).await?.len() <= MAX_PHONEME_TOKENS {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  Ok(low)
}

async fn tokenize(text: &str, tokenizer: &Tokenizer) -> Result<Vec<i64>, AppError> {
  Ok(tokenizer.encode(&phonemize(text.to_owned(), true).await?))
}

fn with_padding(tokens: &[i64]) -> Vec<i64> {
  let mut padded = Vec::with_capacity(tokens.len() + 2);
  padded.push(0);
  padded.extend_from_slice(tokens);
  padded.push(0);
  padded
}

fn split_token_ids(tokens: &[i64]) -> Vec<Vec<i64>> {
  tokens
    .chunks(MAX_PHONEME_TOKENS)
    .map(with_padding)
    .collect()
}

fn preferred_boundary(words: &[&str]) -> usize {
  words
    .iter()
    .rposition(|word| word.chars().last().is_some_and(|c| ":;,—".contains(c)))
    .map_or(words.len(), |index| index + 1)
}

#[cfg(test)]
mod tests {
  use super::{MAX_PHONEME_TOKENS, preferred_boundary, prepare_segments, split_token_ids};
  use crate::utils::Tokenizer;

  #[test]
  fn split_token_ids_preserves_every_token_with_padding() {
    let tokens: Vec<_> = (1..=MAX_PHONEME_TOKENS * 2 + 1)
      .map(|token| token as i64)
      .collect();

    let chunks = split_token_ids(&tokens);

    assert_eq!(chunks.len(), 3);
    assert_eq!(
      chunks
        .iter()
        .flat_map(|chunk| chunk[1..chunk.len() - 1].iter().copied())
        .collect::<Vec<_>>(),
      tokens
    );
    assert!(
      chunks
        .iter()
        .all(|chunk| chunk.len() <= MAX_PHONEME_TOKENS + 2)
    );
  }

  #[test]
  fn preferred_boundary_uses_the_last_natural_break() {
    assert_eq!(preferred_boundary(&["First:", "second,", "third"]), 2);
    assert_eq!(preferred_boundary(&["first", "second"]), 2);
  }

  #[tokio::test]
  async fn prepare_segments_bounds_a_long_sentence_by_tokens() {
    let tokenizer = Tokenizer::new().await.unwrap();
    let text = (0..160).map(|_| "hello").collect::<Vec<_>>().join(" ");

    let segments = prepare_segments(&text, &tokenizer).await.unwrap();

    assert!(segments.len() > 1);
    assert!(
      segments
        .iter()
        .all(|segment| segment.len() <= MAX_PHONEME_TOKENS + 2)
    );
  }
}
