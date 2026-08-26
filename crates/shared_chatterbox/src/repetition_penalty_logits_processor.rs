use std::collections::HashSet;

use ndarray::{Array2, ArrayView1, Axis};
use ortts_shared::AppError;

pub struct RepetitionPenaltyLogitsProcessor {
  penalty: f32,
}

impl RepetitionPenaltyLogitsProcessor {
  pub fn new(penalty: f32) -> Result<Self, AppError> {
    if penalty <= 0.0 {
      return Err(AppError::anyhow(&anyhow::anyhow!(format!(
        "`penalty` must be a strictly positive float, but is {penalty}"
      ))));
    }

    Ok(Self { penalty })
  }

  #[must_use]
  pub fn call(&self, input_ids: ArrayView1<usize>, scores: &Array2<f32>) -> Array2<f32> {
    let mut scores_processed = scores.clone();
    let unique_token_ids = input_ids.iter().copied().collect::<HashSet<_>>();

    for mut score_row in scores_processed.axis_iter_mut(Axis(0)) {
      for &token_id in &unique_token_ids {
        let vocab_size = score_row.len();
        if token_id < vocab_size {
          let score_ref = &mut score_row[token_id];
          if *score_ref < 0.0 {
            *score_ref *= self.penalty;
          } else {
            *score_ref /= self.penalty;
          }
        }
      }
    }

    scores_processed
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::RepetitionPenaltyLogitsProcessor;

  #[test]
  fn applies_penalty_once_to_duplicate_tokens() {
    let processor = RepetitionPenaltyLogitsProcessor::new(1.2).unwrap();
    let input_ids = array![1, 1, 2, 2];
    let scores = array![[0.0, 12.0, -12.0]];

    let processed = processor.call(input_ids.view(), &scores);

    assert!((processed[[0, 1]] - 10.0).abs() < 1e-5);
    assert!((processed[[0, 2]] - -14.4).abs() < 1e-5);
  }
}
