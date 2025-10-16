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
    } else {
      Ok(Self { penalty })
    }
  }

  pub fn call(&self, input_ids: ArrayView1<usize>, scores: &Array2<f32>) -> Array2<f32> {
    let mut scores_processed = scores.clone();

    for mut score_row in scores_processed.axis_iter_mut(Axis(0)) {
      for &token_id in input_ids.iter() {
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
