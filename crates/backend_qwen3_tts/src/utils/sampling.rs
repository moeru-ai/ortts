use rand::{Rng, rngs::StdRng};

#[derive(Clone, Copy, Debug)]
pub struct SamplingOptions {
  pub do_sample: bool,
  pub top_k: usize,
  pub top_p: f64,
  pub temperature: f64,
}

impl SamplingOptions {
  pub const fn main() -> Self {
    Self {
      do_sample: true,
      top_k: 50,
      top_p: 1.0,
      temperature: 0.9,
    }
  }
}

pub fn apply_repetition_penalty(logits: &mut [f32], previous: &[usize], penalty: f32) {
  if (penalty - 1.0).abs() < f32::EPSILON {
    return;
  }
  let mut seen = std::collections::HashSet::new();
  for &token in previous {
    if !seen.insert(token) {
      continue;
    }
    if let Some(score) = logits.get_mut(token) {
      *score = if score.is_sign_negative() {
        *score * penalty
      } else {
        *score / penalty
      };
    }
  }
}

pub fn sample(logits: &[f32], options: SamplingOptions, rng: &mut StdRng) -> usize {
  if !options.do_sample || options.temperature <= 0.0 {
    return argmax(logits);
  }

  let mut candidates: Vec<(usize, f64)> = logits
    .iter()
    .enumerate()
    .filter_map(|(index, &score)| {
      let score = f64::from(score) / options.temperature.max(1e-6);
      score.is_finite().then_some((index, score))
    })
    .collect();
  candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
  if options.top_k > 0 && candidates.len() > options.top_k {
    candidates.truncate(options.top_k);
  }
  if candidates.is_empty() {
    return argmax(logits);
  }

  let max = candidates[0].1;
  let mut weights: Vec<f64> = candidates
    .iter()
    .map(|(_, score)| (*score - max).exp())
    .collect();
  let total: f64 = weights.iter().sum();
  for weight in &mut weights {
    *weight /= total;
  }

  if options.top_p < 1.0 {
    let mut cumulative = 0.0;
    let mut keep = weights.len();
    for (index, probability) in weights.iter().enumerate() {
      cumulative += probability;
      if cumulative >= options.top_p {
        keep = index + 1;
        break;
      }
    }
    candidates.truncate(keep);
    weights.truncate(keep);
  }

  let total: f64 = weights.iter().sum();
  let mut target = rng.random::<f64>() * total;
  for ((token, _), weight) in candidates.iter().zip(weights.iter()) {
    if target <= *weight {
      return *token;
    }
    target -= weight;
  }
  candidates.last().map_or(0, |(token, _)| *token)
}

fn argmax(logits: &[f32]) -> usize {
  logits
    .iter()
    .enumerate()
    .max_by(|(_, left), (_, right)| left.total_cmp(right))
    .map_or(0, |(index, _)| index)
}

#[cfg(test)]
mod tests {
  use rand::{SeedableRng, rngs::StdRng};

  use super::{SamplingOptions, apply_repetition_penalty, sample};

  #[test]
  fn repetition_penalty_matches_transformers_rule() {
    let mut logits = vec![4.0, -4.0, 1.0];
    apply_repetition_penalty(&mut logits, &[0, 1, 0], 2.0);
    assert_eq!(logits, vec![2.0, -8.0, 1.0]);
  }

  #[test]
  fn greedy_sampling_returns_the_largest_logit() {
    let mut rng = StdRng::seed_from_u64(0);
    let options = SamplingOptions {
      do_sample: false,
      ..SamplingOptions::main()
    };
    assert_eq!(sample(&[0.0, 3.0, 2.0], options, &mut rng), 1);
  }
}
