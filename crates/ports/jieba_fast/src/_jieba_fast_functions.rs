//! Port of the C extension `_jieba_fast_functions_py3`
//! (`jieba_fast/source/jieba_fast_functions_wrap_py3.i`).
//!
//! The C code differs from the pure Python functions it replaces, and the
//! port keeps the C behaviour:
//!
//! - `_get_DAG_and_calc` keeps at most 12 word ends per start position.
//! - Ties in the route pick the first (smallest) end, because the C loop only
//!   replaces the best candidate when it is strictly greater. Python's
//!   `max((prob, x) ...)` picks the largest end.
//! - `_viterbi` breaks ties toward the first previous state and the final
//!   `E` state, and when no candidate beats `MIN_FLOAT` it keeps `MIN_FLOAT`
//!   and takes the larger previous state character.

use crate::char_offsets;
use crate::fxhash::FxHashMap;

/// C `INT_MIN` as a `double`, the start value of `max_freq`.
const INT_MIN: f64 = i32::MIN as f64;

/// Port of `_get_DAG_and_calc(FREQ, sentence, route, total)`. Returns
/// `route`: for each character index the inclusive end index of the best
/// word (`route[N]` is 0, like the C code's sentinel).
pub fn _get_DAG_and_calc(
  FREQ: &FxHashMap<Box<str>, i64>,
  sentence: &str,
  total: f64,
) -> Vec<usize> {
  let offs = char_offsets(sentence);
  let n = offs.len() - 1;
  let mut dag = vec![[0usize; 20]; n];
  let mut points = vec![0usize; n];
  let mut route = vec![(0.0f64, 0usize); n + 1];
  let logtotal = total.ln();

  for k in 0..n {
    let mut i = k;
    while i < n && points[k] < 12 {
      let Some(&t_f) = FREQ.get(&sentence[offs[k]..offs[i + 1]]) else {
        break;
      };
      if t_f != 0 {
        dag[k][points[k]] = i;
        points[k] += 1;
      }
      i += 1;
    }
    if points[k] == 0 {
      dag[k][0] = k;
      points[k] = 1;
    }
  }

  for idx in (0..n).rev() {
    let mut max_freq = INT_MIN;
    let mut max_x = 0;
    for &x in &dag[idx][..points[idx]] {
      let mut fq: i64 = 1;
      if let Some(&o_freq) = FREQ.get(&sentence[offs[idx]..offs[x + 1]]) {
        fq = o_freq;
        if fq == 0 {
          fq = 1;
        }
      }
      let fq_2 = route[x + 1].0;
      let fq_last = (fq as f64).ln() - logtotal + fq_2;
      if fq_last > max_freq {
        max_freq = fq_last;
        max_x = x;
      }
    }
    route[idx] = (max_freq, max_x);
  }
  route.into_iter().map(|(_, x)| x).collect()
}

/// `#define MIN_FLOAT -3.14e100`.
pub const MIN_FLOAT: f64 = -3.14e100;

/// HMM tables for `_viterbi`, indexed by the position of the state in the
/// `states` string that `finalseg` passes (`"BMES"`).
pub struct ViterbiModel {
  /// `start_p[state]`.
  pub start_p: [f64; 4],
  /// `trans_p[from][to]`, `None` when the key is missing.
  pub trans_p: [[Option<f64>; 4]; 4],
  /// `emip_p[state][char]`.
  pub emit_p: [FxHashMap<char, f64>; 4],
}

/// The `states` string `"BMES"` as indices.
pub const STATES: [char; 4] = ['B', 'M', 'E', 'S'];

fn state_index(c: char) -> usize {
  match c {
    'B' => 0,
    'M' => 1,
    'E' => 2,
    'S' => 3,
    _ => unreachable!(),
  }
}

/// `PrevStatus_str`.
fn prev_status(y: char) -> [char; 2] {
  match y {
    'B' => ['E', 'S'],
    'M' => ['M', 'B'],
    'S' => ['S', 'E'],
    'E' => ['B', 'M'],
    _ => unreachable!(),
  }
}

/// Port of `_viterbi(obs, 'BMES', start_p, trans_p, emip_p)`. Returns
/// `(prob, path)` with one state character per observation.
pub fn _viterbi(obs: &[char], model: &ViterbiModel) -> (f64, Vec<char>) {
  let obs_len = obs.len();
  let mut v = vec![[0.0f64; 4]; obs_len];
  let mut path = vec![['\0'; 4]; obs_len];

  for (i, &s) in STATES.iter().enumerate() {
    let mut t_double = MIN_FLOAT;
    if let Some(&item) = model.emit_p[i].get(&obs[0]) {
      t_double = item;
    }
    let t_double_2 = model.start_p[i];
    v[0][i] = t_double + t_double_2;
    path[0][i] = s;
  }
  for i in 1..obs_len {
    let t_obs = obs[i];
    for (j, &y) in STATES.iter().enumerate() {
      let mut em_p = MIN_FLOAT;
      if let Some(&item) = model.emit_p[j].get(&t_obs) {
        em_p = item;
      }
      let mut max_prob = MIN_FLOAT;
      let mut best_state = '\0';
      let prev = prev_status(y);
      for &y0 in &prev {
        let mut prob = em_p;
        prob += v[i - 1][state_index(y0)];
        match model.trans_p[state_index(y0)][j] {
          None => prob += MIN_FLOAT,
          Some(t) => prob += t,
        }
        if prob > max_prob {
          max_prob = prob;
          best_state = y0;
        }
      }
      if best_state == '\0' {
        for &y0 in &prev {
          if y0 > best_state {
            best_state = y0;
          }
        }
      }
      v[i][j] = max_prob;
      path[i][j] = best_state;
    }
  }

  let last = &v[obs_len - 1];
  let mut max_prob = last[state_index('E')];
  let mut best_state = 'E';
  if last[state_index('S')] > max_prob {
    max_prob = last[state_index('S')];
    best_state = 'S';
  }

  let mut t_list = vec!['\0'; obs_len];
  let mut now_state = best_state;
  for i in (0..obs_len).rev() {
    t_list[i] = now_state;
    now_state = path[i][state_index(now_state)];
  }
  (max_prob, t_list)
}
