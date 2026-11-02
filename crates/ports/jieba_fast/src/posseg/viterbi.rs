//! Port of `jieba_fast/posseg/viterbi.py`.

use super::Model;

/// Port of `MIN_FLOAT`.
pub const MIN_FLOAT: f64 = -3.14e100;
/// Port of `MIN_INF = float("-inf")`.
pub const MIN_INF: f64 = f64::NEG_INFINITY;

/// `(prob, state) > (best_prob, best_state)` with Python tuple comparison,
/// where states compare by their `(position, tag)` tuple (`Model::rank`).
#[inline]
fn tuple_gt(model: &Model, a: (f64, u16), b: (f64, u16)) -> bool {
  a.0 > b.0 || (a.0 == b.0 && model.rank[a.1 as usize] > model.rank[b.1 as usize])
}

/// Port of `viterbi(obs, states, start_p, trans_p, emit_p)` with the posseg
/// tables in `model`. Returns `(prob, route)` with state indices into
/// `model.states`.
///
/// The Python sets and dicts are iterated in hash order, but only through
/// `max()` over `(prob, state)` tuples, whose result does not depend on the
/// order, so the port iterates in index order.
pub fn viterbi(obs: &[char], model: &Model) -> (f64, Vec<u16>) {
  let n_states = model.states.len();
  let all_states: Vec<u16> = (0..n_states as u16).collect();
  let states_of = |c: &char| -> &[u16] {
    model
      .char_state_tab
      .get(c)
      .map_or(&all_states[..], |v| &v[..])
  };

  // V[t][y] (dense) and the keys of mem_path[t] (the states present at t).
  let mut v: Vec<Vec<f64>> = Vec::with_capacity(obs.len());
  let mut mem_path: Vec<Vec<u16>> = Vec::with_capacity(obs.len());
  let mut keys: Vec<Vec<u16>> = Vec::with_capacity(obs.len());

  let mut v0 = vec![0.0; n_states];
  let mut k0 = Vec::new();
  for &y in states_of(&obs[0]) {
    v0[y as usize] = model.start_p[y as usize]
      + model.emit_p[y as usize]
        .get(&obs[0])
        .copied()
        .unwrap_or(MIN_FLOAT);
    k0.push(y);
  }
  v.push(v0);
  mem_path.push(vec![u16::MAX; n_states]);
  keys.push(k0);

  let mut expect = vec![false; n_states];
  for t in 1..obs.len() {
    let prev_states: Vec<u16> = keys[t - 1]
      .iter()
      .copied()
      .filter(|&x| !model.trans_keys[x as usize].is_empty())
      .collect();

    expect.iter_mut().for_each(|e| *e = false);
    let mut any_expect = false;
    for &x in &prev_states {
      for &y in &model.trans_keys[x as usize] {
        expect[y as usize] = true;
        any_expect = true;
      }
    }
    let mut obs_states: Vec<u16> = states_of(&obs[t])
      .iter()
      .copied()
      .filter(|&y| expect[y as usize])
      .collect();
    if obs_states.is_empty() {
      obs_states = if any_expect {
        (0..n_states as u16)
          .filter(|&y| expect[y as usize])
          .collect()
      } else {
        all_states.clone()
      };
    }

    let mut vt = vec![0.0; n_states];
    let mut mt = vec![u16::MAX; n_states];
    let prev_v = &v[t - 1];
    for &y in &obs_states {
      let em = model.emit_p[y as usize]
        .get(&obs[t])
        .copied()
        .unwrap_or(MIN_FLOAT);
      let mut best: Option<(f64, u16)> = None;
      for &y0 in &prev_states {
        let tr = model.trans_p[y0 as usize * n_states + y as usize];
        let cand = (prev_v[y0 as usize] + tr + em, y0);
        if best.is_none_or(|b| tuple_gt(model, cand, b)) {
          best = Some(cand);
        }
      }
      let (prob, state) = best.expect("ValueError: max() arg is an empty sequence");
      vt[y as usize] = prob;
      mt[y as usize] = state;
    }
    v.push(vt);
    mem_path.push(mt);
    keys.push(obs_states);
  }

  let last_t = obs.len() - 1;
  let mut best: Option<(f64, u16)> = None;
  for &y in &keys[last_t] {
    let cand = (v[last_t][y as usize], y);
    if best.is_none_or(|b| tuple_gt(model, cand, b)) {
      best = Some(cand);
    }
  }
  let (prob, mut state) = best.expect("ValueError: max() arg is an empty sequence");

  let mut route = vec![0u16; obs.len()];
  for i in (0..obs.len()).rev() {
    route[i] = state;
    state = mem_path[i][state as usize];
  }
  (prob, route)
}
