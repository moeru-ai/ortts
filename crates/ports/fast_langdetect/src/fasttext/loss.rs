//! Port of fastText `src/loss.{h,cc}`: `HierarchicalSoftmaxLoss::predict`.

use super::densematrix::DenseMatrix;
use super::heap::{Predictions, push_heap_pop_heap_back, sort_heap};

/// Port of `Loss::std_log`: `std::log(x + 1e-5)`, where `x + 1e-5` is a
/// `double` and the result is narrowed back to `real` (`f32`).
pub fn std_log(x: f32) -> f32 {
  (f64::from(x) + 1e-5).ln() as f32
}

/// Port of `HierarchicalSoftmaxLoss::Node`.
#[derive(Debug, Clone, Copy)]
pub struct Node {
  /// `Node::parent`.
  pub parent: i32,
  /// `Node::left`.
  pub left: i32,
  /// `Node::right`.
  pub right: i32,
  /// `Node::count`.
  pub count: i64,
  /// `Node::binary`.
  pub binary: bool,
}

/// Port of fastText `class HierarchicalSoftmaxLoss`.
#[derive(Debug, Clone)]
pub struct HierarchicalSoftmaxLoss {
  tree: Vec<Node>,
  osz: i32,
}

impl HierarchicalSoftmaxLoss {
  /// Port of `HierarchicalSoftmaxLoss::HierarchicalSoftmaxLoss` (`buildTree`):
  /// the Huffman tree over the label counts.
  pub fn new(counts: &[i64]) -> Self {
    let osz = counts.len() as i32;
    let n = (2 * osz - 1).max(0) as usize;
    let mut tree = vec![
      Node {
        parent: -1,
        left: -1,
        right: -1,
        count: 1_000_000_000_000_000, // 1e15
        binary: false,
      };
      n
    ];
    for (node, &c) in tree.iter_mut().zip(counts) {
      node.count = c;
    }
    let mut leaf = osz - 1;
    let mut node = osz;
    for i in osz..2 * osz - 1 {
      let mut mini = [0i32; 2];
      for m in &mut mini {
        if leaf >= 0 && tree[leaf as usize].count < tree[node as usize].count {
          *m = leaf;
          leaf -= 1;
        } else {
          *m = node;
          node += 1;
        }
      }
      let iu = i as usize;
      tree[iu].left = mini[0];
      tree[iu].right = mini[1];
      tree[iu].count = tree[mini[0] as usize].count + tree[mini[1] as usize].count;
      tree[mini[0] as usize].parent = i;
      tree[mini[1] as usize].parent = i;
      tree[mini[1] as usize].binary = true;
    }
    Self { tree, osz }
  }

  /// Port of `HierarchicalSoftmaxLoss::predict`.
  pub fn predict(
    &self,
    k: i32,
    threshold: f32,
    heap: &mut Predictions,
    hidden: &[f32],
    wo: &DenseMatrix,
  ) {
    self.dfs(k, threshold, 2 * self.osz - 2, 0.0, heap, hidden, wo);
    sort_heap(heap);
  }

  /// Port of `HierarchicalSoftmaxLoss::dfs`.
  #[allow(clippy::too_many_arguments)]
  fn dfs(
    &self,
    k: i32,
    threshold: f32,
    node: i32,
    score: f32,
    heap: &mut Predictions,
    hidden: &[f32],
    wo: &DenseMatrix,
  ) {
    if score < std_log(threshold) {
      return;
    }
    if heap.len() == k as usize && score < heap[0].0 {
      return;
    }
    let n = self.tree[node as usize];
    if n.left == -1 && n.right == -1 {
      push_heap_pop_heap_back(heap, (score, node), k as usize);
      return;
    }
    let mut f = wo.dot_row(hidden, i64::from(node - self.osz));
    // `f = 1. / (1 + std::exp(-f));`: `std::exp(float)`, then a `double` division.
    f = (1.0f64 / f64::from(1.0f32 + (-f).exp())) as f32;
    self.dfs(
      k,
      threshold,
      n.left,
      score + std_log((1.0f64 - f64::from(f)) as f32),
      heap,
      hidden,
      wo,
    );
    self.dfs(k, threshold, n.right, score + std_log(f), heap, hidden, wo);
  }
}
