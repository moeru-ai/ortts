//! The `std::push_heap` / `std::pop_heap` / `std::sort_heap` calls of fastText
//! with `comparePairs` (`l.first > r.first`, a min-heap on the score).
//!
//! Ties between equal scores are ordered by the heap algorithm, so this mirrors
//! the libc++ implementation (`__sift_up`, `__floyd_sift_down`) that the macOS
//! reference wheel is built with.

/// Port of fastText `Predictions` (`std::vector<std::pair<real, int32_t>>`).
pub type Predictions = Vec<(f32, i32)>;

/// Port of fastText `comparePairs`.
#[inline]
fn compare_pairs(l: &(f32, i32), r: &(f32, i32)) -> bool {
  l.0 > r.0
}

/// libc++ `std::__sift_up(first, last, comp, len)` on `heap[..len]`.
fn sift_up(heap: &mut [(f32, i32)], len: usize) {
  if len > 1 {
    let mut last = len - 1;
    let mut l = (len - 2) / 2;
    if compare_pairs(&heap[l], &heap[last]) {
      let t = heap[last];
      loop {
        heap[last] = heap[l];
        last = l;
        if l == 0 {
          break;
        }
        l = (l - 1) / 2;
        if !compare_pairs(&heap[l], &t) {
          break;
        }
      }
      heap[last] = t;
    }
  }
}

/// libc++ `std::__floyd_sift_down(first, comp, len)`; returns the hole index.
fn floyd_sift_down(heap: &mut [(f32, i32)], len: usize) -> usize {
  let mut hole = 0usize;
  let mut child_i = 0usize;
  let mut child = 0usize;
  loop {
    child_i += child + 1;
    child = 2 * child + 1;
    if child + 1 < len && compare_pairs(&heap[child_i], &heap[child_i + 1]) {
      child_i += 1;
      child += 1;
    }
    heap[hole] = heap[child_i];
    hole = child_i;
    if child > (len - 2) / 2 {
      return hole;
    }
  }
}

/// libc++ `std::__pop_heap(first, last, comp, len)` on `heap[..len]`.
fn pop_heap(heap: &mut [(f32, i32)], len: usize) {
  if len > 1 {
    let top = heap[0];
    let mut hole = floyd_sift_down(heap, len);
    let last = len - 1;
    if hole == last {
      heap[hole] = top;
    } else {
      heap[hole] = heap[last];
      hole += 1;
      heap[last] = top;
      sift_up(heap, hole);
    }
  }
}

/// `heap.push_back(x); std::push_heap(...); if (heap.size() > k) { std::pop_heap(...); heap.pop_back(); }`
pub fn push_heap_pop_heap_back(heap: &mut Predictions, x: (f32, i32), k: usize) {
  heap.push(x);
  let len = heap.len();
  sift_up(heap, len);
  if heap.len() > k {
    let len = heap.len();
    pop_heap(heap, len);
    heap.pop();
  }
}

/// libc++ `std::sort_heap(first, last, comp)`.
pub fn sort_heap(heap: &mut Predictions) {
  let mut n = heap.len();
  while n > 1 {
    pop_heap(heap, n);
    n -= 1;
  }
}
