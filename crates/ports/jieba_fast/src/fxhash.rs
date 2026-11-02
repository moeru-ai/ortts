//! Small FxHash (the rustc hasher) for the dictionary maps. It replaces
//! Python's `dict` hashing; lookups give the same results with any hasher.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
  hash: u64,
}

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
  #[inline]
  fn add(&mut self, word: u64) {
    self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
  }
}

impl Hasher for FxHasher {
  #[inline]
  fn write(&mut self, bytes: &[u8]) {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
      self.add(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    let rest = chunks.remainder();
    if !rest.is_empty() {
      let mut buf = [0u8; 8];
      buf[..rest.len()].copy_from_slice(rest);
      self.add(u64::from_le_bytes(buf));
    }
  }

  #[inline]
  fn write_u8(&mut self, i: u8) {
    self.add(u64::from(i));
  }

  #[inline]
  fn write_u32(&mut self, i: u32) {
    self.add(u64::from(i));
  }

  #[inline]
  fn write_u64(&mut self, i: u64) {
    self.add(i);
  }

  #[inline]
  fn write_usize(&mut self, i: usize) {
    self.add(i as u64);
  }

  #[inline]
  fn finish(&self) -> u64 {
    self.hash
  }
}

pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
