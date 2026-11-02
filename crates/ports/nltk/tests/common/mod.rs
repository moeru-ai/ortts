#![allow(dead_code)]

use std::path::PathBuf;

use serde::de::DeserializeOwned;

pub fn fixture_path(name: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("fixtures")
    .join(name)
}

pub fn read_json<T: DeserializeOwned>(name: &str) -> T {
  let text = std::fs::read_to_string(fixture_path(name)).expect("fixture exists");
  serde_json::from_str(&text).expect("fixture is valid json")
}

pub fn read_jsonl<T: DeserializeOwned>(name: &str) -> Vec<T> {
  let text = std::fs::read_to_string(fixture_path(name)).expect("fixture exists");
  text
    .lines()
    .filter(|l| !l.is_empty())
    .map(|l| serde_json::from_str(l).expect("fixture line is valid json"))
    .collect()
}

/// Every Unicode scalar value, in order.
pub fn all_chars() -> impl Iterator<Item = char> {
  (0u32..0x110000).filter_map(char::from_u32)
}

pub fn in_ranges(ranges: &[(u32, u32)], c: u32) -> bool {
  ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi)
}
