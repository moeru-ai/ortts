//! Subset port of the Python ToJyutping 3.2.0 package used by GPT-SoVITS.
//!
//! The crate root mirrors `ToJyutping/__init__.py`: it re-exports the bound
//! methods of the default `ToJyutping` converter as free functions.

pub mod jyutping;
pub mod to_jyutping;
pub mod trie;
pub mod utils;

pub use to_jyutping::{JyutpingConverter, ToJyutping};

/// Port of `ToJyutping.get_jyutping_list` (`ToJyutping.get_jyutping_list`).
pub fn get_jyutping_list(s: &str) -> Vec<(String, Option<String>)> {
  ToJyutping.get_jyutping_list(s)
}

/// Port of `ToJyutping.get_jyutping` (`ToJyutping.get_jyutping`).
pub fn get_jyutping(s: &str) -> String {
  ToJyutping.get_jyutping(s)
}

/// Port of `ToJyutping.get_jyutping_text` (`ToJyutping.get_jyutping_text`).
pub fn get_jyutping_text(s: &str) -> String {
  ToJyutping.get_jyutping_text(s)
}
