//! Port of `split_lang/split/__init__.py`.

pub mod splitter;
pub mod utils;

pub use splitter::LangSplitter;
pub use utils::{PUNCTUATION, contains_hangul, contains_ja_kana, contains_zh_ja};
