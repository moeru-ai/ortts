//! Port of `split_lang/detect_lang/__init__.py`.

pub mod detector;

pub use detector::{detect_lang_combined, possible_detection_list};
