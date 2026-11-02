//! Port of `pypinyin/compat.py`.

/// Port of `pypinyin.compat.SUPPORT_UCS4` (`len('\U00020000') == 1`).
///
/// Always `true`: Rust strings hold full Unicode scalar values, like the
/// Python 3 builds that GPT-SoVITS runs on.
pub const SUPPORT_UCS4: bool = true;
