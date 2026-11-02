//! Port of `pypinyin/contrib/uv.py`.

use crate::constants::Style;

/// Port of `pypinyin.contrib.uv.V2UMixin` as used by
/// `pypinyin.converter._v2UConverter` (`V2UMixin` over `DefaultConverter`):
/// use `ü` instead of `v` in the converted pinyin.
#[derive(Clone, Copy, Debug, Default)]
pub struct V2UMixin;

impl V2UMixin {
  /// Port of `V2UMixin.post_convert_style`. The `super()` call resolves to
  /// `DefaultConverter.post_convert_style`, which returns `None`.
  pub fn post_convert_style(
    &self,
    _han: &str,
    _orig_pinyin: &str,
    converted_pinyin: &str,
    _style: Style,
    _strict: bool,
  ) -> Option<String> {
    Some(converted_pinyin.replace('v', "ü"))
  }
}
