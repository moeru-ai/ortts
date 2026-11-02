//! Port of `GPT_SoVITS/text/zh_normalization/phonecode.py` (GPT-SoVITS @ 04a44eb).
//!
//! 规范化固话/手机号码
//! 手机 <http://www.jihaoba.com/news/show/13680>
//! 移动：139、138、137、136、135、134、159、158、157、150、151、152、188、187、182、183、184、178、198
//! 联通：130、131、132、156、155、186、185、176
//! 电信：133、153、189、180、181、177

use std::sync::LazyLock;

use super::Error;
use super::num::verbalize_digit;
use super::pyre::{PyMatch, PyRegex, py_split_whitespace};

/// Port of `RE_MOBILE_PHONE`.
pub static RE_MOBILE_PHONE: LazyLock<PyRegex> = LazyLock::new(|| {
  PyRegex::compile(r"(?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d)")
});

/// Port of `RE_TELEPHONE`.
pub static RE_TELEPHONE: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(r"(?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})(?!\d)"));

/// Port of `RE_NATIONAL_UNIFORM_NUMBER` (全国统一的号码400开头).
pub static RE_NATIONAL_UNIFORM_NUMBER: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(r"(400)(-)?\d{3}(-)?\d{4}"));

/// Port of `phone2str`.
pub fn phone2str(phone_string: &str, mobile: bool) -> Result<String, Error> {
  let parts: Vec<&str> = if mobile {
    py_split_whitespace(phone_string.trim_matches('+'))
  } else {
    phone_string.split('-').collect()
  };
  let verbalized = parts
    .iter()
    .map(|part| verbalize_digit(part, true))
    .collect::<Result<Vec<_>, _>>()?;
  Ok(verbalized.join("，"))
}

/// Port of `replace_phone`.
pub fn replace_phone(m: &PyMatch) -> Result<String, Error> {
  phone2str(m.group0(), false)
}

/// Port of `replace_mobile`.
pub fn replace_mobile(m: &PyMatch) -> Result<String, Error> {
  phone2str(m.group0(), true)
}
