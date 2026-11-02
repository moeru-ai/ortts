//! Port of the parts of `cn2an/conf.py` that the an2cn path uses.
//!
//! Python dicts keyed by `0..=9` become arrays indexed by the digit.

/// Port of `cn2an.conf.NUMBER_LOW_AN2CN`.
pub const NUMBER_LOW_AN2CN: [char; 10] =
  ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

/// Port of `cn2an.conf.NUMBER_UP_AN2CN`.
pub const NUMBER_UP_AN2CN: [char; 10] =
  ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖'];

/// Port of `cn2an.conf.UNIT_LOW_ORDER_AN2CN`.
pub const UNIT_LOW_ORDER_AN2CN: [&str; 16] = [
  "", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十", "百", "千",
];

/// Port of `cn2an.conf.UNIT_UP_ORDER_AN2CN`.
pub const UNIT_UP_ORDER_AN2CN: [&str; 16] = [
  "", "拾", "佰", "仟", "万", "拾", "佰", "仟", "亿", "拾", "佰", "仟", "万", "拾", "佰", "仟",
];
