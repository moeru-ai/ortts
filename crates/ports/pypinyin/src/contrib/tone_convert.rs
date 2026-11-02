//! Port of `pypinyin/contrib/tone_convert.py`, which re-exports
//! `pypinyin.style._tone_convert`.

pub use crate::style::_tone_convert::{
  _fix_v_u, _get_number_from_pinyin, _improve_tone3, _v_to_u, to_finals, to_finals_tone,
  to_finals_tone2, to_finals_tone3, to_initials, to_normal, to_tone, to_tone2, to_tone3,
  tone_to_normal, tone_to_tone2, tone_to_tone3, tone2_to_normal, tone2_to_tone, tone2_to_tone3,
  tone3_to_normal, tone3_to_tone, tone3_to_tone2,
};
