//! Port of `split_lang/model.py`.

/// Port of `class LangSectionType(Enum)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LangSectionType {
  /// `ZH_JA = "zh_ja"`.
  ZhJa,
  /// `KO = "ko"`.
  Ko,
  /// `PUNCTUATION = "punctuation"`.
  Punctuation,
  /// `NEWLINE = "newline"`.
  Newline,
  /// `DIGIT = "digit"`.
  Digit,
  /// `OTHERS = "others"`.
  Others,
  /// `ALL = "all"`.
  All,
}

impl LangSectionType {
  /// The enum `value`.
  pub fn value(self) -> &'static str {
    match self {
      Self::ZhJa => "zh_ja",
      Self::Ko => "ko",
      Self::Punctuation => "punctuation",
      Self::Newline => "newline",
      Self::Digit => "digit",
      Self::Others => "others",
      Self::All => "all",
    }
  }
}

/// Port of `class SubString(BaseModel)`.
///
/// `index` and `length` count Unicode code points, like Python `len()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubString {
  /// `lang`: language of `text`.
  pub lang: String,
  /// `text`: text of the substring.
  pub text: String,
  /// `index`: index of `text` in the original string.
  pub index: usize,
  /// `length`: length of `text`.
  pub length: usize,
}

/// Port of `class SubStringSection(BaseModel)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubStringSection {
  /// `lang_section_type`.
  pub lang_section_type: LangSectionType,
  /// `text`: original text of this section.
  pub text: String,
  /// `substrings`: substrings split from `text`.
  pub substrings: Vec<SubString>,
}
