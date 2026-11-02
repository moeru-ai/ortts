//! Port of the part of the `opencc` package (OpenCC 1.4.2, the C++ library
//! behind `opencc.OpenCC`) that `GPT_SoVITS/text/g2pw/onnx_api.py` runs:
//! `OpenCC("s2tw").convert(text)`.
//!
//! `_G2PWBaseOnnxConverter.__call__` converts every sentence with
//! `OpenCC("s2tw")` when `enable_non_tradional_chinese` is set, which
//! `G2PWPinyin` does by default (and `chinese2.py` does not override it).
//!
//! The `s2tw` config (`data/g2pw/opencc/s2tw.json`) is:
//!
//! 1. normalization: a conversion with `CJK_Compatibility_Ideographs` over the
//!    whole text, no segmentation (`ConfigBasedConverter`);
//! 2. segmentation: `mmseg` (`MaxMatchSegmentation`) with the union group
//!    `STPhrases` + `STPhrases_GeneratedFromRegionalPhrases`;
//! 3. conversion chain: the short-circuit group (that union group, then
//!    `STCharacters`), then the short-circuit group (`TWVariantsPhrases`,
//!    `TWVariants`).
//!
//! The dictionaries are the `.ocd2` files of the pinned wheel, dumped to the
//! OpenCC text format by `parity/g2pw/convert_data.py`.
//! The structs mirror the C++ classes (`PrefixMatch` with its `LeafMatcher`
//! and `GroupMatcher`, `MaxMatchSegmentation`, `Conversion`,
//! `ConversionChain`, `SingleStageConverter`).

use std::collections::HashMap;
use std::sync::LazyLock;

const CJK_COMPATIBILITY_IDEOGRAPHS: &str =
  include_str!("../../data/g2pw/opencc/CJK_Compatibility_Ideographs.txt");
const ST_PHRASES: &str = include_str!("../../data/g2pw/opencc/STPhrases.txt");
const ST_PHRASES_GENERATED_FROM_REGIONAL_PHRASES: &str =
  include_str!("../../data/g2pw/opencc/STPhrases_GeneratedFromRegionalPhrases.txt");
const ST_CHARACTERS: &str = include_str!("../../data/g2pw/opencc/STCharacters.txt");
const TW_VARIANTS_PHRASES: &str = include_str!("../../data/g2pw/opencc/TWVariantsPhrases.txt");
const TW_VARIANTS: &str = include_str!("../../data/g2pw/opencc/TWVariants.txt");

/// Port of `TextDict` parsing: one `key<TAB>value1 value2 ...` entry per
/// line. Only `DictEntry::GetDefault()` (the first value) is used by
/// conversion, so only that value is kept.
fn parse_text_dict(text: &str) -> Vec<(String, String)> {
  let mut entries = Vec::new();
  for line in text.split('\n') {
    let line = line.strip_suffix('\r').unwrap_or(line);
    if line.is_empty() {
      continue;
    }
    let (key, values) = line
      .split_once('\t')
      .expect("OpenCC text dict line without a tab");
    let default = values.split(' ').next().unwrap_or("");
    entries.push((key.to_string(), default.to_string()));
  }
  entries
}

/// A dictionary node of a config: a leaf dictionary or a `group` with its
/// `match_policy` (port of `Dict` / `DictGroup` / `UnionDictGroup`).
enum Dict {
  Leaf(Vec<(String, String)>),
  ShortCircuit(Vec<Dict>),
  Union(Vec<Dict>),
}

/// Port of `LeafMatcher` (`PrefixMatch.cpp`): a character trie over the keys
/// of one or more leaf dictionaries. The first entry added for a key wins.
#[derive(Default)]
struct LeafMatcher {
  root: Node,
}

#[derive(Default)]
struct Node {
  /// `(keyLength in chars, value)`.
  candidate: Option<(usize, String)>,
  children: HashMap<char, Node>,
}

impl LeafMatcher {
  fn add_dict(&mut self, entries: &[(String, String)]) {
    for (key, value) in entries {
      let mut node = &mut self.root;
      for c in key.chars() {
        node = node.children.entry(c).or_default();
      }
      if node.candidate.is_none() {
        node.candidate = Some((key.chars().count(), value.clone()));
      }
    }
  }

  /// `LeafMatcher::MatchPrefixCandidate`: the longest key that is a prefix
  /// of `word`.
  fn match_prefix(&self, word: &[char]) -> Option<(usize, &str)> {
    let mut node = &self.root;
    let mut matched: Option<(usize, &str)> = None;
    for c in word {
      match node.children.get(c) {
        Some(child) => node = child,
        None => break,
      }
      if let Some((len, value)) = &node.candidate
        && matched.is_none_or(|(m, _)| *len > m)
      {
        matched = Some((*len, value.as_str()));
      }
    }
    matched
  }
}

/// Port of `PrefixMatch::Tables::Matcher` (`LeafMatcher` / `GroupMatcher`).
enum Matcher {
  Leaf(LeafMatcher),
  ShortCircuit(Vec<Matcher>),
  Union(Vec<Matcher>),
}

impl Matcher {
  /// Port of `BuildMatcher` (with the `PrefixMatch` constructor's unwrapping
  /// of single-child groups and `CanFlattenAsUnion`).
  fn build(dict: &Dict) -> Self {
    let mut dict = dict;
    while let Dict::ShortCircuit(items) | Dict::Union(items) = dict {
      if items.len() == 1 {
        dict = &items[0];
      } else {
        break;
      }
    }
    Self::build_inner(dict)
  }

  fn build_inner(dict: &Dict) -> Self {
    match dict {
      Dict::Leaf(entries) => {
        let mut leaf = LeafMatcher::default();
        leaf.add_dict(entries);
        Matcher::Leaf(leaf)
      }
      _ if can_flatten_as_union(dict) => {
        let mut leaf = LeafMatcher::default();
        collect_all_leaf_dicts(dict, &mut leaf);
        Matcher::Leaf(leaf)
      }
      Dict::ShortCircuit(items) => {
        Matcher::ShortCircuit(items.iter().map(Self::build_inner).collect())
      }
      Dict::Union(items) => Matcher::Union(items.iter().map(Self::build_inner).collect()),
    }
  }

  /// `MatchPrefixCandidate`: `(key length in chars, value)`.
  fn match_prefix(&self, word: &[char]) -> Option<(usize, &str)> {
    match self {
      Matcher::Leaf(leaf) => leaf.match_prefix(word),
      // MatchPrefixShortCircuit
      Matcher::ShortCircuit(children) => children.iter().find_map(|c| c.match_prefix(word)),
      // MatchPrefixUnion: the longest match, earlier child wins ties.
      Matcher::Union(children) => {
        let mut best: Option<(usize, &str)> = None;
        for child in children {
          if let Some(candidate) = child.match_prefix(word)
            && best.is_none_or(|(b, _)| candidate.0 > b)
          {
            best = Some(candidate);
          }
        }
        best
      }
    }
  }
}

/// Port of `CanFlattenAsUnion`.
fn can_flatten_as_union(dict: &Dict) -> bool {
  match dict {
    Dict::Leaf(_) => true,
    Dict::ShortCircuit(_) => false,
    Dict::Union(items) => items.iter().all(can_flatten_as_union),
  }
}

/// Port of `CollectAllLeafDicts`.
fn collect_all_leaf_dicts(dict: &Dict, out: &mut LeafMatcher) {
  match dict {
    Dict::Leaf(entries) => out.add_dict(entries),
    Dict::ShortCircuit(items) | Dict::Union(items) => {
      for item in items {
        collect_all_leaf_dicts(item, out);
      }
    }
  }
}

/// Port of `UTF8Util::IdeographicDescriptionOperatorArity`.
fn ideographic_description_operator_arity(c: char) -> usize {
  match c as u32 {
    0x2FF2 | 0x2FF3 => 3,
    0x2FFE | 0x2FFF => 1,
    0x2FF0 | 0x2FF1 | 0x2FF4..=0x2FFD => 2,
    _ => 0,
  }
}

enum IdsParseStatus {
  Complete,
  Incomplete,
  Invalid,
}

/// Port of `UTF8Util::ConsumeIdeographicDescriptionSequence`, counting in
/// chars instead of bytes.
fn consume_ideographic_description_sequence(
  s: &[char],
  depth_left: usize,
  max_code_points: usize,
  consumed: &mut usize,
  code_points: &mut usize,
) -> IdsParseStatus {
  if s.is_empty() {
    return IdsParseStatus::Incomplete;
  }
  if depth_left == 0 || *code_points >= max_code_points {
    return IdsParseStatus::Invalid;
  }
  *code_points += 1;
  let arity = ideographic_description_operator_arity(s[0]);
  if arity == 0 {
    *consumed = 1;
    return IdsParseStatus::Complete;
  }
  let mut offset = 1;
  for _ in 0..arity {
    if offset >= s.len() {
      return IdsParseStatus::Incomplete;
    }
    let mut operand_length = 0;
    match consume_ideographic_description_sequence(
      &s[offset..],
      depth_left - 1,
      max_code_points,
      &mut operand_length,
      code_points,
    ) {
      IdsParseStatus::Complete => {}
      status => return status,
    }
    offset += operand_length;
  }
  *consumed = offset;
  IdsParseStatus::Complete
}

/// Port of `UTF8Util::NextIdeographicDescriptionSequenceLength` (in chars).
fn next_ideographic_description_sequence_length(s: &[char]) -> usize {
  const MAX_IDS_DEPTH: usize = 16;
  const MAX_IDS_CODE_POINTS: usize = 64;
  if s.is_empty() || ideographic_description_operator_arity(s[0]) == 0 {
    return 0;
  }
  let mut consumed = 0;
  let mut code_points = 0;
  match consume_ideographic_description_sequence(
    s,
    MAX_IDS_DEPTH,
    MAX_IDS_CODE_POINTS,
    &mut consumed,
    &mut code_points,
  ) {
    IdsParseStatus::Complete => consumed,
    _ => 0,
  }
}

/// The length of the next unmatched unit: a complete ideographic
/// description sequence, or one character. (`SkipUnmatchable` only skips
/// characters that cannot start a key, so it does not change the result and
/// is not ported.)
fn unmatched_length(s: &[char]) -> usize {
  match next_ideographic_description_sequence_length(s) {
    0 => 1,
    n => n,
  }
}

/// Port of `Conversion`: prefix-match replacement with one dictionary.
struct Conversion {
  prefix_match: Matcher,
}

impl Conversion {
  /// `Conversion::AppendConverted`.
  fn append_converted(&self, phrase: &[char], output: &mut String) {
    let mut pos = 0;
    while pos < phrase.len() {
      let rest = &phrase[pos..];
      match self.prefix_match.match_prefix(rest) {
        Some((len, value)) => {
          output.push_str(value);
          pos += len.min(rest.len());
        }
        None => {
          let len = unmatched_length(rest);
          output.extend(&rest[..len]);
          pos += len;
        }
      }
    }
  }
}

/// Port of `ConversionChain::AppendConvertedSegment`.
fn append_converted_segment(conversions: &[Conversion], segment: &[char], output: &mut String) {
  let mut converted: Vec<char> = segment.to_vec();
  for conversion in conversions {
    let mut next = String::new();
    conversion.append_converted(&converted, &mut next);
    converted = next.chars().collect();
  }
  output.extend(converted);
}

/// Port of `MaxMatchSegmentation::Segment` (`SegmentText`): matched keys
/// become their own segment; runs of unmatched characters are merged into
/// one segment.
fn segment_text(prefix_match: &Matcher, text: &[char]) -> Vec<Vec<char>> {
  let mut segments = Vec::new();
  let mut seg_start = 0;
  let mut seg_length = 0;
  let mut pos = 0;
  while pos < text.len() {
    let rest = &text[pos..];
    let matched_length = match prefix_match.match_prefix(rest) {
      None => {
        let len = unmatched_length(rest);
        seg_length += len;
        len
      }
      Some((len, _)) => {
        if seg_length > 0 {
          segments.push(text[seg_start..seg_start + seg_length].to_vec());
          seg_length = 0;
        }
        segments.push(rest[..len].to_vec());
        seg_start = pos + len;
        len
      }
    };
    pos += matched_length;
  }
  if seg_length > 0 {
    segments.push(text[seg_start..seg_start + seg_length].to_vec());
  }
  segments
}

/// Port of `opencc.OpenCC` for the `s2tw` config.
pub struct OpenCC {
  normalization: Vec<Conversion>,
  segmentation: Matcher,
  conversion_chain: Vec<Conversion>,
}

fn st_phrases_union() -> Dict {
  Dict::Union(vec![
    Dict::Leaf(parse_text_dict(ST_PHRASES)),
    Dict::Leaf(parse_text_dict(ST_PHRASES_GENERATED_FROM_REGIONAL_PHRASES)),
  ])
}

static S2TW: LazyLock<OpenCC> = LazyLock::new(|| OpenCC {
  normalization: vec![Conversion {
    prefix_match: Matcher::build(&Dict::Leaf(parse_text_dict(CJK_COMPATIBILITY_IDEOGRAPHS))),
  }],
  segmentation: Matcher::build(&st_phrases_union()),
  conversion_chain: vec![
    Conversion {
      prefix_match: Matcher::build(&Dict::ShortCircuit(vec![
        st_phrases_union(),
        Dict::Leaf(parse_text_dict(ST_CHARACTERS)),
      ])),
    },
    Conversion {
      prefix_match: Matcher::build(&Dict::ShortCircuit(vec![
        Dict::Leaf(parse_text_dict(TW_VARIANTS_PHRASES)),
        Dict::Leaf(parse_text_dict(TW_VARIANTS)),
      ])),
    },
  ],
});

impl OpenCC {
  /// Port of `OpenCC("s2tw")`. Only the `s2tw` config is ported; the
  /// converter is built once and shared.
  pub fn s2tw() -> &'static OpenCC {
    &S2TW
  }

  /// Port of `OpenCC.convert(text)` (`ConfigBasedConverter::Convert`):
  /// the normalization pass over the whole text, then segmentation and the
  /// conversion chain per segment.
  ///
  /// `SingleStageConverter::Convert` hands each segment to the chain as a
  /// C string, so a segment is cut at its first NUL character, as in the C++
  /// library (`"a\0b"` converts to `"a"`).
  pub fn convert(&self, text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut normalized = String::new();
    append_converted_segment(&self.normalization, &chars, &mut normalized);

    let normalized: Vec<char> = normalized.chars().collect();
    let mut converted = String::new();
    for segment in segment_text(&self.segmentation, &normalized) {
      let end = segment
        .iter()
        .position(|&c| c == '\0')
        .unwrap_or(segment.len());
      append_converted_segment(&self.conversion_chain, &segment[..end], &mut converted);
    }
    converted
  }
}
