//! Port of `GPT_SoVITS/TTS_infer_pack/text_segmentation_method.py` (GPT-SoVITS @ 04a44eb).
//!
//! Lengths and indices count Unicode code points, like Python `str`.
//! Where Python raises, the Rust function returns an error with the same
//! exception name.

use std::fmt;
use std::sync::{LazyLock, RwLock};

/// Port of the module-level `punctuation` set.
pub const PUNCTUATION: &[char] = &['!', '?', '…', ',', '.', '-', ' '];

/// Port of `splits` (a Python `set`, kept in upstream source order; no
/// caller observes the order).
pub const SPLITS: &[char] = &[
  '，', '。', '？', '！', ',', '.', '?', '!', '~', ':', '：', '—', '…',
];

/// Port of the `punds` set local to `cut5`. Upstream lists `";"` twice; the
/// set keeps it once.
pub const CUT5_PUNDS: &[char] = &[
  ',', '.', ';', '?', '!', '、', '，', '。', '？', '！', '：', '…',
];

/// The `IndexError` that Python raises (for example `split("")` reading
/// `todo_text[-1]`). Holds the Python message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexError(pub &'static str);

impl fmt::Display for IndexError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "IndexError: {}", self.0)
  }
}

impl std::error::Error for IndexError {}

/// The `ValueError` that `get_method` raises for an unknown name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueError(pub String);

impl fmt::Display for ValueError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "ValueError: {}", self.0)
  }
}

impl std::error::Error for ValueError {}

/// A segmentation method as stored in `METHODS` (`Callable[[str], str]`).
pub type Method = fn(&str) -> Result<String, IndexError>;

/// Port of `METHODS`: a dict, so insertion order is kept and re-registering a
/// name replaces the function in place. Python fills it at import time
/// through the `@register_method` decorators, in source order.
static METHODS: LazyLock<RwLock<Vec<(String, Method)>>> = LazyLock::new(|| {
  let methods: [(&str, Method); 6] = [
    ("cut0", cut0),
    ("cut1", cut1),
    ("cut2", cut2),
    ("cut3", cut3),
    ("cut4", cut4),
    ("cut5", cut5),
  ];
  RwLock::new(
    methods
      .into_iter()
      .map(|(name, func)| (name.to_string(), func))
      .collect(),
  )
});

/// Port of `get_method(name)`.
pub fn get_method(name: &str) -> Result<Method, ValueError> {
  let methods = METHODS.read().unwrap_or_else(|e| e.into_inner());
  methods
    .iter()
    .find(|(n, _)| n == name)
    .map(|(_, func)| *func)
    .ok_or_else(|| ValueError(format!("Method {name} not found")))
}

/// Port of `get_method_names()`.
pub fn get_method_names() -> Vec<String> {
  let methods = METHODS.read().unwrap_or_else(|e| e.into_inner());
  methods.iter().map(|(n, _)| n.clone()).collect()
}

/// Port of `register_method(name)(func)`: stores `func` under `name` and
/// returns it, as the decorator does.
pub fn register_method(name: &str, func: Method) -> Method {
  let mut methods = METHODS.write().unwrap_or_else(|e| e.into_inner());
  match methods.iter_mut().find(|(n, _)| n == name) {
    Some(entry) => entry.1 = func,
    None => methods.push((name.to_string(), func)),
  }
  func
}

/// `set(item).issubset(set_)` for a set of single characters.
fn issubset(item: &str, set_: &[char]) -> bool {
  item.chars().all(|ch| set_.contains(&ch))
}

/// Keyword arguments of `split_big_text`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitBigTextOptions {
  /// `max_len=510`.
  pub max_len: usize,
}

impl Default for SplitBigTextOptions {
  fn default() -> Self {
    Self { max_len: 510 }
  }
}

/// Port of `split_big_text(text, max_len=510)`.
///
/// `re.split("([" + "".join(splits) + "])", text)` alternates text pieces
/// with single-character separators; this walks the same pieces without a
/// regex. The order of `splits` in the character class does not matter
/// because none of its characters is special inside `[...]`.
pub fn split_big_text(text: &str, options: SplitBigTextOptions) -> Vec<String> {
  let max_len = options.max_len;

  // re.split with one capturing group: piece, sep, piece, sep, ..., piece.
  let mut segments: Vec<&str> = Vec::new();
  let mut start = 0;
  for (i, ch) in text.char_indices() {
    if SPLITS.contains(&ch) {
      segments.push(&text[start..i]);
      segments.push(&text[i..i + ch.len_utf8()]);
      start = i + ch.len_utf8();
    }
  }
  segments.push(&text[start..]);

  let mut result = Vec::new();
  let mut current_segment = String::new();
  let mut current_len = 0;

  for segment in segments {
    let segment_len = segment.chars().count();
    if current_len + segment_len > max_len {
      result.push(std::mem::take(&mut current_segment));
      current_segment.push_str(segment);
      current_len = segment_len;
    } else {
      current_segment.push_str(segment);
      current_len += segment_len;
    }
  }

  if !current_segment.is_empty() {
    result.push(current_segment);
  }

  result
}

/// Port of `split(todo_text)`.
///
/// An empty `todo_text` raises `IndexError` in Python (`todo_text[-1]`).
pub fn split(todo_text: &str) -> Result<Vec<String>, IndexError> {
  let mut todo_text = todo_text.replace("……", "。").replace("——", "，");
  match todo_text.chars().next_back() {
    None => return Err(IndexError("string index out of range")),
    Some(last) => {
      if !SPLITS.contains(&last) {
        todo_text.push('。');
      }
    }
  }
  let chars: Vec<char> = todo_text.chars().collect();
  let len_text = chars.len();
  let mut i_split_head = 0;
  let mut i_split_tail = 0;
  let mut todo_texts = Vec::new();
  loop {
    if i_split_head >= len_text {
      break;
    }
    if SPLITS.contains(&chars[i_split_head]) {
      i_split_head += 1;
      todo_texts.push(chars[i_split_tail..i_split_head].iter().collect());
      i_split_tail = i_split_head;
    } else {
      i_split_head += 1;
    }
  }
  Ok(todo_texts)
}

/// Port of `cut0(inp)` (不切: no split).
pub fn cut0(inp: &str) -> Result<String, IndexError> {
  if !issubset(inp, PUNCTUATION) {
    Ok(inp.to_string())
  } else {
    Ok("\n".to_string())
  }
}

/// Port of `cut1(inp)` (凑四句一切: one segment per four sentences).
pub fn cut1(inp: &str) -> Result<String, IndexError> {
  let inp = inp.trim_matches('\n');
  let inps = split(inp)?;
  // split_idx = list(range(0, len(inps), 4)); split_idx[-1] = None
  let mut split_idx: Vec<Option<usize>> = (0..inps.len()).step_by(4).map(Some).collect();
  match split_idx.last_mut() {
    Some(last) => *last = None,
    None => return Err(IndexError("list assignment index out of range")),
  }
  let opts: Vec<String> = if split_idx.len() > 1 {
    let mut opts = Vec::new();
    for idx in 0..split_idx.len() - 1 {
      let begin = split_idx[idx].unwrap_or(inps.len());
      let end = split_idx[idx + 1].unwrap_or(inps.len());
      opts.push(inps[begin..end].concat());
    }
    opts
  } else {
    vec![inp.to_string()]
  };
  let opts: Vec<String> = opts
    .into_iter()
    .filter(|item| !issubset(item, PUNCTUATION))
    .collect();
  Ok(opts.join("\n"))
}

/// Port of `cut2(inp)` (凑50字一切: one segment per more than 50 characters).
pub fn cut2(inp: &str) -> Result<String, IndexError> {
  let inp = inp.trim_matches('\n');
  let inps = split(inp)?;
  if inps.len() < 2 {
    return Ok(inp.to_string());
  }
  let mut opts: Vec<String> = Vec::new();
  let mut summ = 0;
  let mut tmp_str = String::new();
  for item in &inps {
    summ += item.chars().count();
    tmp_str.push_str(item);
    if summ > 50 {
      summ = 0;
      opts.push(std::mem::take(&mut tmp_str));
    }
  }
  if !tmp_str.is_empty() {
    opts.push(tmp_str);
  }
  if opts.len() > 1 && opts[opts.len() - 1].chars().count() < 50 {
    let last = opts.pop().unwrap_or_default();
    let n = opts.len();
    opts[n - 1].push_str(&last);
  }
  let opts: Vec<String> = opts
    .into_iter()
    .filter(|item| !issubset(item, PUNCTUATION))
    .collect();
  Ok(opts.join("\n"))
}

/// Port of `cut3(inp)` (按中文句号。切: split on `。`).
pub fn cut3(inp: &str) -> Result<String, IndexError> {
  let inp = inp.trim_matches('\n');
  let opts: Vec<&str> = inp
    .trim_matches('。')
    .split('。')
    .filter(|item| !issubset(item, PUNCTUATION))
    .collect();
  Ok(opts.join("\n"))
}

/// Port of `cut4(inp)` (按英文句号.切: split on `.` not next to a digit).
///
/// `re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))`, where `\d` is Python's
/// Unicode decimal digit (`str.isdecimal`).
pub fn cut4(inp: &str) -> Result<String, IndexError> {
  let inp = inp.trim_matches('\n');
  let text = inp.trim_matches('.');
  let chars: Vec<(usize, char)> = text.char_indices().collect();
  let mut pieces: Vec<&str> = Vec::new();
  let mut start = 0;
  for (k, &(i, ch)) in chars.iter().enumerate() {
    if ch != '.' {
      continue;
    }
    let digit_before = k > 0 && py_isdecimal(chars[k - 1].1);
    let digit_after = k + 1 < chars.len() && py_isdecimal(chars[k + 1].1);
    if !digit_before && !digit_after {
      pieces.push(&text[start..i]);
      start = i + 1;
    }
  }
  pieces.push(&text[start..]);
  let opts: Vec<&str> = pieces
    .into_iter()
    .filter(|item| !issubset(item, PUNCTUATION))
    .collect();
  Ok(opts.join("\n"))
}

/// Port of `cut5(inp)` (按标点符号切: split after each punctuation mark,
/// except a `.` between two digits).
pub fn cut5(inp: &str) -> Result<String, IndexError> {
  let inp = inp.trim_matches('\n');
  let punds = CUT5_PUNDS;
  let chars: Vec<char> = inp.chars().collect();
  let mut mergeitems: Vec<String> = Vec::new();
  let mut items = String::new();

  for (i, &ch) in chars.iter().enumerate() {
    if punds.contains(&ch) {
      if ch == '.'
        && i > 0
        && i + 1 < chars.len()
        && py_isdigit(chars[i - 1])
        && py_isdigit(chars[i + 1])
      {
        items.push(ch);
      } else {
        items.push(ch);
        mergeitems.push(std::mem::take(&mut items));
      }
    } else {
      items.push(ch);
    }
  }

  if !items.is_empty() {
    mergeitems.push(items);
  }

  let opt: Vec<String> = mergeitems
    .into_iter()
    .filter(|item| !issubset(item, punds))
    .collect();
  Ok(opt.join("\n"))
}

fn in_ranges(ch: char, ranges: &[(char, char)]) -> bool {
  ranges
    .binary_search_by(|&(lo, hi)| {
      if hi < ch {
        std::cmp::Ordering::Less
      } else if lo > ch {
        std::cmp::Ordering::Greater
      } else {
        std::cmp::Ordering::Equal
      }
    })
    .is_ok()
}

/// Python `str.isdecimal()` for one character, which is also what `re`
/// matches with `\d` in a `str` pattern (Unicode 14.0.0 in Python 3.11).
pub fn py_isdecimal(ch: char) -> bool {
  in_ranges(ch, PY_DECIMAL_RANGES)
}

/// Python `str.isdigit()` for one character (Unicode 14.0.0 in Python 3.11).
pub fn py_isdigit(ch: char) -> bool {
  in_ranges(ch, PY_DIGIT_RANGES)
}

// BEGIN GENERATED (gen_digit_tables.py, Unicode 14.0.0)
/// Code points where Python `str.isdecimal()` (and `re` `\d`) is true.
const PY_DECIMAL_RANGES: &[(char, char)] = &[
  ('\u{0030}', '\u{0039}'),
  ('\u{0660}', '\u{0669}'),
  ('\u{06f0}', '\u{06f9}'),
  ('\u{07c0}', '\u{07c9}'),
  ('\u{0966}', '\u{096f}'),
  ('\u{09e6}', '\u{09ef}'),
  ('\u{0a66}', '\u{0a6f}'),
  ('\u{0ae6}', '\u{0aef}'),
  ('\u{0b66}', '\u{0b6f}'),
  ('\u{0be6}', '\u{0bef}'),
  ('\u{0c66}', '\u{0c6f}'),
  ('\u{0ce6}', '\u{0cef}'),
  ('\u{0d66}', '\u{0d6f}'),
  ('\u{0de6}', '\u{0def}'),
  ('\u{0e50}', '\u{0e59}'),
  ('\u{0ed0}', '\u{0ed9}'),
  ('\u{0f20}', '\u{0f29}'),
  ('\u{1040}', '\u{1049}'),
  ('\u{1090}', '\u{1099}'),
  ('\u{17e0}', '\u{17e9}'),
  ('\u{1810}', '\u{1819}'),
  ('\u{1946}', '\u{194f}'),
  ('\u{19d0}', '\u{19d9}'),
  ('\u{1a80}', '\u{1a89}'),
  ('\u{1a90}', '\u{1a99}'),
  ('\u{1b50}', '\u{1b59}'),
  ('\u{1bb0}', '\u{1bb9}'),
  ('\u{1c40}', '\u{1c49}'),
  ('\u{1c50}', '\u{1c59}'),
  ('\u{a620}', '\u{a629}'),
  ('\u{a8d0}', '\u{a8d9}'),
  ('\u{a900}', '\u{a909}'),
  ('\u{a9d0}', '\u{a9d9}'),
  ('\u{a9f0}', '\u{a9f9}'),
  ('\u{aa50}', '\u{aa59}'),
  ('\u{abf0}', '\u{abf9}'),
  ('\u{ff10}', '\u{ff19}'),
  ('\u{104a0}', '\u{104a9}'),
  ('\u{10d30}', '\u{10d39}'),
  ('\u{11066}', '\u{1106f}'),
  ('\u{110f0}', '\u{110f9}'),
  ('\u{11136}', '\u{1113f}'),
  ('\u{111d0}', '\u{111d9}'),
  ('\u{112f0}', '\u{112f9}'),
  ('\u{11450}', '\u{11459}'),
  ('\u{114d0}', '\u{114d9}'),
  ('\u{11650}', '\u{11659}'),
  ('\u{116c0}', '\u{116c9}'),
  ('\u{11730}', '\u{11739}'),
  ('\u{118e0}', '\u{118e9}'),
  ('\u{11950}', '\u{11959}'),
  ('\u{11c50}', '\u{11c59}'),
  ('\u{11d50}', '\u{11d59}'),
  ('\u{11da0}', '\u{11da9}'),
  ('\u{16a60}', '\u{16a69}'),
  ('\u{16ac0}', '\u{16ac9}'),
  ('\u{16b50}', '\u{16b59}'),
  ('\u{1d7ce}', '\u{1d7ff}'),
  ('\u{1e140}', '\u{1e149}'),
  ('\u{1e2f0}', '\u{1e2f9}'),
  ('\u{1e950}', '\u{1e959}'),
  ('\u{1fbf0}', '\u{1fbf9}'),
];
/// Code points where Python `str.isdigit()` is true.
const PY_DIGIT_RANGES: &[(char, char)] = &[
  ('\u{0030}', '\u{0039}'),
  ('\u{00b2}', '\u{00b3}'),
  ('\u{00b9}', '\u{00b9}'),
  ('\u{0660}', '\u{0669}'),
  ('\u{06f0}', '\u{06f9}'),
  ('\u{07c0}', '\u{07c9}'),
  ('\u{0966}', '\u{096f}'),
  ('\u{09e6}', '\u{09ef}'),
  ('\u{0a66}', '\u{0a6f}'),
  ('\u{0ae6}', '\u{0aef}'),
  ('\u{0b66}', '\u{0b6f}'),
  ('\u{0be6}', '\u{0bef}'),
  ('\u{0c66}', '\u{0c6f}'),
  ('\u{0ce6}', '\u{0cef}'),
  ('\u{0d66}', '\u{0d6f}'),
  ('\u{0de6}', '\u{0def}'),
  ('\u{0e50}', '\u{0e59}'),
  ('\u{0ed0}', '\u{0ed9}'),
  ('\u{0f20}', '\u{0f29}'),
  ('\u{1040}', '\u{1049}'),
  ('\u{1090}', '\u{1099}'),
  ('\u{1369}', '\u{1371}'),
  ('\u{17e0}', '\u{17e9}'),
  ('\u{1810}', '\u{1819}'),
  ('\u{1946}', '\u{194f}'),
  ('\u{19d0}', '\u{19da}'),
  ('\u{1a80}', '\u{1a89}'),
  ('\u{1a90}', '\u{1a99}'),
  ('\u{1b50}', '\u{1b59}'),
  ('\u{1bb0}', '\u{1bb9}'),
  ('\u{1c40}', '\u{1c49}'),
  ('\u{1c50}', '\u{1c59}'),
  ('\u{2070}', '\u{2070}'),
  ('\u{2074}', '\u{2079}'),
  ('\u{2080}', '\u{2089}'),
  ('\u{2460}', '\u{2468}'),
  ('\u{2474}', '\u{247c}'),
  ('\u{2488}', '\u{2490}'),
  ('\u{24ea}', '\u{24ea}'),
  ('\u{24f5}', '\u{24fd}'),
  ('\u{24ff}', '\u{24ff}'),
  ('\u{2776}', '\u{277e}'),
  ('\u{2780}', '\u{2788}'),
  ('\u{278a}', '\u{2792}'),
  ('\u{a620}', '\u{a629}'),
  ('\u{a8d0}', '\u{a8d9}'),
  ('\u{a900}', '\u{a909}'),
  ('\u{a9d0}', '\u{a9d9}'),
  ('\u{a9f0}', '\u{a9f9}'),
  ('\u{aa50}', '\u{aa59}'),
  ('\u{abf0}', '\u{abf9}'),
  ('\u{ff10}', '\u{ff19}'),
  ('\u{104a0}', '\u{104a9}'),
  ('\u{10a40}', '\u{10a43}'),
  ('\u{10d30}', '\u{10d39}'),
  ('\u{10e60}', '\u{10e68}'),
  ('\u{11052}', '\u{1105a}'),
  ('\u{11066}', '\u{1106f}'),
  ('\u{110f0}', '\u{110f9}'),
  ('\u{11136}', '\u{1113f}'),
  ('\u{111d0}', '\u{111d9}'),
  ('\u{112f0}', '\u{112f9}'),
  ('\u{11450}', '\u{11459}'),
  ('\u{114d0}', '\u{114d9}'),
  ('\u{11650}', '\u{11659}'),
  ('\u{116c0}', '\u{116c9}'),
  ('\u{11730}', '\u{11739}'),
  ('\u{118e0}', '\u{118e9}'),
  ('\u{11950}', '\u{11959}'),
  ('\u{11c50}', '\u{11c59}'),
  ('\u{11d50}', '\u{11d59}'),
  ('\u{11da0}', '\u{11da9}'),
  ('\u{16a60}', '\u{16a69}'),
  ('\u{16ac0}', '\u{16ac9}'),
  ('\u{16b50}', '\u{16b59}'),
  ('\u{1d7ce}', '\u{1d7ff}'),
  ('\u{1e140}', '\u{1e149}'),
  ('\u{1e2f0}', '\u{1e2f9}'),
  ('\u{1e950}', '\u{1e959}'),
  ('\u{1f100}', '\u{1f10a}'),
  ('\u{1fbf0}', '\u{1fbf9}'),
];
// END GENERATED
