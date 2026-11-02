//! Port of `ToJyutping/utils.py` (`punct_dict`, the punctuation sets and
//! `format_romanization_text`).

use std::collections::HashMap;
use std::sync::OnceLock;

/// Keys of `ToJyutping.utils.punct_dict` (first `zip` argument).
const PUNCT_DICT_KEYS: &str = "!\"'(),-./:;?[]{}~·‐‑‒–—―‘’“”…⋮⋯⸱⸳⸺⸻、。〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟・︐︑︒︓︔︕︖︗︘︙︱︲︵︶︷︸︹︺︻︼︽︾︿﹀﹁﹂﹃﹄﹇﹈﹐﹑﹒﹔﹕﹖﹗﹘﹙﹚﹛﹜﹝﹞﹣！＂＇（），－．／：；？［］｛｝～｟｠｡｢｣､･";

/// Values of `ToJyutping.utils.punct_dict` (second `zip` argument).
const PUNCT_DICT_VALUES: &str = "!\"'(),-./:;?[]{}~·------‘’“”………··--,.‘’“”“”‘’[][][][][]~“””·,,.:;!?[]…--(){}[][]“”‘’“”‘’[],,.;:?!-(){}[]-!\"'(),-./:;?[]{}~().“”,·";

/// Port of `ToJyutping.utils.punct_dict`.
pub fn punct_dict() -> &'static HashMap<char, char> {
  static PUNCT_DICT: OnceLock<HashMap<char, char>> = OnceLock::new();
  PUNCT_DICT.get_or_init(|| {
    PUNCT_DICT_KEYS
      .chars()
      .zip(PUNCT_DICT_VALUES.chars())
      .collect()
  })
}

/// Port of `ToJyutping.utils.left_bracket`.
pub const LEFT_BRACKET: &str = "([{‘“";
/// Port of `ToJyutping.utils.right_bracket`.
pub const RIGHT_BRACKET: &str = ")]}’”";
/// Port of `ToJyutping.utils.left_punct` (equal to `left_bracket`).
pub const LEFT_PUNCT: &str = LEFT_BRACKET;
/// Port of `ToJyutping.utils.right_punct`.
pub const RIGHT_PUNCT: &str = "!,.:;?…)]}’”";
/// Port of `ToJyutping.utils.other_punct`.
pub const OTHER_PUNCT: &str = "\"'·-~";
/// Port of `ToJyutping.utils.left_or_other_punct`.
pub const LEFT_OR_OTHER_PUNCT: &str = " ([{‘“\"'·-~";
/// Port of `ToJyutping.utils.right_or_other_punct`.
pub const RIGHT_OR_OTHER_PUNCT: &str = "!,.:;?…)]}’”\"'·-~";
/// Port of `ToJyutping.utils.minus_signs`.
pub const MINUS_SIGNS: &str = "-﹣－";
/// Port of `ToJyutping.utils.decimal_seps`.
pub const DECIMAL_SEPS: &str = "',.·⸱⸳﹒＇．";
/// Port of `ToJyutping.utils.digits`.
pub const DIGITS: &str =
  "0０𝟎𝟘𝟢𝟬𝟶🯰1１𝟏𝟙𝟣𝟭𝟷🯱2２𝟐𝟚𝟤𝟮𝟸🯲3３𝟑𝟛𝟥𝟯𝟹🯳4４𝟒𝟜𝟦𝟰𝟺🯴5５𝟓𝟝𝟧𝟱𝟻🯵6６𝟔𝟞𝟨𝟲𝟼🯶7７𝟕𝟟𝟩𝟳𝟽🯷8８𝟖𝟠𝟪𝟴𝟾🯸9９𝟗𝟡𝟫𝟵𝟿🯹";

/// Port of `ToJyutping.utils.left_bracket_to_right`.
fn left_bracket_to_right(c: char) -> char {
  let i = LEFT_BRACKET
    .chars()
    .position(|x| x == c)
    .expect("c is a left bracket");
  RIGHT_BRACKET.chars().nth(i).expect("same length")
}

/// Python `str.isspace()` for a single character.
pub fn py_isspace(c: char) -> bool {
  matches!(
    c,
    '\t'
      | '\n'
      | '\x0b'
      | '\x0c'
      | '\r'
      | '\x1c'..='\x1f'
      | ' '
      | '\u{85}'
      | '\u{a0}'
      | '\u{1680}'
      | '\u{2000}'..='\u{200a}'
      | '\u{2028}'
      | '\u{2029}'
      | '\u{202f}'
      | '\u{205f}'
      | '\u{3000}'
  )
}

/// Python `str.isspace()`: true for a non-empty string of whitespace.
pub fn py_str_isspace(s: &str) -> bool {
  !s.is_empty() && s.chars().all(py_isspace)
}

/// `x in set` where `x` is an optional Python `str` and `set` is a set of
/// one-character strings. `None` and strings of other lengths are never
/// members.
fn str_in(x: Option<&str>, set: &str) -> bool {
  let Some(x) = x else {
    return false;
  };
  let mut chars = x.chars();
  match (chars.next(), chars.next()) {
    (Some(c), None) => set.contains(c),
    _ => false,
  }
}

/// `x in set` where `x` is an optional single character.
fn char_in(x: Option<char>, set: &str) -> bool {
  x.is_some_and(|c| set.contains(c))
}

/// Python `len(x)` for a `str`.
fn py_len(x: &str) -> usize {
  x.chars().count()
}

/// Port of `ToJyutping.utils.format_romanization_text`: applies `inner` to
/// every run of characters matched by `[^\0-\x1f\x80-\x9f]+` and keeps the
/// control characters between runs unchanged.
pub fn format_romanization_text<F>(s: &str, conv: F) -> String
where
  F: Fn(&str) -> Vec<(String, Option<String>)>,
{
  let is_control = |c: char| matches!(c, '\0'..='\x1f' | '\u{80}'..='\u{9f}');
  let mut out = String::new();
  let mut run = String::new();
  for ch in s.chars() {
    if is_control(ch) {
      if !run.is_empty() {
        out += &format_romanization_text_inner(&run, &conv);
        run.clear();
      }
      out.push(ch);
    } else {
      run.push(ch);
    }
  }
  if !run.is_empty() {
    out += &format_romanization_text_inner(&run, &conv);
  }
  out
}

/// Port of the nested `inner(m)` of `format_romanization_text`.
fn format_romanization_text_inner<F>(m: &str, conv: &F) -> String
where
  F: Fn(&str) -> Vec<(String, Option<String>)>,
{
  let punct_dict = punct_dict();
  let mut t: Vec<Option<String>> = vec![None];
  let mut d: Vec<Option<char>> = vec![None];
  for (k, v) in conv(m) {
    if let Some(v) = v.filter(|v| !v.is_empty()) {
      t.push(Some(v));
      d.push(None);
    } else if !py_str_isspace(&k) {
      let kc = k
        .chars()
        .next()
        .expect("conv yields one character per item");
      t.push(Some(
        punct_dict
          .get(&kc)
          .map(|c| c.to_string())
          .unwrap_or_default(),
      ));
      d.push(Some(kc));
    }
  }
  t.push(None);
  d.push(None);

  let mut l = String::new();
  let mut b: Vec<char> = Vec::new();
  for i in 1..t.len() - 1 {
    let p = t[i - 1].as_deref();
    let c = t[i].as_deref().expect("inner items are always strings");
    let n = t[i + 1].as_deref();

    let between = || {
      let mut j = i - 1;
      while j != 0 && str_in(t[j].as_deref(), RIGHT_BRACKET) {
        j -= 1;
      }
      let f = j != 0 && t[j].as_deref().is_some_and(|x| py_len(x) > 1);
      let mut j = i + 1;
      while j < t.len() - 1 && str_in(t[j].as_deref(), LEFT_BRACKET) {
        j += 1;
      }
      let g = t[j].as_deref().is_some_and(|x| py_len(x) > 1);
      f && g
    };

    let lspace = |l: &mut String| {
      if let Some(last) = l.chars().last()
        && !LEFT_OR_OTHER_PUNCT.contains(last)
      {
        l.push(' ');
      }
    };

    let rspace = |l: &mut String| {
      let cond = if char_in(d[i + 1], MINUS_SIGNS) {
        i < d.len() - 2 && char_in(d[i + 2], DIGITS)
      } else {
        !str_in(n, RIGHT_OR_OTHER_PUNCT)
      };
      if cond {
        l.push(' ');
      }
    };

    if py_len(c) > 1 {
      lspace(&mut l);
      l += c;
      rspace(&mut l);
    } else if c.is_empty()
      || char_in(d[i], MINUS_SIGNS)
        && char_in(d[i + 1], DIGITS)
        && !(p == Some("") || p == Some("-"))
    {
      if !l.ends_with("[…]") {
        l += "[…]";
      }
    } else if char_in(d[i], DECIMAL_SEPS) && char_in(d[i + 1], DIGITS) && char_in(d[i - 1], DIGITS)
    {
      continue;
    } else {
      let cc = c.chars().next().expect("len(c) == 1");
      if LEFT_PUNCT.contains(cc) {
        lspace(&mut l);
        l += c;
        b.push(left_bracket_to_right(cc));
      } else if RIGHT_PUNCT.contains(cc) {
        l += c;
        rspace(&mut l);
        if let Some(pos) = b.iter().rposition(|&x| x == cc) {
          b.truncate(pos);
        }
      } else if cc == '-' {
        if p == Some("-") {
          continue;
        }
        if n == Some("-") || between() {
          l += " – ";
        } else {
          l += c;
        }
      } else if cc == '~' {
        if p == Some("~") && n != Some("~") || between() {
          l += "~ ";
        } else {
          l += c;
        }
      } else if cc == '·' {
        l += c;
      } else {
        let mut y = None;
        for (j, &x) in b.iter().enumerate().rev() {
          if RIGHT_BRACKET.contains(x) {
            break;
          }
          if x == cc {
            y = Some(j);
            break;
          }
        }
        if let Some(j) = y {
          b.truncate(j);
          l += c;
          rspace(&mut l);
        } else {
          lspace(&mut l);
          l += c;
          b.push(cc);
        }
      }
    }
  }
  l.split(py_isspace)
    .filter(|x| !x.is_empty())
    .collect::<Vec<_>>()
    .join(" ")
}
