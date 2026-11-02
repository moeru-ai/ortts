//! Golden parity tests against pypinyin 0.55.0.
//!
//! The fixtures are written by `parity/gen_fixtures.py`. See
//! that script for the compact encodings. Every case is compared exactly.
//!
//! Python exceptions are Rust panics in the port. The tests catch them with
//! `catch_unwind` and compare the Python exception type.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

use pypinyin::_re_digit::is_digit;
use pypinyin::constants::{PINYIN_DICT, RE_HANS};
use pypinyin::contrib::tone_convert as tc;
use pypinyin::converter::default_converter;
use pypinyin::seg::mmseg::{self, PrefixSet, Seg};
use pypinyin::seg::simpleseg::{seg as simpleseg_seg, simple_seg};
use pypinyin::utils::_remove_dup_and_empty;
use pypinyin::{
  Converter, DefaultConverter, Errors, LazyPinyinOptions, NoPinyinResult, Pinyin, PinyinMethods,
  PinyinOptions, Style, UltimateConverter, lazy_pinyin, pinyin,
};
use serde_json::Value;

const ITEM_SEP: &str = "`";
const READ_SEP: &str = "{";
const EMPTY: &str = "\u{2205}";
const ERROR_PREFIX: &str = "\u{2297}";

fn load(name: &str) -> Value {
  let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
  let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
  serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// Panic messages of the port that stand for Python exceptions.
const PY_ERRORS: &[(&str, &str)] = &[
  (
    "unsupported operand type(s) for +: 'NoneType' and 'int'",
    "TypeError",
  ),
  ("TypeError: ord() expected a character", "TypeError"),
  ("string index out of range", "IndexError"),
];

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
  if let Some(s) = payload.downcast_ref::<&str>() {
    (*s).to_string()
  } else if let Some(s) = payload.downcast_ref::<String>() {
    s.clone()
  } else {
    String::new()
  }
}

/// Run `f`, and turn a panic that ports a Python exception into the Python
/// exception type name. Other panics propagate.
fn py_call<T>(f: impl FnOnce() -> T) -> Result<T, &'static str> {
  static HOOK: Once = Once::new();
  HOOK.call_once(|| {
    let default = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
      let msg = panic_message(info.payload());
      if !PY_ERRORS.iter().any(|(m, _)| msg.contains(m)) {
        default(info);
      }
    }));
  });
  match panic::catch_unwind(AssertUnwindSafe(f)) {
    Ok(v) => Ok(v),
    Err(payload) => {
      let msg = panic_message(payload.as_ref());
      match PY_ERRORS.iter().find(|(m, _)| msg.contains(m)) {
        Some((_, ty)) => Err(ty),
        None => panic::resume_unwind(payload),
      }
    }
  }
}

fn style(name: &str) -> Style {
  Style::from_name(name).unwrap_or_else(|| panic!("unknown style {name}"))
}

fn errors(name: &str) -> Errors {
  match name {
    "default" => Errors::Default,
    "ignore" => Errors::Ignore,
    "replace" => Errors::Replace,
    _ => panic!("unknown errors {name}"),
  }
}

fn b(v: &Value, key: &str) -> bool {
  v[key]
    .as_bool()
    .unwrap_or_else(|| panic!("missing bool {key} in {v}"))
}

fn s<'a>(v: &'a Value, key: &str) -> &'a str {
  v[key]
    .as_str()
    .unwrap_or_else(|| panic!("missing str {key} in {v}"))
}

/// Result of `pinyin()` (nested) or `lazy_pinyin()` (flat).
enum Out {
  Pinyin(Vec<Vec<String>>),
  Lazy(Vec<String>),
}

/// Port of `run(c, hans)` in the generator.
fn run(c: &Value, hans: &str) -> Result<Out, &'static str> {
  let st = style(s(c, "style"));
  let errs = errors(s(c, "errors"));
  match s(c, "fn") {
    "pinyin" => {
      let options = PinyinOptions {
        style: st,
        heteronym: b(c, "heteronym"),
        errors: errs,
        strict: b(c, "strict"),
        v_to_u: b(c, "v_to_u"),
        neutral_tone_with_five: b(c, "neutral_tone_with_five"),
      };
      py_call(|| Out::Pinyin(pinyin(hans, &options)))
    }
    "lazy_pinyin" => {
      assert!(!b(c, "heteronym"));
      let options = LazyPinyinOptions {
        style: st,
        errors: errs,
        strict: b(c, "strict"),
        v_to_u: b(c, "v_to_u"),
        neutral_tone_with_five: b(c, "neutral_tone_with_five"),
      };
      py_call(|| Out::Lazy(lazy_pinyin(hans, &options)))
    }
    other => panic!("unknown fn {other}"),
  }
}

/// Port of `encode` in the generator (" " between items, "," between readings).
fn encode(out: &Out) -> String {
  match out {
    Out::Pinyin(r) => r
      .iter()
      .map(|item| item.join(","))
      .collect::<Vec<_>>()
      .join(" "),
    Out::Lazy(r) => r.join(" "),
  }
}

/// Port of `encode_sep` in the generator.
fn encode_sep(out: &Result<Out, &'static str>) -> String {
  match out {
    Err(ty) => format!("{ERROR_PREFIX}{ty}"),
    Ok(Out::Pinyin(r)) if r.is_empty() => EMPTY.to_string(),
    Ok(Out::Lazy(r)) if r.is_empty() => EMPTY.to_string(),
    Ok(Out::Pinyin(r)) => r
      .iter()
      .map(|item| item.join(READ_SEP))
      .collect::<Vec<_>>()
      .join(ITEM_SEP),
    Ok(Out::Lazy(r)) => r.join(ITEM_SEP),
  }
}

/// Exact match counter that reports the first mismatches.
struct Tally {
  name: String,
  cases: usize,
  mismatches: Vec<String>,
}

impl Tally {
  fn new(name: &str) -> Self {
    Self {
      name: name.to_string(),
      cases: 0,
      mismatches: Vec::new(),
    }
  }

  fn check<T: PartialEq + std::fmt::Debug>(
    &mut self,
    what: impl FnOnce() -> String,
    got: T,
    want: T,
  ) {
    self.cases += 1;
    if got != want {
      self
        .mismatches
        .push(format!("{}: got {got:?}, want {want:?}", what()));
    }
  }

  fn finish(self) {
    eprintln!(
      "{}: {} cases, {} exact matches",
      self.name,
      self.cases,
      self.cases - self.mismatches.len()
    );
    assert!(self.cases > 0, "{}: no cases", self.name);
    assert!(
      self.mismatches.is_empty(),
      "{}: {} of {} cases differ:\n{}",
      self.name,
      self.mismatches.len(),
      self.cases,
      self
        .mismatches
        .iter()
        .take(30)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
    );
  }
}

/// `outputs` of a case, resolving `{"same_as": i}`.
fn case_outputs(cases: &[Value], i: usize) -> &Value {
  let case = &cases[i];
  match case.get("same_as") {
    Some(j) => case_outputs(cases, j.as_u64().unwrap() as usize),
    None => &case["outputs"],
  }
}

fn run_dict_fixture(name: &str, inputs: Vec<String>) {
  let data = load(name);
  let mut t = Tally::new(name);
  for case in data["cases"].as_array().unwrap() {
    let c = &case["config"];
    let want: Vec<&str> = s(case, "outputs").split('\n').collect();
    assert_eq!(want.len(), inputs.len(), "{name}: output count");
    for (input, want) in inputs.iter().zip(want) {
      let got = encode(&run(c, input).unwrap_or_else(|e| panic!("{e} for {input:?}")));
      t.check(|| format!("{c} {input:?}"), got.as_str(), want);
    }
  }
  t.finish();
}

#[test]
fn chars() {
  let data = load("chars.json");
  let inputs: Vec<String> = s(&data, "inputs").chars().map(String::from).collect();
  // Every character of PINYIN_DICT, in code point order.
  let mut keys: Vec<u32> = PINYIN_DICT.keys().copied().collect();
  keys.sort_unstable();
  let expected: Vec<String> = keys
    .into_iter()
    .map(|k| char::from_u32(k).unwrap().to_string())
    .collect();
  assert_eq!(inputs, expected);
  run_dict_fixture("chars.json", inputs);
}

#[test]
fn phrases() {
  let data = load("phrases.json");
  let inputs: Vec<String> = s(&data, "inputs").split('\n').map(String::from).collect();
  assert!(inputs.len() >= 20000);
  run_dict_fixture("phrases.json", inputs);
}

#[test]
fn sentences() {
  let data = load("sentences.json");
  let configs = data["configs"].as_array().unwrap();
  let subset_configs = data["subset_configs"].as_array().unwrap();
  let subset_size = data["subset_size"].as_u64().unwrap() as usize;
  let mut t = Tally::new("sentences.json");
  for (i, case) in data["cases"].as_array().unwrap().iter().enumerate() {
    let text = s(case, "text");
    let mut all: Vec<&Value> = configs.iter().collect();
    if i < subset_size {
      all.extend(subset_configs);
    }
    let want = case["outputs"].as_array().unwrap();
    assert_eq!(want.len(), all.len());
    for (c, want) in all.into_iter().zip(want) {
      let got = encode_sep(&run(c, text));
      t.check(
        || format!("{c} {text:?}"),
        got.as_str(),
        want.as_str().unwrap(),
      );
    }
  }
  t.finish();
}

#[test]
fn syllables() {
  let data = load("syllables.json");
  let inputs: Vec<&str> = data["inputs"]
    .as_array()
    .unwrap()
    .iter()
    .map(|v| v.as_str().unwrap())
    .collect();
  let cases = data["cases"].as_array().unwrap();
  let mut t = Tally::new("syllables.json");
  for (i, case) in cases.iter().enumerate() {
    let c = &case["config"];
    let conv = UltimateConverter::new(b(c, "v_to_u"), b(c, "neutral_tone_with_five"));
    let st = style(s(c, "style"));
    let strict = b(c, "strict");
    let want: Vec<&str> = case_outputs(cases, i)
      .as_str()
      .unwrap()
      .split('\n')
      .collect();
    assert_eq!(want.len(), inputs.len());
    for (input, want) in inputs.iter().zip(want) {
      let got = match py_call(|| conv.convert_style("", input, st, strict)) {
        Ok(v) => v,
        Err(ty) => format!("!{ty}"),
      };
      t.check(|| format!("{c} {input:?}"), got.as_str(), want);
    }
  }
  t.finish();
}

/// Call a `pypinyin.contrib.tone_convert` function with Python keyword
/// arguments (missing ones take the Python defaults).
fn call_tone_convert(func: &str, kwargs: &Value, pinyin: &str) -> String {
  let kw = |k: &str, default: bool| kwargs.get(k).map_or(default, |v| v.as_bool().unwrap());
  let v = kw("v_to_u", false);
  let n5 = kw("neutral_tone_with_five", false);
  let strict = kw("strict", true);
  match func {
    "to_normal" => tc::to_normal(pinyin, v),
    "to_tone" => tc::to_tone(pinyin),
    "to_tone2" => tc::to_tone2(pinyin, v, n5),
    "to_tone3" => tc::to_tone3(pinyin, v, n5),
    "to_initials" => tc::to_initials(pinyin, strict),
    "to_finals" => tc::to_finals(pinyin, strict, v),
    "to_finals_tone" => tc::to_finals_tone(pinyin, strict),
    "to_finals_tone2" => tc::to_finals_tone2(pinyin, strict, v, n5),
    "to_finals_tone3" => tc::to_finals_tone3(pinyin, strict, v, n5),
    "tone_to_normal" => tc::tone_to_normal(pinyin, v),
    "tone_to_tone2" => tc::tone_to_tone2(pinyin, v, n5),
    "tone_to_tone3" => tc::tone_to_tone3(pinyin, v, n5),
    "tone2_to_normal" => tc::tone2_to_normal(pinyin, v),
    "tone2_to_tone" => tc::tone2_to_tone(pinyin),
    "tone2_to_tone3" => tc::tone2_to_tone3(pinyin, v),
    "tone3_to_normal" => tc::tone3_to_normal(pinyin, v),
    "tone3_to_tone" => tc::tone3_to_tone(pinyin),
    "tone3_to_tone2" => tc::tone3_to_tone2(pinyin, v),
    other => panic!("unknown tone_convert function {other}"),
  }
}

#[test]
fn tone_convert() {
  let data = load("tone_convert.json");
  for group in data.as_array().unwrap() {
    let name = s(group, "group");
    let inputs: Vec<String> = match &group["inputs"] {
      Value::String(joined) => joined.split('\n').map(String::from).collect(),
      Value::Array(list) => list
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect(),
      other => panic!("bad inputs {other}"),
    };
    let cases = group["cases"].as_array().unwrap();
    let mut t = Tally::new(&format!("tone_convert.json[{name}]"));
    for (i, case) in cases.iter().enumerate() {
      let func = s(case, "fn");
      let kwargs = &case["kwargs"];
      let want: Vec<String> = match case_outputs(cases, i) {
        Value::String(joined) => joined.split('\n').map(String::from).collect(),
        Value::Array(list) => list
          .iter()
          .map(|v| v.as_str().unwrap().to_string())
          .collect(),
        other => panic!("bad outputs {other}"),
      };
      assert_eq!(want.len(), inputs.len());
      for (input, want) in inputs.iter().zip(want) {
        let got = call_tone_convert(func, kwargs, input);
        t.check(|| format!("{func}({input:?}, {kwargs})"), got, want);
      }
    }
    t.finish();
  }
}

// --- edge cases: segmentation, list input, errors, subclassing ---------------

/// Port of `callable_errors` in the generator.
fn callable_errors(chars: &str) -> Option<NoPinyinResult> {
  let n = chars.chars().count();
  match n % 4 {
    0 => None,
    1 => Some(NoPinyinResult::Str(format!("<{chars}>"))),
    2 => Some(NoPinyinResult::List(
      chars.chars().map(|c| format!("#{c}")).collect(),
    )),
    _ => Some(NoPinyinResult::Nested(
      chars
        .chars()
        .map(|c| vec![format!("@{c}"), format!("&{c}")])
        .collect(),
    )),
  }
}

/// Python `str.isdigit()`. The generator asserts that it agrees with `\d`
/// on every character of the edge case texts.
fn py_isdigit(s: &str) -> bool {
  !s.is_empty() && s.chars().all(is_digit)
}

/// Port of `NoPinyinOverrideConverter` in the generator: a subclass of
/// `UltimateConverter` that overrides the no-pinyin and style hooks.
struct NoPinyinOverrideConverter {
  inner: UltimateConverter,
}

impl Converter for NoPinyinOverrideConverter {
  fn pre_handle_nopinyin(
    &self,
    chars: &str,
    _: Style,
    _: bool,
    _: &Errors,
    _: bool,
  ) -> Option<NoPinyinResult> {
    if chars.starts_with('!') {
      return Some(NoPinyinResult::List(vec![
        "bang".to_string();
        chars.chars().count()
      ]));
    }
    None
  }

  fn post_handle_nopinyin(
    &self,
    chars: &str,
    _: Style,
    _: bool,
    _: &Errors,
    _: bool,
    _: Option<&NoPinyinResult>,
  ) -> Option<NoPinyinResult> {
    if py_isdigit(chars) {
      return Some(NoPinyinResult::Nested(
        chars
          .chars()
          .map(|c| vec![format!("d{c}"), format!("D{c}")])
          .collect(),
      ));
    }
    None
  }

  fn post_pinyin(
    &self,
    han: &str,
    heteronym: bool,
    pinyin: &[Vec<String>],
  ) -> Option<Vec<Vec<String>>> {
    let data = self.inner.post_pinyin(han, heteronym, pinyin);
    if han == "中国" {
      return Some(vec![vec!["zhòng".to_string()], vec!["guó".to_string()]]);
    }
    data
  }

  fn pre_convert_style(&self, han: &str, _: &str, _: Style, _: bool) -> Option<String> {
    if han == "好" {
      return Some("hào".to_string());
    }
    None
  }

  fn post_convert_style(
    &self,
    han: &str,
    orig_pinyin: &str,
    converted_pinyin: &str,
    style: Style,
    strict: bool,
  ) -> Option<String> {
    let data = self
      .inner
      .post_convert_style(han, orig_pinyin, converted_pinyin, style, strict);
    if han == "吗" {
      return data.map(|d| d.to_uppercase());
    }
    data
  }

  fn is_ultimate_converter(&self) -> bool {
    true
  }
}

/// Port of `SimpleSegPinyin` in the generator (like GPT-SoVITS
/// `G2PWPinyin.get_seg`).
struct SimpleSegPinyin {
  converter: UltimateConverter,
}

impl PinyinMethods for SimpleSegPinyin {
  type Converter = UltimateConverter;

  fn converter(&self) -> &UltimateConverter {
    &self.converter
  }

  fn get_seg(&self) -> fn(&str) -> Vec<String> {
    |hans| simple_seg(hans)
  }
}

/// Port of `PrePostSegPinyin` in the generator.
struct PrePostSegPinyin {
  converter: UltimateConverter,
}

impl PinyinMethods for PrePostSegPinyin {
  type Converter = UltimateConverter;

  fn converter(&self) -> &UltimateConverter {
    &self.converter
  }

  fn pre_seg(&self, hans: &str) -> Option<Vec<String>> {
    let rest = hans.strip_prefix('#')?;
    Some(rest.chars().map(String::from).collect())
  }

  fn post_seg(&self, _hans: &str, seg_data: &[String]) -> Option<Vec<String>> {
    if seg_data.len() > 2 {
      return Some(seg_data.iter().rev().cloned().collect());
    }
    None
  }
}

/// Port of `fake_g2pw` in the generator: a deterministic stand-in for the
/// G2PW model.
fn fake_g2pw(han: &str) -> Vec<Vec<Option<String>>> {
  if han.chars().count().is_multiple_of(5) {
    return Vec::new();
  }
  let row = han
    .chars()
    .map(|c| {
      let cp = c as u32;
      match PINYIN_DICT.get(&cp) {
        Some(pys) if !cp.is_multiple_of(3) => Some(tc::to_tone3(
          pys.split(',').next_back().unwrap(),
          false,
          false,
        )),
        _ => None,
      }
    })
    .collect();
  vec![row]
}

/// Port of GPT-SoVITS `GPT_SoVITS/text/g2pw/g2pw.py` `Converter` (a subclass
/// of `UltimateConverter`) over [`fake_g2pw`]: it shows how the extension
/// point is expressed in Rust.
struct G2PWLikeConverter {
  inner: UltimateConverter,
  _g2pw: fn(&str) -> Vec<Vec<Option<String>>>,
}

impl G2PWLikeConverter {
  /// Port of `Converter._to_pinyin`.
  fn _to_pinyin(
    &self,
    han: &str,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    let mut pinyins = Vec::new();

    let g2pw_pinyin = (self._g2pw)(han);

    if g2pw_pinyin.is_empty() {
      // g2pw 不支持的汉字改为使用 pypinyin 原有逻辑
      return default_converter::convert(self, han, Style::TONE, heteronym, errors, strict);
    }

    let chars: Vec<char> = han.chars().collect();
    for (i, item) in g2pw_pinyin[0].iter().enumerate() {
      match item {
        None => {
          // g2pw 不支持的汉字改为使用 pypinyin 原有逻辑
          let py = default_converter::convert(
            self,
            &chars[i].to_string(),
            Style::TONE,
            heteronym,
            errors,
            strict,
          );
          pinyins.extend(py);
        }
        Some(item) => pinyins.push(vec![tc::to_tone(item)]),
      }
    }

    pinyins
  }
}

impl Converter for G2PWLikeConverter {
  fn convert(
    &self,
    words: &str,
    style: Style,
    heteronym: bool,
    errors: &Errors,
    strict: bool,
  ) -> Vec<Vec<String>> {
    let mut pys = Vec::new();
    if RE_HANS.is_match(words) {
      pys = self._to_pinyin(words, heteronym, errors, strict);
      if let Some(post_data) = self.post_pinyin(words, heteronym, &pys) {
        pys = post_data;
      }

      pys = self.convert_styles(pys, words, style, heteronym, errors, strict);
    } else {
      let py = self.handle_nopinyin(words, style, heteronym, errors, strict);
      if !py.is_empty() {
        pys.extend(py);
      }
    }

    _remove_dup_and_empty(&pys)
  }

  fn post_convert_style(
    &self,
    han: &str,
    orig_pinyin: &str,
    converted_pinyin: &str,
    style: Style,
    strict: bool,
  ) -> Option<String> {
    self
      .inner
      .post_convert_style(han, orig_pinyin, converted_pinyin, style, strict)
  }

  fn post_pinyin(
    &self,
    han: &str,
    heteronym: bool,
    pinyin: &[Vec<String>],
  ) -> Option<Vec<Vec<String>>> {
    self.inner.post_pinyin(han, heteronym, pinyin)
  }

  fn is_ultimate_converter(&self) -> bool {
    true
  }
}

/// Port of GPT-SoVITS `G2PWPinyin` over [`G2PWLikeConverter`].
struct G2PWLikePinyin {
  _converter: G2PWLikeConverter,
}

impl G2PWLikePinyin {
  fn new(v_to_u: bool, neutral_tone_with_five: bool) -> Self {
    Self {
      _converter: G2PWLikeConverter {
        inner: UltimateConverter::new(v_to_u, neutral_tone_with_five),
        _g2pw: fake_g2pw,
      },
    }
  }
}

impl PinyinMethods for G2PWLikePinyin {
  type Converter = G2PWLikeConverter;

  fn converter(&self) -> &G2PWLikeConverter {
    &self._converter
  }

  fn get_seg(&self) -> fn(&str) -> Vec<String> {
    |hans| simple_seg(hans)
  }
}

fn str_list(v: &Value) -> Vec<String> {
  v.as_array()
    .unwrap()
    .iter()
    .map(|x| x.as_str().unwrap().to_string())
    .collect()
}

fn nested(v: &Value) -> Vec<Vec<String>> {
  v.as_array().unwrap().iter().map(str_list).collect()
}

#[test]
fn edge_cases() {
  let data = load("edge_cases.json");
  let mut custom = PrefixSet::new();
  custom.train(["中国", "中国人", "中华人民", "人民", "我爱"]);
  let loose = Seg::new(custom.clone(), false);
  let strict_custom = Seg::new(custom, true);

  let mut t = Tally::new("edge_cases.json");
  for case in data.as_array().unwrap() {
    let kind = s(case, "kind");
    let args = &case["args"];
    let want = &case["result"];
    let what = || format!("{kind} {args}");
    let hans_list = || -> Vec<String> { str_list(&args["hans"]) };
    match kind {
      "simple_seg" => t.check(what, simple_seg(s(args, "hans")), str_list(want)),
      "seg" => t.check(what, simpleseg_seg(s(args, "hans")), str_list(want)),
      "mmseg" => t.check(what, mmseg::seg.cut(s(args, "hans")), str_list(want)),
      "re_hans" => t.check(
        what,
        RE_HANS.is_match(s(args, "hans")),
        want.as_bool().unwrap(),
      ),
      "mmseg_custom" => {
        let seg = if b(args, "no_non_phrases") {
          &strict_custom
        } else {
          &loose
        };
        t.check(what, seg.cut(s(args, "hans")), str_list(want));
      }
      "simple_seg_list" => {
        let list = hans_list();
        t.check(what, simple_seg(&list), str_list(want));
      }
      "pinyin_list" => {
        let list = hans_list();
        let st = style(s(args, "style"));
        let got = match s(args, "converter") {
          "ultimate" => pinyin(
            &list,
            &PinyinOptions {
              style: st,
              heteronym: true,
              neutral_tone_with_five: true,
              ..Default::default()
            },
          ),
          "default" => {
            Pinyin::new(DefaultConverter::new()).pinyin(&list, st, true, &Errors::Default, true)
          }
          other => panic!("unknown converter {other}"),
        };
        t.check(what, got, nested(want));
      }
      "lazy_pinyin_list" => {
        let list = hans_list();
        let got = lazy_pinyin(
          &list,
          &LazyPinyinOptions {
            style: style(s(args, "style")),
            neutral_tone_with_five: true,
            ..Default::default()
          },
        );
        t.check(what, got, str_list(want));
      }
      "callable_errors" => {
        let got = pinyin(
          s(args, "hans"),
          &PinyinOptions {
            style: Style::TONE3,
            heteronym: b(args, "heteronym"),
            errors: Errors::callable(callable_errors),
            ..Default::default()
          },
        );
        t.check(what, got, nested(want));
      }
      "hooks_converter" => {
        let p = Pinyin::new(NoPinyinOverrideConverter {
          inner: UltimateConverter::new(false, true),
        });
        let got = p.pinyin(s(args, "hans"), Style::TONE3, true, &Errors::Default, true);
        t.check(what, got, nested(want));
      }
      "simple_seg_pinyin" => {
        let p = SimpleSegPinyin {
          converter: UltimateConverter::new(false, true),
        };
        let got = p.pinyin(s(args, "hans"), Style::TONE3, false, &Errors::Default, true);
        t.check(what, got, nested(want));
      }
      "pre_post_seg_pinyin" => {
        let p = PrePostSegPinyin {
          converter: UltimateConverter::new(false, false),
        };
        let got = p.lazy_pinyin(s(args, "hans"), Style::TONE, &Errors::Default, true);
        t.check(what, got, str_list(want));
      }
      "g2pw_like" => {
        let p = G2PWLikePinyin::new(false, true);
        let got = p.pinyin(
          s(args, "hans"),
          style(s(args, "style")),
          false,
          &Errors::Default,
          true,
        );
        t.check(what, got, nested(want));
      }
      "default_converter" => {
        let p: Pinyin = Pinyin::default();
        let got = p.pinyin(s(args, "hans"), Style::TONE3, true, &Errors::Default, true);
        t.check(what, got, nested(want));
      }
      "replace_errors" => {
        let hans = s(args, "hans");
        let got = py_call(|| {
          pinyin(
            hans,
            &PinyinOptions {
              errors: Errors::Replace,
              ..Default::default()
            },
          )
        })
        .ok();
        let want = if want.is_null() {
          None
        } else {
          Some(nested(want))
        };
        t.check(what, got, want);
      }
      other => panic!("unknown edge case kind {other}"),
    }
  }
  t.finish();
}
