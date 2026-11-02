//! Golden parity tests against fixtures from
//! `parity/generate_fixtures.py`.

use cn2an::{An2Cn, Error, Transform};
use serde::Deserialize;

#[derive(Deserialize)]
struct TransformCase {
  i: String,
  o: String,
}

#[derive(Deserialize)]
struct An2CnCase {
  i: String,
  low: Option<String>,
  up: Option<String>,
  rmb: Option<String>,
  direct: Option<String>,
}

fn load<T: for<'de> Deserialize<'de>>(text: &str) -> Vec<T> {
  text
    .lines()
    .filter(|line| !line.is_empty())
    .map(|line| serde_json::from_str(line).expect("fixture line"))
    .collect()
}

fn check_transform(fixture: &str, direct: bool) {
  let transform = Transform::new();
  let cases: Vec<TransformCase> = load(fixture);
  assert!(!cases.is_empty());
  let mut mismatches = Vec::new();
  for case in &cases {
    let got = transform.transform(&case.i, "an2cn", direct).unwrap();
    if got != case.o {
      mismatches.push(format!("{:?}: got {:?}, want {:?}", case.i, got, case.o));
    }
  }
  assert!(
    mismatches.is_empty(),
    "{} of {} mismatches:\n{}",
    mismatches.len(),
    cases.len(),
    mismatches.join("\n")
  );
}

#[test]
fn transform_an2cn() {
  check_transform(include_str!("fixtures/transform_an2cn.jsonl"), false);
}

#[test]
fn transform_an2cn_direct() {
  check_transform(include_str!("fixtures/transform_an2cn_direct.jsonl"), true);
}

#[test]
fn an2cn_all_modes() {
  let ac = An2Cn::new();
  let cases: Vec<An2CnCase> = load(include_str!("fixtures/an2cn.jsonl"));
  assert!(!cases.is_empty());
  let mut mismatches = Vec::new();
  for case in &cases {
    for (mode, want) in [
      ("low", &case.low),
      ("up", &case.up),
      ("rmb", &case.rmb),
      ("direct", &case.direct),
    ] {
      let got = ac.an2cn(&case.i, mode).ok();
      if &got != want {
        mismatches.push(format!(
          "{:?} {mode}: got {:?}, want {:?}",
          case.i, got, want
        ));
      }
    }
  }
  assert!(
    mismatches.is_empty(),
    "{} mismatches:\n{}",
    mismatches.len(),
    mismatches.join("\n")
  );
}

#[test]
fn errors() {
  assert!(matches!(
    An2Cn::new().an2cn("1", "foo"),
    Err(Error::ValueError(_))
  ));
  assert!(matches!(
    cn2an::transform("1", "foo", false),
    Err(Error::ValueError(_))
  ));
  assert!(matches!(
    cn2an::transform("一", "cn2an", false),
    Err(Error::NotPorted(_))
  ));
  assert_eq!(cn2an::an2cn("123", "low").unwrap(), "一百二十三");
  assert_eq!(
    cn2an::transform("2023年5月1日", "an2cn", false).unwrap(),
    "二零二三年五月一日"
  );
}
