//! Port of `GPT_SoVITS/text/zh_normalization/num.py` (GPT-SoVITS @ 04a44eb).
//!
//! Rules to verbalize numbers into Chinese characters.
//! <https://zh.wikipedia.org/wiki/中文数字#現代中文>

use std::sync::LazyLock;

use super::Error;
use super::pyre::{PyMatch, PyRegex};

/// Port of `DIGITS` (`{str(i): tran for i, tran in enumerate("零一二三四五六七八九")}`).
pub const DIGITS: [(&str, &str); 10] = [
  ("0", "零"),
  ("1", "一"),
  ("2", "二"),
  ("3", "三"),
  ("4", "四"),
  ("5", "五"),
  ("6", "六"),
  ("7", "七"),
  ("8", "八"),
  ("9", "九"),
];

/// `DIGITS[key]`, raising `KeyError` for keys that are not `"0"`..`"9"`.
pub fn digits(key: &str) -> Result<&'static str, Error> {
  DIGITS
    .iter()
    .find(|(k, _)| *k == key)
    .map(|(_, v)| *v)
    .ok_or_else(|| Error::KeyError(key.to_string()))
}

fn digit_char(c: char) -> Result<&'static str, Error> {
  let mut buf = [0u8; 4];
  digits(c.encode_utf8(&mut buf))
}

/// Port of `UNITS` (an `OrderedDict` from power of ten to unit).
pub const UNITS: [(usize, &str); 5] = [(1, "十"), (2, "百"), (3, "千"), (4, "万"), (8, "亿")];

fn unit(power: usize) -> &'static str {
  UNITS
    .iter()
    .find(|(p, _)| *p == power)
    .map(|(_, u)| *u)
    .expect("unit")
}

/// Port of `COM_QUANTIFIERS`.
pub const COM_QUANTIFIERS: &str = "(处|台|架|枚|趟|幅|平|方|堵|间|床|株|批|项|例|列|篇|栋|注|亩|封|艘|把|目|套|段|人|所|朵|匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|毫|厘|(公)分|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|小时|旬|纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|百|十|)吨|(亿|千万|百万|万|千|百|)块|角|毛|分)";

/// Port of `RE_FRAC` (分数表达式).
pub static RE_FRAC: LazyLock<PyRegex> = LazyLock::new(|| PyRegex::compile(r"(-?)(\d+)/(\d+)"));

/// Port of `replace_frac`.
pub fn replace_frac(m: &PyMatch) -> Result<String, Error> {
  let sign = m.group(1);
  let nominator = m.group(2).unwrap_or_default();
  let denominator = m.group(3).unwrap_or_default();
  let sign = if sign.is_some_and(|s| !s.is_empty()) {
    "负"
  } else {
    ""
  };
  let nominator = num2str(nominator)?;
  let denominator = num2str(denominator)?;
  Ok(format!("{sign}{denominator}分之{nominator}"))
}

/// Port of `RE_PERCENTAGE` (百分数表达式).
pub static RE_PERCENTAGE: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(r"(-?)(\d+(\.\d+)?)%"));

/// Port of `replace_percentage`.
pub fn replace_percentage(m: &PyMatch) -> Result<String, Error> {
  let sign = m.group(1);
  let percent = m.group(2).unwrap_or_default();
  let sign = if sign.is_some_and(|s| !s.is_empty()) {
    "负"
  } else {
    ""
  };
  let percent = num2str(percent)?;
  Ok(format!("{sign}百分之{percent}"))
}

/// Port of `RE_INTEGER` (带负号的整数 -10).
pub static RE_INTEGER: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(concat!(r"(-)", r"(\d+)")));

/// Port of `replace_negative_num`.
pub fn replace_negative_num(m: &PyMatch) -> Result<String, Error> {
  let sign = m.group(1);
  let number = m.group(2).unwrap_or_default();
  let sign = if sign.is_some_and(|s| !s.is_empty()) {
    "负"
  } else {
    ""
  };
  let number = num2str(number)?;
  Ok(format!("{sign}{number}"))
}

/// Port of `RE_DEFAULT_NUM` (编号-无符号整形 00078).
pub static RE_DEFAULT_NUM: LazyLock<PyRegex> = LazyLock::new(|| PyRegex::compile(r"\d{3}\d*"));

/// Port of `replace_default_num`.
pub fn replace_default_num(m: &PyMatch) -> Result<String, Error> {
  let number = m.group0();
  verbalize_digit(number, true)
}

/// Port of `RE_ASMD` (加减乘除).
pub static RE_ASMD: LazyLock<PyRegex> = LazyLock::new(|| {
  PyRegex::compile(
    r"((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))([\+\-\×÷=])((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))",
  )
});

/// Port of `asmd_map`.
pub const ASMD_MAP: [(&str, &str); 5] = [
  ("+", "加"),
  ("-", "减"),
  ("×", "乘"),
  ("÷", "除"),
  ("=", "等于"),
];

/// Port of `replace_asmd`.
pub fn replace_asmd(m: &PyMatch) -> Result<String, Error> {
  let op = m.group(8).unwrap_or_default();
  let mapped = ASMD_MAP
    .iter()
    .find(|(k, _)| *k == op)
    .map(|(_, v)| *v)
    .ok_or_else(|| Error::KeyError(op.to_string()))?;
  Ok(format!(
    "{}{}{}",
    m.group(1).unwrap_or_default(),
    mapped,
    m.group(9).unwrap_or_default()
  ))
}

/// Port of `RE_POWER` (次方专项).
pub static RE_POWER: LazyLock<PyRegex> = LazyLock::new(|| PyRegex::compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]+"));

/// Port of `power_map`.
pub const POWER_MAP: [(char, &str); 13] = [
  ('⁰', "0"),
  ('¹', "1"),
  ('²', "2"),
  ('³', "3"),
  ('⁴', "4"),
  ('⁵', "5"),
  ('⁶', "6"),
  ('⁷', "7"),
  ('⁸', "8"),
  ('⁹', "9"),
  ('ˣ', "x"),
  ('ʸ', "y"),
  ('ⁿ', "n"),
];

/// Port of `replace_power`.
pub fn replace_power(m: &PyMatch) -> Result<String, Error> {
  let mut power_num = String::new();
  for c in m.group0().chars() {
    let mapped = POWER_MAP
      .iter()
      .find(|(k, _)| *k == c)
      .map(|(_, v)| *v)
      .ok_or_else(|| Error::KeyError(c.to_string()))?;
    power_num.push_str(mapped);
  }
  Ok(format!("的{power_num}次方"))
}

/// Port of `RE_DECIMAL_NUM` (纯小数).
pub static RE_DECIMAL_NUM: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(concat!(r"(-?)((\d+)(\.\d+))", r"|(\.(\d+))")));

/// Port of `RE_POSITIVE_QUANTIFIERS` (正整数 + 量词).
pub static RE_POSITIVE_QUANTIFIERS: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(&format!("{}{}", r"(\d+)([多余几\+])?", COM_QUANTIFIERS)));

/// Port of `RE_NUMBER`.
pub static RE_NUMBER: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(concat!(r"(-?)((\d+)(\.\d+)?)", r"|(\.(\d+))")));

/// Port of `replace_positive_quantifier`.
pub fn replace_positive_quantifier(m: &PyMatch) -> Result<String, Error> {
  let number = m.group(1).unwrap_or_default();
  let mut match_2 = m.group(2);
  if match_2 == Some("+") {
    match_2 = Some("多");
  }
  let match_2 = match_2.unwrap_or("");
  let quantifiers = m.group(3).unwrap_or_default();
  let mut number = num2str(number)?;
  if number == "二" {
    number = "两".to_string();
  }
  Ok(format!("{number}{match_2}{quantifiers}"))
}

/// Port of `replace_number`.
pub fn replace_number(m: &PyMatch) -> Result<String, Error> {
  let sign = m.group(1);
  let number = m.group(2);
  let pure_decimal = m.group(5);
  if let Some(pure_decimal) = pure_decimal.filter(|s| !s.is_empty()) {
    num2str(pure_decimal)
  } else {
    let sign = if sign.is_some_and(|s| !s.is_empty()) {
      "负"
    } else {
      ""
    };
    let number = num2str(number.unwrap_or_default())?;
    Ok(format!("{sign}{number}"))
  }
}

/// Port of `RE_RANGE` (范围表达式, compiled with `re.VERBOSE`).
pub static RE_RANGE: LazyLock<PyRegex> = LazyLock::new(|| {
  PyRegex::compile_verbose(
    r"
    (?<![\d\+\-\×÷=])      # 使用反向前瞻以确保数字范围之前没有其他数字和操作符
    ((-?)((\d+)(\.\d+)?))  # 匹配范围起始的负数或正数（整数或小数）
    [-~]                   # 匹配范围分隔符
    ((-?)((\d+)(\.\d+)?))  # 匹配范围结束的负数或正数（整数或小数）
    (?![\d\+\-\×÷=])       # 使用正向前瞻以确保数字范围之后没有其他数字和操作符
    ",
  )
});

/// Port of `replace_range`.
pub fn replace_range(m: &PyMatch) -> Result<String, Error> {
  let first = m.group(1).unwrap_or_default();
  let second = m.group(6).unwrap_or_default();
  let first = RE_NUMBER.sub(replace_number, first)?;
  let second = RE_NUMBER.sub(replace_number, second)?;
  Ok(format!("{first}到{second}"))
}

/// Port of `RE_TO_RANGE` (~至表达式).
pub static RE_TO_RANGE: LazyLock<PyRegex> = LazyLock::new(|| {
  PyRegex::compile(
    r"((-?)((\d+)(\.\d+)?)|(\.(\d+)))(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)[~]((-?)((\d+)(\.\d+)?)|(\.(\d+)))(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)",
  )
});

/// Port of `replace_to_range`.
pub fn replace_to_range(m: &PyMatch) -> Result<String, Error> {
  Ok(m.group0().replace('~', "至"))
}

/// Port of `RE_VERSION_NUM`.
pub static RE_VERSION_NUM: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(r"((\d+)(\.\d+)(\.\d+)?(\.\d+)+)"));

/// Port of `replace_vrsion_num` (upstream spelling).
pub fn replace_vrsion_num(m: &PyMatch) -> Result<String, Error> {
  let mut result = String::new();
  for c in m.group(1).unwrap_or_default().chars() {
    if c == '.' {
      result.push('点');
    } else {
      result.push_str(&num2str(c.encode_utf8(&mut [0u8; 4]))?);
    }
  }
  Ok(result)
}

fn lstrip_zero(s: &[char]) -> &[char] {
  let n = s.iter().take_while(|c| **c == '0').count();
  &s[n..]
}

fn get_value_chars(value_string: &[char], use_zero: bool) -> Result<Vec<&'static str>, Error> {
  let stripped = lstrip_zero(value_string);
  if stripped.is_empty() {
    Ok(vec![])
  } else if stripped.len() == 1 {
    if use_zero && stripped.len() < value_string.len() {
      Ok(vec![digits("0")?, digit_char(stripped[0])?])
    } else {
      Ok(vec![digit_char(stripped[0])?])
    }
  } else {
    let largest_unit = UNITS
      .iter()
      .rev()
      .map(|(p, _)| *p)
      .find(|p| *p < stripped.len())
      .expect("a unit smaller than the length");
    // Python slices `value_string` (with its leading zeros), not `stripped`.
    let split = value_string.len().saturating_sub(largest_unit);
    let first_part = &value_string[..split];
    let second_part = &value_string[split..];
    let mut out = get_value_chars(first_part, true)?;
    out.push(unit(largest_unit));
    out.extend(get_value_chars(second_part, true)?);
    Ok(out)
  }
}

/// Port of `_get_value`.
pub fn _get_value(value_string: &str, use_zero: bool) -> Result<Vec<&'static str>, Error> {
  let chars: Vec<char> = value_string.chars().collect();
  get_value_chars(&chars, use_zero)
}

/// Port of `verbalize_cardinal`.
pub fn verbalize_cardinal(value_string: &str) -> Result<String, Error> {
  if value_string.is_empty() {
    return Ok(String::new());
  }

  // 000 -> '零' , 0 -> '零'
  let value_string = value_string.trim_start_matches('0');
  if value_string.is_empty() {
    return Ok(digits("0")?.to_string());
  }

  let mut result_symbols = _get_value(value_string, true)?;
  // verbalized number starting with '一十*' is abbreviated as `十*`
  if result_symbols.len() >= 2 && result_symbols[0] == digits("1")? && result_symbols[1] == unit(1)
  {
    result_symbols.remove(0);
  }
  Ok(result_symbols.concat())
}

/// Port of `verbalize_digit`.
pub fn verbalize_digit(value_string: &str, alt_one: bool) -> Result<String, Error> {
  let mut result = String::new();
  for c in value_string.chars() {
    result.push_str(digit_char(c)?);
  }
  if alt_one {
    result = result.replace('一', "幺");
  }
  Ok(result)
}

/// Port of `num2str`.
pub fn num2str(value_string: &str) -> Result<String, Error> {
  let integer_decimal: Vec<&str> = value_string.split('.').collect();
  let (integer, decimal) = match integer_decimal.as_slice() {
    [integer] => (*integer, ""),
    [integer, decimal] => (*integer, *decimal),
    _ => {
      return Err(Error::ValueError(format!(
        "The value string: '${value_string}' has more than one point in it."
      )));
    }
  };

  let mut result = verbalize_cardinal(integer)?;

  let decimal = if decimal.ends_with('0') {
    format!("{}0", decimal.trim_end_matches('0'))
  } else {
    decimal.trim_end_matches('0').to_string()
  };

  if !decimal.is_empty() {
    // '.22' is verbalized as '零点二二'
    // '3.20' is verbalized as '三点二
    if result.is_empty() {
      result = "零".to_string();
    }
    result.push('点');
    result.push_str(&verbalize_digit(&decimal, false)?);
  }
  Ok(result)
}
