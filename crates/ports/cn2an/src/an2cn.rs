//! Port of `cn2an/an2cn.py`.

use crate::conf::{NUMBER_LOW_AN2CN, NUMBER_UP_AN2CN, UNIT_LOW_ORDER_AN2CN, UNIT_UP_ORDER_AN2CN};
use crate::proces::preprocess::preprocess;
use crate::{Error, Result};

fn value_error(msg: impl Into<String>) -> Error {
  Error::ValueError(msg.into())
}

/// Python `int(d)` for a one-character string inside An2Cn. After
/// `__check_inputs_is_valid` the character is an ASCII digit, `.` or `-`;
/// only the digits convert.
fn py_int_char(d: char) -> Result<usize> {
  d.to_digit(10)
    .filter(|_| d.is_ascii_digit())
    .map(|v| v as usize)
    .ok_or_else(|| value_error(format!("invalid literal for int() with base 10: '{d}'")))
}

/// Python `str(int(s))` for the strings `__integer_convert` receives, which
/// only contain ASCII digits and `-` (validated by `__check_inputs_is_valid`,
/// and split on `.`).
fn py_int_str(s: &str) -> Result<String> {
  let (negative, body) = match s.strip_prefix('-') {
    Some(rest) => (true, rest),
    None => (false, s.strip_prefix('+').unwrap_or(s)),
  };
  if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit()) {
    return Err(value_error(format!(
      "invalid literal for int() with base 10: '{s}'"
    )));
  }
  let trimmed = body.trim_start_matches('0');
  if trimmed.is_empty() {
    Ok("0".to_owned())
  } else if negative {
    Ok(format!("-{trimmed}"))
  } else {
    Ok(trimmed.to_owned())
  }
}

/// Port of `cn2an.an2cn.An2Cn`.
#[derive(Debug, Clone)]
pub struct An2Cn {
  /// Port of `An2Cn.all_num`.
  pub all_num: &'static str,
  /// Port of `An2Cn.number_low`.
  pub number_low: [char; 10],
  /// Port of `An2Cn.number_up`.
  pub number_up: [char; 10],
  /// Port of `An2Cn.mode_list`.
  pub mode_list: [&'static str; 4],
}

impl Default for An2Cn {
  fn default() -> Self {
    Self::new()
  }
}

impl An2Cn {
  /// Port of `An2Cn.__init__`.
  pub fn new() -> Self {
    Self {
      all_num: "0123456789",
      number_low: NUMBER_LOW_AN2CN,
      number_up: NUMBER_UP_AN2CN,
      mode_list: ["low", "up", "rmb", "direct"],
    }
  }

  /// Port of `An2Cn.an2cn` for `str` inputs.
  ///
  /// `mode` is one of `"low"`, `"up"`, `"rmb"` and `"direct"`. The Python
  /// `None`/`int`/`float` inputs are not ported; an empty string returns
  /// the same error as `None`.
  pub fn an2cn(&self, inputs: &str, mode: &str) -> Result<String> {
    if inputs.is_empty() {
      return Err(value_error("输入数据为空！"));
    }
    if !self.mode_list.contains(&mode) {
      return Err(value_error(format!("mode 仅支持 {:?} ！", self.mode_list)));
    }

    // 数据预处理：繁体转简体，全角转半角
    let inputs = preprocess(
      inputs,
      &["traditional_to_simplified", "full_angle_to_half_angle"],
    )?;

    // 检查数据是否有效
    self.check_inputs_is_valid(&inputs)?;

    // 判断正负
    let (sign, inputs) = match inputs.strip_prefix('-') {
      Some(rest) => ("负", rest),
      None => ("", inputs.as_str()),
    };

    let output = if mode == "direct" {
      self.direct_convert(inputs)?
    } else {
      // 切割整数部分和小数部分
      let split_result: Vec<&str> = inputs.split('.').collect();
      match split_result.as_slice() {
        [integer_data] => {
          if mode == "rmb" {
            self.integer_convert(integer_data, "up")? + "元整"
          } else {
            self.integer_convert(integer_data, mode)?
          }
        }
        [integer_data, decimal_data] => {
          if mode == "rmb" {
            Self::rmb_convert(
              self.integer_convert(integer_data, "up")?,
              self.decimal_convert(decimal_data, "up")?,
            )?
          } else {
            self.integer_convert(integer_data, mode)? + &self.decimal_convert(decimal_data, mode)?
          }
        }
        _ => return Err(value_error(format!("输入格式错误：{inputs}！"))),
      }
    };

    Ok(format!("{sign}{output}"))
  }

  /// The `mode == "rmb"` branch of `An2Cn.an2cn` for inputs with a decimal
  /// point.
  fn rmb_convert(int_data: String, dec_data: String) -> Result<String> {
    let dec: Vec<char> = dec_data.chars().collect();
    let int_zero = int_data == "零";
    let output = match dec.len() {
      0 => int_data + "元整",
      1 => return Err(value_error(format!("异常输出：{dec_data}"))),
      2 => {
        if dec[1] != '零' {
          if int_zero {
            format!("{}角", dec[1])
          } else {
            format!("{int_data}元{}角", dec[1])
          }
        } else {
          int_data + "元整"
        }
      }
      _ => {
        if dec[1] != '零' {
          if dec[2] != '零' {
            if int_zero {
              format!("{}角{}分", dec[1], dec[2])
            } else {
              format!("{int_data}元{}角{}分", dec[1], dec[2])
            }
          } else if int_zero {
            format!("{}角", dec[1])
          } else {
            format!("{int_data}元{}角", dec[1])
          }
        } else if dec[2] != '零' {
          if int_zero {
            format!("{}分", dec[2])
          } else {
            format!("{int_data}元零{}分", dec[2])
          }
        } else {
          int_data + "元整"
        }
      }
    };
    Ok(output)
  }

  /// Port of `An2Cn.__direct_convert`.
  fn direct_convert(&self, inputs: &str) -> Result<String> {
    let mut output = String::new();
    for d in inputs.chars() {
      if d == '.' {
        output.push('点');
      } else {
        output.push(self.number_low[py_int_char(d)?]);
      }
    }
    Ok(output)
  }

  /// Port of `An2Cn.__check_inputs_is_valid`.
  fn check_inputs_is_valid(&self, check_data: &str) -> Result<()> {
    for data in check_data.chars() {
      if !self.all_num.contains(data) && data != '.' && data != '-' {
        return Err(value_error(format!("输入的数据不在转化范围内：{data}！")));
      }
    }
    Ok(())
  }

  /// Port of `An2Cn.__integer_convert`.
  fn integer_convert(&self, integer_data: &str, mode: &str) -> Result<String> {
    let (numeral_list, unit_list) = match mode {
      "low" => (&NUMBER_LOW_AN2CN, &UNIT_LOW_ORDER_AN2CN),
      "up" => (&NUMBER_UP_AN2CN, &UNIT_UP_ORDER_AN2CN),
      _ => return Err(value_error(format!("error mode: {mode}"))),
    };

    // 去除前面的 0，比如 007 => 7
    let integer_data: Vec<char> = py_int_str(integer_data)?.chars().collect();

    let len_integer_data = integer_data.len();
    if len_integer_data > unit_list.len() {
      return Err(value_error(format!(
        "超出数据范围，最长支持 {} 位",
        unit_list.len()
      )));
    }

    let mut output_an = String::new();
    for (i, &d) in integer_data.iter().enumerate() {
      let d = py_int_char(d)?;
      let unit = unit_list[len_integer_data - i - 1];
      if d != 0 {
        output_an.push(numeral_list[d]);
        output_an.push_str(unit);
      } else {
        if (len_integer_data - i - 1).is_multiple_of(4) {
          output_an.push(numeral_list[d]);
          output_an.push_str(unit);
        }

        if i > 0 {
          // Python indexes `output_an[-1]`, which raises IndexError
          // on an empty string.
          let last = output_an
            .chars()
            .last()
            .ok_or_else(|| value_error("string index out of range"))?;
          if last != '零' {
            output_an.push(numeral_list[d]);
          }
        }
      }
    }

    let output_an = output_an
      .replace("零零", "零")
      .replace("零万", "万")
      .replace("零亿", "亿")
      .replace("亿万", "亿");
    let output_an = output_an.trim_matches('零');
    let mut output_an = sub_big_unit_zero_thousand(output_an);

    // 解决「一十几」问题
    if output_an.starts_with("一十") {
      output_an.remove(0);
    }

    // 0 - 1 之间的小数
    if output_an.is_empty() {
      output_an.push('零');
    }

    Ok(output_an)
  }

  /// Port of `An2Cn.__decimal_convert`.
  fn decimal_convert(&self, decimal_data: &str, o_mode: &str) -> Result<String> {
    let len_decimal_data = decimal_data.chars().count();

    // Python warns that only the first 16 digits are kept.
    let decimal_data: String = decimal_data.chars().take(16).collect();

    let mut output_an = String::new();
    if len_decimal_data > 0 {
      output_an.push('点');
    }

    let numeral_list = match o_mode {
      "low" => &NUMBER_LOW_AN2CN,
      "up" => &NUMBER_UP_AN2CN,
      _ => return Err(value_error(format!("error mode: {o_mode}"))),
    };

    for data in decimal_data.chars() {
      output_an.push(numeral_list[py_int_char(data)?]);
    }
    Ok(output_an)
  }
}

/// `re.sub(r"([万亿])零([一二三四五六七八九壹贰叁肆伍陆柒捌玖][千仟])", r"\1\2", s)`
/// from `An2Cn.__integer_convert`.
fn sub_big_unit_zero_thousand(s: &str) -> String {
  const DIGITS: &str = "一二三四五六七八九壹贰叁肆伍陆柒捌玖";
  let chars: Vec<char> = s.chars().collect();
  let mut out = String::new();
  let mut i = 0;
  while i < chars.len() {
    if i + 3 < chars.len()
      && matches!(chars[i], '万' | '亿')
      && chars[i + 1] == '零'
      && DIGITS.contains(chars[i + 2])
      && matches!(chars[i + 3], '千' | '仟')
    {
      out.push(chars[i]);
      out.push(chars[i + 2]);
      out.push(chars[i + 3]);
      i += 4;
    } else {
      out.push(chars[i]);
      i += 1;
    }
  }
  out
}
