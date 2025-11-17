/// Helper function to format monetary values.
pub fn flip_money(match_str: &str) -> String {
  let currency_char = match_str.chars().next().unwrap_or(' '); // Get the first character
  let bill = if currency_char == '$' {
    "dollar"
  } else if currency_char == '£' {
    "pound"
  } else {
    return match_str.to_string();
  };

  let num_part = &match_str[currency_char.len_utf8()..];

  if num_part.parse::<f64>().is_err() {
    return format!("{} {}s", num_part, bill);
  }

  if !num_part.contains('.') {
    if num_part.is_empty() {
      return format!("0 {}s", bill);
    } else if num_part.parse::<u64>().map_or(false, |n| n == 1) {
      return format!("1 {}", bill);
    } else {
      return format!("{} {}s", num_part, bill);
    }
  }

  let parts: Vec<&str> = num_part.split('.').collect();
  let b = parts.get(0).unwrap_or(&"0");
  let c = parts.get(1).unwrap_or(&"0");

  let padded_c = format!("{c:0<2}");
  let d = padded_c.parse::<u64>().unwrap_or(0);

  let coins = if currency_char == '$' {
    if d == 1 { "cent" } else { "cents" }
  } else {
    if d == 1 { "penny" } else { "pence" }
  };

  let bill_suffix = if b == &"1" { "" } else { "s" };

  format!("{} {}{} and {} {}", b, bill, bill_suffix, d, coins)
}
