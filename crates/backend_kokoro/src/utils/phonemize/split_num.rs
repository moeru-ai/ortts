/// Helper function to split numbers into phonetic equivalents
pub fn split_num(match_str: &str) -> String {
  if match_str.contains('.') {
    return match_str.to_string();
  } else if match_str.contains(':') {
    let parts: Vec<&str> = match_str.split(':').collect();
    if let (Ok(h), Ok(m)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
      if m == 0 {
        return format!("{h} o'clock");
      } else if m < 10 {
        return format!("{h} oh {m}");
      }
      return format!("{h} {m}");
    }
  }

  if match_str.len() >= 4 {
    let year_str = &match_str[..4];
    if let Ok(year) = year_str.parse::<u32>() {
      let suffix = if match_str.ends_with('s') { "s" } else { "" };

      if year < 1100 || year % 1000 < 10 {
        return match_str.to_string();
      }

      let left = &match_str[..2];
      let right_part = &match_str[2..4];
      if let Ok(right) = right_part.parse::<u32>() {
        if year % 1000 >= 100 && year % 1000 <= 999 {
          if right == 0 {
            return format!("{left} hundred{suffix}");
          } else if right < 10 {
            return format!("{left} oh {right}{suffix}");
          }
        }
        return format!("{left} {right}{suffix}");
      }
    }
  }

  match_str.to_string()
}
