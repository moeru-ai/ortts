/// Helper function to process decimal numbers
pub fn point_num(match_str: &str) -> String {
  let parts: Vec<&str> = match_str.split('.').collect();

  if parts.len() != 2 {
    return match_str.to_string();
  }

  let a = parts[0];
  let b = parts[1];

  let b_formatted = b
    .chars()
    .map(|c| c.to_string())
    .collect::<Vec<String>>()
    .join(" ");

  format!("{a} point {b_formatted}")
}
