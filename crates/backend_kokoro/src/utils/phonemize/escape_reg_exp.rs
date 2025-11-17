/// Escapes regular expression special characters from a string by replacing them with their escaped counterparts.
pub fn escape_reg_exp(string: &str) -> String {
  regex::escape(string)
}
