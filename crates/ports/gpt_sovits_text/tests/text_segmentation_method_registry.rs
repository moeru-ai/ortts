//! `register_method` mutates the global `METHODS` registry, so this test runs
//! in its own test binary, apart from the parity tests that read the registry.

use gpt_sovits_text::tts_infer_pack::text_segmentation_method::{self as tsm, IndexError};

#[test]
fn register_method_replaces_in_place() {
  fn upper(inp: &str) -> Result<String, IndexError> {
    Ok(inp.to_uppercase())
  }
  // A new name is appended; an existing one keeps its position.
  tsm::register_method("test_upper", upper);
  assert_eq!(
    tsm::get_method("test_upper").unwrap()("ab"),
    Ok("AB".to_string())
  );
  tsm::register_method("test_upper", tsm::cut0);
  assert_eq!(
    tsm::get_method("test_upper").unwrap()("ab"),
    Ok("ab".to_string())
  );
  let names = tsm::get_method_names();
  assert_eq!(names[..6], ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]);
  assert_eq!(names.iter().filter(|n| *n == "test_upper").count(), 1);
}
