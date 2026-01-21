use std::{collections::HashSet, fs, path::Path};

use walkdir::WalkDir;

#[must_use]
pub fn dir_size(path: &Path) -> u64 {
  let mut total_size = 0;
  let mut seen_files = HashSet::new();

  for entry in WalkDir::new(path) {
    let Ok(entry) = entry else { continue };
    let path = entry.path();

    let real_path = if entry.path_is_symlink() {
      fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    } else {
      path.to_path_buf()
    };

    if !real_path.is_file() {
      continue;
    }

    if seen_files.insert(real_path.clone())
      && let Ok(meta) = fs::metadata(&real_path)
    {
      total_size += meta.len();
    }
  }

  total_size
}
