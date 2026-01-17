use std::{collections::HashSet, fs, path::PathBuf};

use walkdir::WalkDir;

pub fn dir_size(path: &PathBuf) -> u64 {
  let mut total_size = 0;
  let mut seen_files = HashSet::new(); // 简单去重

  for entry in WalkDir::new(path) {
    let entry = match entry {
      Ok(e) => e,
      Err(_) => continue,
    };
    let path = entry.path();

    let real_path = if entry.path_is_symlink() {
      fs::canonicalize(path).unwrap_or(path.to_path_buf())
    } else {
      path.to_path_buf()
    };

    if !real_path.is_file() {
      continue;
    }

    // 使用 file_id 库是最稳健的，这里演示如果不想加太多依赖，
    // 可以用 canonicalize 后的路径字符串做 Key (在 Unix 上通常有效)
    if seen_files.insert(real_path.clone()) {
      if let Ok(meta) = fs::metadata(&real_path) {
        total_size += meta.len();
      }
    }
  }

  total_size
}
