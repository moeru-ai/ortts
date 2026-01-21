use std::{fs, path::Path};

use chrono::{DateTime, Local};
use chrono_humanize::HumanTime;
use comfy_table::{ContentArrangement, Table, presets::NOTHING};
use humansize::{DECIMAL, format_size};
use ortts_shared::AppError;

use crate::AvailableModel;

use crate::dir_size;

struct ModelMetadata {
  commit: String,
  size: String,
  modified: String,
}

pub fn list() -> () {
  let mut table = Table::new();

  table
    .load_preset(NOTHING)
    .set_content_arrangement(ContentArrangement::Dynamic)
    .set_header(vec!["MODEL", "ID", "COMMIT", "SIZE", "MODIFIED"]);

  // remove left padding
  if let Some(column) = table.column_mut(0) {
    column.set_padding((0, 1));
  }

  let cache = hf_hub::Cache::from_env();
  let cache_path = cache.path();

  if !cache_path.exists() {
    eprintln!("Cache directory not found at {:?}", cache_path.display());
    return;
  }

  if let Ok(entries) = fs::read_dir(cache_path) {
    for entry in entries.flatten() {
      let folder_name = entry.file_name().to_string_lossy().to_string();

      if folder_name.starts_with("models--") {
        let id = folder_name
          .strip_prefix("models--")
          .unwrap()
          .replace("--", "/");

        if let Some(model) = AvailableModel::from_hf_id(&id)
          && let Ok(metadata) = get_model_metadata(&entry.path())
        {
          let model = model.model_name().to_owned();

          table.add_row(vec![
            model,
            id,
            metadata.commit,
            metadata.size,
            metadata.modified,
          ]);
        }
      }
    }
  }

  println!("{table}");
}

fn get_model_metadata(path: &Path) -> Result<ModelMetadata, AppError> {
  let ref_path = path.join("refs").join("main");

  if !ref_path.exists() {
    return Err(AppError::anyhow(&anyhow::anyhow!("No refs/main found")));
  }

  let commit_hash = fs::read_to_string(&ref_path)?.trim().to_string();

  let snapshot_path = path.join("snapshots").join(&commit_hash);

  if !snapshot_path.exists() {
    return Err(AppError::anyhow(&anyhow::anyhow!(
      "Snapshot directory missing"
    )));
  }

  let modified = fs::metadata(&snapshot_path)
    .and_then(|m| m.modified())
    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

  let dt: DateTime<Local> = modified.into();
  let ht = HumanTime::from(dt);

  Ok(ModelMetadata {
    commit: commit_hash[..7].to_string(),
    size: format_size(dir_size(&snapshot_path), DECIMAL),
    modified: ht.to_string(),
  })
}
