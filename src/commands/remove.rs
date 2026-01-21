use std::fs;

use ortts_shared::AppError;

use crate::AvailableModel;

pub fn remove(models: Vec<String>) -> Result<(), AppError> {
  let cache = hf_hub::Cache::from_env();
  let cache_path = cache.path();

  for model_id in models {
    match AvailableModel::from_model_name(&model_id) {
      Some(model) => {
        let model_path_name = format!("models--{}", model.hf_id().replace('/', "--"));
        let model_path = cache_path.join(model_path_name);

        if model_path.exists() {
          fs::remove_dir_all(&model_path)?;
          println!("deleted '{model_id}'");
        } else {
          panic!("model '{model_id}' not found")
        }
      }
      None => {
        panic!("model '{model_id}' not found")
      }
    }
  }

  Ok(())
}
