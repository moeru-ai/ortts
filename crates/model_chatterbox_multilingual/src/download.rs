use std::path::PathBuf;

use ortts_shared::AppError;

pub async fn get_cangjie_file() -> Result<PathBuf, AppError> {
  let api = hf_hub::api::tokio::Api::new()?;

  let path = api.model(String::from("onnx-community/chatterbox-multilingual-ONNX"))
    .download("Cangjie5_TC.json")
    .await?;

  Ok(path)
}
