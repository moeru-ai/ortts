use ortts_shared::AppError;

use ort::session::Session;

pub fn create_session(model_path: &str) -> Result<Session, AppError> {
  use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
  };
  use ort::session::builder::GraphOptimizationLevel;

  let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_execution_providers([
      CUDAExecutionProvider::default().with_device_id(0).build(),
      CoreMLExecutionProvider::default().build(),
      DirectMLExecutionProvider::default()
        .with_device_id(0)
        .build(),
      CPUExecutionProvider::default().build(),
    ])?
    .commit_from_file(model_path)?;

  Ok(session)
}
