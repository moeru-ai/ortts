use std::path::PathBuf;

use ort::{
  execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
  },
  session::{Session, builder::GraphOptimizationLevel},
};
use ortts_shared::AppError;

pub fn inference_session(model_filepath: PathBuf) -> Result<Session, AppError> {
  Ok(
    Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(4)?
      .with_execution_providers([
        CUDAExecutionProvider::default().with_device_id(0).build(),
        CoreMLExecutionProvider::default().build(),
        DirectMLExecutionProvider::default()
          .with_device_id(0)
          .build(),
        CPUExecutionProvider::default().build(),
      ])?
      .commit_from_file(model_filepath)?,
  )
}
