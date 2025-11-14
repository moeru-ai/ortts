use ort::{
  execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, WebGPUExecutionProvider,
  },
  session::{Session, builder::GraphOptimizationLevel},
};
use ortts_shared::AppError;
use std::path::PathBuf;

pub fn inference_session(model_filepath: PathBuf) -> Result<Session, AppError> {
  #[cfg(feature = "ep_webgpu")]
  tracing::info!("WebGPU Execution Provider is enabled.");

  #[cfg(feature = "ep_cuda")]
  tracing::info!("CUDA Execution Provider is enabled.");

  #[cfg(feature = "ep_coreml")]
  tracing::info!("CoreML Execution Provider is enabled.");

  #[cfg(feature = "ep_directml")]
  tracing::info!("DirectML Execution Provider is enabled.");

  Ok(
    Session::builder()?
      .with_intra_threads(num_cpus::get())?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_execution_providers([
        CUDAExecutionProvider::default().with_device_id(0).build(),
        CoreMLExecutionProvider::default().build(),
        DirectMLExecutionProvider::default()
          .with_device_id(0)
          .build(),
        WebGPUExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(),
      ])?
      .commit_from_file(model_filepath)?,
  )
}
