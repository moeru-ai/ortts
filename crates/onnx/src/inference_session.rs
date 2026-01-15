use ort::{
  execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, WebGPUExecutionProvider,
  },
  session::{Session, builder::GraphOptimizationLevel},
};
use ortts_shared::AppError;
use std::{
  collections::HashMap,
  ops::{Deref, DerefMut},
  path::PathBuf,
  sync::{Arc, Mutex, OnceLock},
};

// TODO(@nekomeowww,@sumimakito): The current session pool implementation is way too simple, but it works,
// for now. The possible approach to improve involves different strategies where memory aware, request queuing
// (should be implemented on request handler side instead of here), eviction policies, etc.
//
// Our current implementation is more like this one:
// https://github.com/pykeio/ort/issues/469
static SESSION_POOLS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<Vec<Session>>>>>> = OnceLock::new();

pub fn inference_session(model_filepath: PathBuf) -> Result<SessionPool, AppError> {
  let pool = session_pool(&model_filepath);
  if let Some(session) = acquire_inference_session(&pool) {
    return Ok(SessionPool::new(pool, session));
  }

  let session = build_session(&model_filepath)?;
  Ok(SessionPool::new(pool, session))
}

fn session_pool(model_filepath: &PathBuf) -> Arc<Mutex<Vec<Session>>> {
  let cache = SESSION_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
  let mut pools = cache.lock().expect("session pool mutex poisoned");
  pools
    .entry(model_filepath.clone())
    .or_insert_with(|| Arc::new(Mutex::new(Vec::new())))
    .clone()
}

fn acquire_inference_session(pool: &Arc<Mutex<Vec<Session>>>) -> Option<Session> {
  let mut sessions = pool.lock().expect("session pool mutex poisoned");
  sessions.pop()
}

fn build_session(model_filepath: &PathBuf) -> Result<Session, AppError> {
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

#[derive(Debug)]
pub struct SessionPool {
  pool: Arc<Mutex<Vec<Session>>>,
  session: Option<Session>,
}

impl SessionPool {
  fn new(pool: Arc<Mutex<Vec<Session>>>, session: Session) -> Self {
    Self {
      pool,
      session: Some(session),
    }
  }
}

impl Deref for SessionPool {
  type Target = Session;

  fn deref(&self) -> &Self::Target {
    self
      .session
      .as_ref()
      .expect("session should be present while guard is alive")
  }
}

impl DerefMut for SessionPool {
  fn deref_mut(&mut self) -> &mut Self::Target {
    self
      .session
      .as_mut()
      .expect("session should be present while guard is alive")
  }
}

impl Drop for SessionPool {
  fn drop(&mut self) {
    if let Some(session) = self.session.take() {
      let mut sessions = self.pool.lock().expect("session pool mutex poisoned");
      sessions.push(session);
    }
  }
}
