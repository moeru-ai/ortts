use ort::session::{Session, builder::GraphOptimizationLevel};
use ortts_shared::AppError;
use std::{
  collections::HashMap,
  ops::{Deref, DerefMut},
  path::PathBuf,
  sync::{Arc, Mutex, OnceLock, RwLock},
};

const MAX_SESSIONS_PER_MODEL: usize = 2;

#[derive(Debug)]
struct SessionPoolState {
  sessions: Mutex<Vec<Session>>,
  permits: Arc<tokio::sync::Semaphore>,
}

static SESSION_POOLS: OnceLock<RwLock<HashMap<PathBuf, Arc<SessionPoolState>>>> = OnceLock::new();

pub async fn inference_session(model_filepath: &PathBuf) -> Result<SessionPool, AppError> {
  let pool = session_pool(model_filepath);
  let permit = pool
    .permits
    .clone()
    .acquire_owned()
    .await
    .map_err(|error| semaphore_error(error.to_string()))?;
  session_with_permit(pool, model_filepath, permit)
}

pub fn inference_session_blocking(model_filepath: &PathBuf) -> Result<SessionPool, AppError> {
  let pool = session_pool(model_filepath);
  let permit = futures::executor::block_on(pool.permits.clone().acquire_owned())
    .map_err(|error| semaphore_error(error.to_string()))?;
  session_with_permit(pool, model_filepath, permit)
}

fn session_with_permit(
  pool: Arc<SessionPoolState>,
  model_filepath: &PathBuf,
  permit: tokio::sync::OwnedSemaphorePermit,
) -> Result<SessionPool, AppError> {
  let session = if let Some(session) = acquire_inference_session(&pool) {
    session
  } else {
    build_session(model_filepath)?
  };
  Ok(SessionPool::new(pool, session, permit))
}

fn semaphore_error(message: String) -> AppError {
  AppError::new(
    message,
    String::from("internal_server_error"),
    None,
    None,
    None,
  )
}

fn session_pool(model_filepath: &PathBuf) -> Arc<SessionPoolState> {
  let cache = SESSION_POOLS.get_or_init(Default::default);
  if let Ok(map) = cache.read()
    && let Some(pool) = map.get(model_filepath)
  {
    return pool.clone();
  }

  cache
    .write()
    .expect("session pool rw lock poisoned")
    .entry(model_filepath.clone())
    .or_insert_with(|| {
      Arc::new(SessionPoolState {
        sessions: Mutex::new(Vec::new()),
        permits: Arc::new(tokio::sync::Semaphore::new(MAX_SESSIONS_PER_MODEL)),
      })
    })
    .clone()
}

fn acquire_inference_session(pool: &Arc<SessionPoolState>) -> Option<Session> {
  let mut sessions = pool.sessions.lock().ok()?;
  sessions.pop()
}

fn build_session(model_filepath: &PathBuf) -> Result<Session, AppError> {
  let mut providers = Vec::new();

  #[cfg(feature = "ep_cuda")]
  {
    tracing::info!("CUDA Execution Provider is enabled.");
    providers.push(
      ort::execution_providers::CUDAExecutionProvider::default()
        .with_device_id(0)
        .build(),
    );
  }

  #[cfg(feature = "ep_coreml")]
  {
    tracing::info!("CoreML Execution Provider is enabled.");
    providers.push(ort::execution_providers::CoreMLExecutionProvider::default().build());
  }

  #[cfg(feature = "ep_directml")]
  {
    tracing::info!("DirectML Execution Provider is enabled.");
    providers.push(
      ort::execution_providers::DirectMLExecutionProvider::default()
        .with_device_id(0)
        .build(),
    )
  }

  #[cfg(feature = "ep_webgpu")]
  {
    tracing::info!("WebGPU Execution Provider is enabled.");
    providers.push(ort::execution_providers::WebGPUExecutionProvider::default().build());
  }

  providers.push(ort::execution_providers::CPUExecutionProvider::default().build());

  Ok(
    Session::builder()?
      .with_intra_threads(num_cpus::get())?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_execution_providers(providers)?
      .commit_from_file(model_filepath)?,
  )
}

#[derive(Debug)]
pub struct SessionPool {
  pool: Arc<SessionPoolState>,
  session: Option<Session>,
  _permit: tokio::sync::OwnedSemaphorePermit,
}

impl SessionPool {
  const fn new(
    pool: Arc<SessionPoolState>,
    session: Session,
    permit: tokio::sync::OwnedSemaphorePermit,
  ) -> Self {
    Self {
      pool,
      session: Some(session),
      _permit: permit,
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
      let mut sessions = self
        .pool
        .sessions
        .lock()
        .expect("session pool mutex poisoned");
      sessions.push(session);
    }
  }
}
