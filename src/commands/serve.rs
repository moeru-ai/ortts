use crate::cli::ServeArgs;

use axum::{http::{Method, StatusCode, Uri}, Router};
use ortts_shared::AppError;
use tokio::signal;

async fn shutdown_signal() {
  let ctrl_c = async {
    signal::ctrl_c()
      .await
      .expect("failed to install Ctrl+C handler");
  };

  #[cfg(unix)]
  let terminate = async {
    signal::unix::signal(signal::unix::SignalKind::terminate())
      .expect("failed to install signal handler")
      .recv()
      .await;
  };

  #[cfg(not(unix))]
  let terminate = std::future::pending::<()>();

  tokio::select! {
      _ = ctrl_c => {},
      _ = terminate => {},
  }
}

async fn handler_404(method: Method, uri: Uri) -> AppError {
  AppError::new(
    format!("Invalid URL ({} {})", method, uri.path()),
    String::from("invalid_request_error"),
    Some(StatusCode::NOT_FOUND),
    None,
    None,
  )
}

pub async fn serve(args: ServeArgs) -> Result<(), AppError> {
  tracing_subscriber::fmt::init();

  let app = Router::new().fallback(handler_404);

  let listener = tokio::net::TcpListener::bind(args.listen).await?;

  tracing::info!("listening on {}", listener.local_addr()?);

  axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal())
    .await?;

  Ok(())
}
