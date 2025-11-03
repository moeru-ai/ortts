use crate::cli::ServeArgs;

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
    () = ctrl_c => {},
    () = terminate => {},
  }
}

pub async fn serve(args: ServeArgs) -> Result<(), AppError> {
  tracing_subscriber::fmt::init();

  let app = ortts_server::new();

  let listener = tokio::net::TcpListener::bind(args.listen).await?;

  tracing::info!("listening on {}", listener.local_addr()?);

  axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal())
    .await?;

  Ok(())
}
