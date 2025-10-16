use axum::{
  debug_handler,
  http::{Method, StatusCode, Uri},
};
use ortts_shared::AppError;

#[debug_handler]
pub async fn not_found(method: Method, uri: Uri) -> AppError {
  AppError::new(
    format!("Invalid URL ({} {})", method, uri.path()),
    String::from("invalid_request_error"),
    Some(StatusCode::NOT_FOUND),
    None,
    None,
  )
}
