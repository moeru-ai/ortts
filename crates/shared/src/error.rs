use axum::{
  Json,
  http::StatusCode,
  response::{IntoResponse, Response},
};
use serde::Serialize;
use serde_json::Value;
use utoipa::ToSchema;

#[derive(Debug, Serialize, ToSchema)]
pub struct AppError {
  pub message: String,
  #[serde(rename = "type")]
  pub kind: String,
  pub param: Option<Value>,
  pub code: Option<String>,
  #[serde(skip)]
  pub status: StatusCode,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AppErrorWrapper {
  pub error: AppError,
}

impl AppError {
  #[must_use]
  pub fn new(
    message: String,
    kind: String,
    status: Option<StatusCode>,
    param: Option<Value>,
    code: Option<String>,
  ) -> Self {
    Self {
      message,
      kind,
      status: status.unwrap_or_else(|| StatusCode::INTERNAL_SERVER_ERROR),
      param,
      code,
    }
  }

  #[must_use]
  pub fn anyhow(error: &anyhow::Error) -> Self {
    Self::new(
      error.to_string(),
      String::from("internal_server_error"),
      None,
      None,
      None,
    )
  }
}

impl IntoResponse for AppError {
  fn into_response(self) -> Response {
    (self.status, Json(AppErrorWrapper { error: self })).into_response()
  }
}

impl<T> From<T> for AppError
where
  T: Into<anyhow::Error>,
{
  fn from(t: T) -> Self {
    Self::anyhow(&t.into())
  }
}
