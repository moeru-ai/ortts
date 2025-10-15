use axum::{Json, debug_handler};
use utoipa::OpenApi;

use crate::openapi::ApiDoc;

/// Return JSON version of an OpenAPI schema
#[utoipa::path(
  get,
  path = "/openapi.json",
  responses(
  (status = 200, description = "JSON file", body = ())
  )
)]
#[debug_handler]
pub async fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
  Json(ApiDoc::openapi())
}
