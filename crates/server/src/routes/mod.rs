mod not_found;
pub use not_found::not_found;

use utoipa::OpenApi;
use utoipa_axum::{router::OpenApiRouter, routes};

use crate::ApiDoc;

pub mod api;

pub fn new() -> OpenApiRouter {
  OpenApiRouter::with_openapi(ApiDoc::openapi()).routes(routes!(api::speech))
}
