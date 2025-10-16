use axum::{Json, Router, routing::get};
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes as route;

mod openapi;
mod routes;

use openapi::ApiDoc;
use utoipa_scalar::{Scalar, Servable};

pub fn new() -> Router {
  let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
    .routes(route!(routes::api::speech))
    .split_for_parts();

  let openapi_json = api.clone();

  let router = router
    .merge(Scalar::with_url("/", api))
    .route("/openapi.json", get(|| async move { Json(openapi_json) }))
    .fallback(routes::not_found);

  router
}
