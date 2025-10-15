use axum::Router;
use utoipa::OpenApi;
use utoipa_axum::routes as route;
use utoipa_axum::router::OpenApiRouter;

mod routes;
mod openapi;

use openapi::ApiDoc;
use utoipa_scalar::{Scalar, Servable};

pub fn new() -> Router {
  let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
    .routes(route!(routes::openapi_json::openapi_json))
    .split_for_parts();

  let router = router
    .merge(Scalar::with_url("/", api))
    .fallback(routes::not_found);

  router
}
