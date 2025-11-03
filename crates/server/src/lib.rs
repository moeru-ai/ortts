use axum::{Json, Router, routing::get};

mod openapi;
mod routes;

use openapi::ApiDoc;
use utoipa_scalar::{Scalar, Servable};

pub fn new() -> Router {
  let (router, api) = routes::new().split_for_parts();

  let openapi_json = api.clone();

  router
    .merge(Scalar::with_url("/", api))
    .route("/openapi.json", get(|| async move { Json(openapi_json) }))
    .fallback(routes::not_found)
}
