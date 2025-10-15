use axum::Router;

mod routes;

pub fn new() -> Router {
  Router::new().fallback(routes::not_found)
}
