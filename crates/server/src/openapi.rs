use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
  info(title = "ORTTS"),
  components(schemas(ortts_shared::AppError, ortts_shared::AppErrorWrapper,))
)]
pub struct ApiDoc;
