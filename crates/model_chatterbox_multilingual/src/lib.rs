mod utils;

#[cfg(test)]
mod tests {
  use crate::utils::inference;

  #[tokio::test]
  #[tracing_test::traced_test]
  async fn test_inference() {
    inference().await;
  }
}
