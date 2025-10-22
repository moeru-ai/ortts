pub mod routes;
mod utils;

#[cfg(test)]
mod tests {
  use std::fs;

  use crate::utils::inference;

  #[tokio::test]
  #[tracing_test::traced_test]
  async fn test_inference() {
    let text =
      String::from("[en]Hello, this is a test message for multilingual text-to-speech synthesis.");
    let output_file_name = "output.wav";
    let bytes = inference(text).await.unwrap();

    fs::write(output_file_name, bytes).unwrap();
    tracing::info!("{} was successfully saved", output_file_name);
  }
}
