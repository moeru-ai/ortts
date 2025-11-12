mod utils;
pub use utils::inference;

#[cfg(test)]
mod tests {
  use crate::utils::inference;
  use ortts_shared::SpeechOptions;
  use std::fs;

  #[tokio::test]
  #[tracing_test::traced_test]
  async fn test_inference() {
    let output_file_name = "output.wav";
    let bytes = inference(SpeechOptions {
      input: String::from(
        "Hello, this is a test message for multilingual text-to-speech synthesis.",
      ),
      model: String::from("kokoro"),
      voice: String::from("af_alloy"),
    })
    .await
    .unwrap();

    fs::write(output_file_name, bytes).unwrap();
    tracing::info!("{} was successfully saved", output_file_name);
  }
}
