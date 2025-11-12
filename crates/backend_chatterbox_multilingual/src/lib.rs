mod utils;
pub use utils::inference;

#[cfg(test)]
mod tests {
  use std::fs;

  use ortts_shared::SpeechOptions;

  use crate::utils::inference;

  #[tokio::test]
  #[tracing_test::traced_test]
  async fn test_inference() {
    let output_file_name = "output.wav";
    let bytes = inference(SpeechOptions {
      input: String::from(
        "[en]Hello, this is a test message for multilingual text-to-speech synthesis.",
      ),
      model: String::from("chatterbox-multilingual/en"),
      voice: String::from("alloy"),
    })
    .await
    .unwrap();

    fs::write(output_file_name, bytes).unwrap();
    tracing::info!("{} was successfully saved", output_file_name);
  }
}
