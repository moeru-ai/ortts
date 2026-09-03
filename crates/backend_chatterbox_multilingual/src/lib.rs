mod utils;
pub use utils::{inference, inference_stream};

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
        "[en]Hello, this is a test message for multilingual text-to-speech synthesis.",
      ),
      model: String::from("chatterbox-multilingual/en"),
      voice: String::from("alloy"),
      stream_format: None,
    })
    .await
    .unwrap();

    fs::write(output_file_name, bytes).unwrap();
    tracing::info!("{} was successfully saved", output_file_name);
  }
}
