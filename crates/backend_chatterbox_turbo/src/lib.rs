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
      input: String::from("Oh, that's hilarious! [chuckle] Um anyway, how are you doing today?"),
      model: String::from("chatterbox-turbo"),
      voice: String::from("alloy"),
      stream_format: ortts_shared::StreamFormat::Audio,
    })
    .await
    .unwrap();

    fs::write(output_file_name, bytes).unwrap();
    tracing::info!("{} was successfully saved", output_file_name);
  }
}
