mod utils;
pub use utils::inference;

#[cfg(test)]
mod tests {
  use crate::utils::inference;
  use futures::StreamExt;
  use ortts_shared::SpeechOptions;

  #[tokio::test]
  async fn test_inference() {
    let mut speech_stream = inference(SpeechOptions {
      input: String::from(
        "[en]Hello, this is a test message for multilingual text-to-speech synthesis.",
      ),
      model: String::from("chatterbox-multilingual/en"),
      voice: String::from("alloy"),
      stream_format: None,
    })
    .await
    .unwrap();

    let mut sample_count = 0;
    while let Some(chunk) = speech_stream.next().await {
      sample_count += chunk.unwrap().len();
    }

    assert!(sample_count > 0);
  }
}
