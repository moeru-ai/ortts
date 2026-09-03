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
      input: String::from("Oh, that's hilarious! [chuckle] Um anyway, how are you doing today?"),
      model: String::from("chatterbox-turbo"),
      voice: String::from("alloy"),
      stream_format: None,
    })
    .await
    .unwrap();

    let spec = speech_stream.spec();
    let mut sample_count = 0;
    while let Some(chunk) = speech_stream.next().await {
      sample_count += chunk.unwrap().len();
    }

    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, 24_000);
    assert_eq!(sample_count, 106_560);
  }
}
