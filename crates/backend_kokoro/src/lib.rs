mod utils;
pub use utils::inference;

#[cfg(test)]
mod tests {
  use crate::utils::inference;
  use futures::StreamExt;
  use ortts_shared::SpeechOptions;
  use std::sync::Mutex;

  static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

  #[tokio::test]
  async fn test_inference() {
    let _guard = INFERENCE_TEST_LOCK.lock().unwrap();
    let mut speech_stream = inference(SpeechOptions {
      input: String::from(
        "Hello, this is a test message for multilingual text-to-speech synthesis.",
      ),
      model: String::from("kokoro"),
      voice: String::from("af_heart"),
      stream_format: None,
    })
    .await
    .unwrap();

    let spec = speech_stream.spec();
    let mut sample_count = 0;
    while let Some(chunk) = speech_stream.next().await {
      sample_count += chunk.unwrap().len();
    }

    assert_eq!(spec.channels, 2);
    assert_eq!(spec.sample_rate, 24_000);
    assert_eq!(sample_count, 120_600 * usize::from(spec.channels));
  }

  #[tokio::test]
  async fn long_input_is_streamed_in_model_bounded_chunks() {
    let _guard = INFERENCE_TEST_LOCK.lock().unwrap();
    let input = (0..160).map(|_| "hello").collect::<Vec<_>>().join(" ");
    let mut speech_stream = inference(SpeechOptions {
      input,
      model: String::from("kokoro"),
      voice: String::from("af_heart"),
      stream_format: Some(ortts_shared::StreamFormat::Sse),
    })
    .await
    .unwrap();

    assert_eq!(speech_stream.spec().channels, 2);
    assert_eq!(speech_stream.spec().sample_rate, 24_000);

    let mut chunk_count = 0;
    while let Some(chunk) = speech_stream.next().await {
      assert!(!chunk.unwrap().is_empty());
      chunk_count += 1;
    }
    assert!(chunk_count > 1);
  }
}
