mod utils;
pub use utils::{inference, inference_stream};

#[cfg(test)]
mod tests {
  use crate::utils::{inference, inference_stream};
  use futures::StreamExt;
  use ortts_shared::SpeechOptions;
  use std::{fs, sync::Mutex};

  static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

  #[tokio::test]
  #[tracing_test::traced_test]
  async fn test_inference() {
    let _guard = INFERENCE_TEST_LOCK.lock().unwrap();
    let output_file_name = "output.wav";
    let bytes = inference(SpeechOptions {
      input: String::from(
        "Hello, this is a test message for multilingual text-to-speech synthesis.",
      ),
      model: String::from("kokoro"),
      voice: String::from("af_heart"),
      stream_format: None,
    })
    .await
    .unwrap();

    let reader = hound::WavReader::new(std::io::Cursor::new(&bytes)).unwrap();
    assert_eq!(reader.spec().channels, 2);
    assert_eq!(reader.spec().sample_rate, 24_000);
    assert_eq!(reader.duration(), 120_600);
    fs::write(output_file_name, bytes).unwrap();
    tracing::info!("{} was successfully saved", output_file_name);
  }

  #[tokio::test]
  async fn long_input_is_streamed_in_model_bounded_chunks() {
    let _guard = INFERENCE_TEST_LOCK.lock().unwrap();
    let input = (0..160).map(|_| "hello").collect::<Vec<_>>().join(" ");
    let mut speech_stream = inference_stream(SpeechOptions {
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
