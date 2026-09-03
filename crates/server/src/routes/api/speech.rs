use axum::{
  Json,
  body::Body,
  debug_handler,
  extract::rejection::JsonRejection,
  http::{StatusCode, header},
  response::{
    IntoResponse, Response,
    sse::{Event, KeepAlive, Sse},
  },
};
use futures::{StreamExt, stream};
use ortts_shared::{
  AppError, SpeechAudio, SpeechAudioDeltaEvent, SpeechAudioDoneEvent, SpeechAudioStream,
  SpeechAudioStreamEvent, SpeechOptions, StreamFormat, pcm_bytes, wav_header,
};
use serde_json::json;
use tracing::error;

async fn inference(options: SpeechOptions) -> Result<SpeechAudioStream, AppError> {
  match options.model.to_lowercase() {
    m if m.starts_with("chatterbox-multilingual") => {
      ortts_backend_chatterbox_multilingual::inference(options).await
    }
    m if m.starts_with("chatterbox-turbo") => {
      ortts_backend_chatterbox_turbo::inference(options).await
    }
    m if m.starts_with("kokoro") => ortts_backend_kokoro::inference(options).await,
    model => Err(AppError::new(
      format!("Model `{model}` is not supported"),
      String::from("invalid_request_error"),
      Some(StatusCode::BAD_REQUEST),
      Some(json!("model")),
      None,
    )),
  }
}

fn sse_response(speech_stream: SpeechAudioStream) -> Response {
  let events = stream::try_unfold(Some((speech_stream, false)), |state| async move {
    let Some((mut speech_stream, mut header_sent)) = state else {
      return Ok(None);
    };

    match speech_stream.next().await {
      Some(Ok(samples)) => {
        let mut bytes = pcm_bytes(&samples);
        if !header_sent {
          let mut header = wav_header(speech_stream.spec());
          header.append(&mut bytes);
          bytes = header;
          header_sent = true;
        }

        Ok(Some((
          Event::default()
            .json_data(SpeechAudioDeltaEvent::new(&bytes))
            .expect("speech SSE event serialization cannot fail"),
          Some((speech_stream, header_sent)),
        )))
      }
      Some(Err(error)) => {
        error!(message = %error.message, "speech inference stream failed");
        Err(std::io::Error::other(error.message))
      }
      None => Ok(Some((
        Event::default()
          .json_data(SpeechAudioDoneEvent::new())
          .expect("speech SSE event serialization cannot fail"),
        None,
      ))),
    }
  });

  Sse::new(events)
    .keep_alive(KeepAlive::default())
    .into_response()
}

fn audio_response(speech_stream: SpeechAudioStream) -> Response {
  // Preserve the WAV metadata in the stream, but use a generic binary MIME
  // type because the final RIFF length is unknown while the body is generated.
  let header = wav_header(speech_stream.spec());
  let audio_stream = stream::once(async move { Ok::<_, std::io::Error>(header) }).chain(
    speech_stream.map(|chunk| match chunk {
      Ok(samples) => Ok(pcm_bytes(&samples)),
      Err(error) => {
        error!(message = %error.message, "speech inference stream failed");
        Err(std::io::Error::other(error.message))
      }
    }),
  );

  (
    StatusCode::OK,
    [(header::CONTENT_TYPE, "application/octet-stream")],
    Body::from_stream(audio_stream),
  )
    .into_response()
}

/// Create speech
///
/// Generates audio from the input text.
#[utoipa::path(
  post,
  path = "/v1/audio/speech",
  request_body = SpeechOptions,
  responses(
    (
      status = 200,
      content(
        (SpeechAudio = "application/octet-stream"),
        (SpeechAudioStreamEvent = "text/event-stream")
      )
    )
  )
)]
#[debug_handler]
pub async fn speech(
  payload: Result<Json<SpeechOptions>, JsonRejection>,
) -> Result<Response, AppError> {
  let Json(options) = payload.map_err(|rejection| {
    AppError::new(
      rejection.body_text(),
      String::from("invalid_request_error"),
      Some(StatusCode::BAD_REQUEST),
      Some(json!("body")),
      None,
    )
  })?;
  options.validate()?;
  let stream_format = options.stream_format;
  let audio_stream = inference(options).await?;

  match stream_format {
    None | Some(StreamFormat::Audio) => Ok(audio_response(audio_stream)),
    Some(StreamFormat::Sse) => Ok(sse_response(audio_stream)),
  }
}

#[cfg(test)]
mod tests {
  use super::{audio_response, sse_response};
  use axum::{body::to_bytes, http::StatusCode};
  use futures::stream;
  use ortts_shared::{AudioSpec, SpeechAudioDoneEvent, SpeechAudioStream};
  use serde_json::{Value, json};

  #[tokio::test]
  async fn audio_response_streams_wav_bytes_without_content_length() {
    let speech_stream = SpeechAudioStream::new(
      AudioSpec::new(1, 24_000),
      stream::iter([Ok(vec![0.0_f32, 0.25]), Ok(vec![-0.5_f32])]),
    );
    let response = audio_response(speech_stream);

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
      response.headers().get("content-type").unwrap(),
      "application/octet-stream"
    );
    assert!(response.headers().get("content-length").is_none());

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let bytes: Vec<u8> = body.into_iter().collect();
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(
      &bytes[44..],
      &[
        0, 0, 0, 0, // 0.0
        0, 0, 128, 62, // 0.25
        0, 0, 0, 191, // -0.5
      ]
    );
  }

  #[tokio::test]
  async fn sse_response_emits_audio_chunks_followed_by_done() {
    let speech_stream = SpeechAudioStream::new(
      AudioSpec::new(1, 24_000),
      stream::iter([Ok(vec![0.0_f32, 0.25]), Ok(vec![-0.5_f32])]),
    );
    let response = sse_response(speech_stream);

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
      response.headers().get("content-type").unwrap(),
      "text/event-stream"
    );
    assert!(response.headers().get("content-length").is_none());

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let events: Vec<Value> = String::from_utf8(body.into_iter().collect())
      .unwrap()
      .lines()
      .filter_map(|line| line.strip_prefix("data: "))
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();

    assert_eq!(events.len(), 3);
    assert_eq!(events[0]["type"], "speech.audio.delta");
    assert!(events[0]["audio"].as_str().unwrap().starts_with("UklGR"));
    assert_eq!(events[1]["type"], "speech.audio.delta");
    assert_eq!(
      events[2],
      serde_json::to_value(SpeechAudioDoneEvent::new()).unwrap()
    );
    assert_eq!(events[2]["type"], json!("speech.audio.done"));
  }
}
