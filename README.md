# ORTTS

`ortts` is an OpenAI-compatible server offering text-to-speech services.

## Backends

- [x] [Chatterbox Multilingual](./crates/backend_chatterbox_multilingual/)
- [x] [Kokoro](./crates/backend_kokoro/)

## Getting Started

You can always access the integrated Scalar OpenAPI documentation at [`http://127.0.0.1:12775`](http://127.0.0.1:12775).

### Running

#### From Source

```shell
git clone https://github.com/moeru-ai/ortts.git
cd ortts
cargo run serve --release
```

#### Using Docker

```shell
docker run -d --restart always \
  -p 12775:12775 \
  ghcr.io/moeru-ai/ortts:latest
```

Or if you have existing cache directory with models downloaded or would love to reuse cache across restarts:

```shell
docker run -d --restart always \
  -p 12775:12775 \
  -v  ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  ghcr.io/moeru-ai/ortts:latest
```

### Example Request

```bash
curl -X POST \
   -H 'Content-Type: application/json' \
   -d '{ "voice": "/path/to/reference/voice/audio/file", "input": "This is a test", "model": "chatterbox-multilingual" }' \
   --output test.wav \
  "http://127.0.0.1:12775/v1/audio/speech"
```

## License

[MIT](LICENSE.md)
