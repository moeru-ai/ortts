# ORTTS

`ortts` is an OpenAI-compatible server offering text-to-speech services.

## Backends

- [x] [Chatterbox Multilingual](./crates/model_chatterbox_multilingual/)

## Example

```bash
curl http://localhost:12775/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox-multilingual",
    "input": "I can eat glass, it does not hurt me.",
    "voice": "alloy"
  }' \
  --output speech.wav

```

## License

[MIT](LICENSE.md)
