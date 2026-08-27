# Qwen3-TTS Base backend

This backend runs the Apache-2.0 Qwen3-TTS 12 Hz 0.6B Base model through the multi-graph
ONNX export published at `onnx-community/Qwen3-TTS-12Hz-0.6B-Base`.

The public ORTTS speech interface stays unchanged:

- `input` is the text to synthesize.
- `voice` is a local reference-audio path.
- `model` is `qwen3-tts-base`, optionally suffixed with `-zh`, `-en`, or `-ja`.

The first version uses Qwen3-TTS's x-vector-only cloning mode. It therefore needs reference
audio but not its transcript. ICL cloning can be added later without changing the backend's
external seam.
