# MLXWhisper

Swift implementation of OpenAI's Whisper speech recognition model. This follows
the design of the Python `mlx_whisper` example and uses `MLX` for computation.
Model weights are fetched from the Hugging Face Hub and audio loading uses
`AVFoundation`. Predefined model identifiers are available via
`WhisperRegistry`.

```
let container = try await WhisperModelFactory.shared.loadContainer(
    configuration: WhisperRegistry.tiny)
let text = try await container.transcribe(file: "speech.flac")
print(text)
```

Whisper models are available on the Hugging Face Hub. Convenience identifiers
are provided via `WhisperRegistry`:

```
let container = try await WhisperModelFactory.shared.loadContainer(
    configuration: WhisperRegistry.base)
let text = try await container.transcribe(file: "speech.flac")
```
