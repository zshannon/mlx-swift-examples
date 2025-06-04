# MLXWhisper

Swift implementation of OpenAI's Whisper speech recognition model using MLX for computation. This library provides both a Swift API for integration into apps and a command-line tool for transcribing audio files.

## Features

- **Pure Swift Implementation**: Built on top of MLX for high-performance computation on Apple Silicon
- **Automatic Model Loading**: Downloads models from Hugging Face Hub automatically
- **Multiple Model Sizes**: Support for tiny, base, small, medium, and large Whisper models
- **Audio Format Support**: Handles various audio formats through AVFoundation
- **Built-in Tokenizer**: Custom WhisperTokenizer with hardcoded special tokens for reliable operation
- **Command-Line Tool**: Ready-to-use CLI for batch processing

## Swift API Usage

### Basic Transcription

```swift
import MLXWhisper

// Load a model and transcribe audio
let container = try await WhisperModelFactory.shared.loadContainer(
    configuration: WhisperRegistry.tiny
)
let text = try container.transcribe(file: "speech.wav")
print(text)
```

### Using Different Models

```swift
// Available model configurations
let tinyModel = WhisperRegistry.tiny      // Fastest, least accurate
let baseModel = WhisperRegistry.base      // Good balance
let smallModel = WhisperRegistry.small    // Better accuracy
let mediumModel = ModelConfiguration(id: "mlx-community/whisper-medium")
let largeModel = ModelConfiguration(id: "mlx-community/whisper-large-v3")

// Load with custom configuration
let container = try await WhisperModelFactory.shared.loadContainer(
    configuration: largeModel,
    dtype: .float16
)
```

### Progress Tracking

```swift
let container = try await WhisperModelFactory.shared.loadContainer(
    configuration: WhisperRegistry.base,
    progressHandler: { progress in
        print("Loading: \(progress.fractionCompleted * 100)%")
    }
)
```

## Command-Line Tool

The `whisper-tool` provides a convenient way to transcribe audio files from the command line.

### Building and Installation

#### Using Swift Package Manager

```bash
# From the project root directory
swift build -c release --product whisper-tool

# Run directly
swift run whisper-tool --help
```

#### Using Xcode Build

```bash
# For optimized release builds
xcodebuild -scheme whisper-tool -configuration Release -derivedDataPath .build

# The binary will be located at:
# .build/Build/Products/Release/whisper-tool
# eg: `.build/Build/Products/Release/whisper-tool -m tiny Tests/MLXWhisperTests/Resources/jfk.wav`
```

### CLI Usage

```bash
# Basic usage
whisper-tool audio.wav

# Specify model size
whisper-tool --model base audio.wav

# Enable verbose output
whisper-tool --model small --verbose audio.wav

# Output as JSON
whisper-tool --model tiny --format json audio.wav

# Specify language (optional, auto-detected by default)
whisper-tool --model base --language en audio.wav
```

### CLI Options

- `--model` / `-m`: Model size (`tiny`, `base`, `small`, `medium`, `large`)
- `--verbose`: Enable detailed progress output
- `--format`: Output format (`text`, `json`)
- `--language`: Language code (e.g., `en`, `es`, `fr`) - auto-detected if not specified
- `--help` / `-h`: Show help information
- `--version`: Show version information

### Example CLI Output

```bash
$ whisper-tool --model base --verbose speech.wav
🎙️ MLX Whisper Transcription Tool
Model: base
Audio file: speech.wav
Format: text

📥 Loading model: base
✅ Model loaded successfully
🎵 Transcribing audio...
📝 Transcription completed in 2.34 seconds
📄 Result:
---
The quick brown fox jumps over the lazy dog.
```

## Supported Audio Formats

MLXWhisper supports various audio formats through AVFoundation:
- WAV
- MP3
- M4A
- FLAC
- AIFF
- And other formats supported by AVFoundation

Audio is automatically converted to the required 16kHz mono format for Whisper processing.

## Model Information

| Model | Parameters | Size | Speed | Quality |
|-------|------------|------|--------|---------|
| tiny  | 39M        | ~39MB  | Fastest | Good |
| base  | 74M        | ~74MB  | Fast    | Better |
| small | 244M       | ~244MB | Medium  | Good |
| medium| 769M       | ~769MB | Slow    | Better |
| large | 1550M      | ~1.5GB | Slowest | Best |

## Advanced Features

### Custom Tokenizer

MLXWhisper includes a built-in `WhisperTokenizer` that handles special tokens:

```swift
let tokenizer = WhisperTokenizer()

// Create prompt tokens for specific language and task
let promptTokens = tokenizer.createPromptTokens(language: "en", task: "transcribe")

// Encode/decode text
let tokens = tokenizer.encode(text: "Hello world")
let decoded = tokenizer.decode(tokens: tokens)
```

### Special Tokens

The tokenizer includes hardcoded special tokens for reliable operation:
- `<|endoftext|>` (50256)
- `<|startoftranscript|>` (50257)
- `<|en|>`, `<|es|>`, etc. (language tokens)
- `<|transcribe|>` (50359)
- `<|translate|>` (50358)
- `<|nospeech|>` (50362)
- `<|notimestamps|>` (50363)

## Performance Tips

1. **Model Selection**: Use the smallest model that meets your accuracy requirements
2. **Batch Processing**: Process multiple files sequentially to reuse loaded models
3. **Audio Preprocessing**: Convert audio to 16kHz mono beforehand for faster processing
4. **Memory Management**: Unload models when not needed for extended periods

## Error Handling

```swift
do {
    let container = try await WhisperModelFactory.shared.loadContainer(
        configuration: WhisperRegistry.base
    )
    let result = try container.transcribe(file: "audio.wav")
    print("Transcription: \(result)")
} catch {
    print("Transcription failed: \(error)")
}
```

## Running Tests

```bash
# Run all MLXWhisper tests
swift test --filter MLXWhisperTests

# Run with verbose output
swift test --filter MLXWhisperTests --verbose
```

The tests include validation using sample audio files that are automatically downloaded when needed.

## Requirements

- macOS 14.0+ or iOS 16.0+
- Xcode 15.0+
- Swift 5.9+
- Apple Silicon recommended for optimal performance

## Dependencies

- [MLX Swift](https://github.com/ml-explore/mlx-swift): Core computation framework
- [swift-transformers](https://github.com/huggingface/swift-transformers): Tokenization support
- AVFoundation: Audio loading and processing
