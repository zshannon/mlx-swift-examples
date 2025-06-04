# WhisperTool

A command-line tool for transcribing audio files using OpenAI's Whisper models, powered by MLX Swift.

## Features

- Support for multiple Whisper model sizes (tiny, base, small, medium, large)
- Fast transcription using Apple Silicon GPU acceleration
- Multiple output formats (text, JSON)
- Language detection and specification
- Verbose mode for detailed progress information

## Usage

### Basic Usage

```bash
swift run whisper-tool path/to/audio.wav
```

### Specify Model Size

```bash
swift run whisper-tool --model base path/to/audio.wav
```

### JSON Output

```bash
swift run whisper-tool --format json --model small path/to/audio.wav
```

### Verbose Mode

```bash
swift run whisper-tool --verbose --model medium path/to/audio.wav
```

### Specify Language

```bash
swift run whisper-tool --language es --model base path/to/spanish_audio.wav
```

## Command Line Options

- `--model, -m`: Model size (tiny, base, small, medium, large) [default: tiny]
- `--format`: Output format (text, json) [default: text]
- `--language`: Language code for transcription (auto-detect if not specified)
- `--verbose`: Enable verbose output with progress information
- `--help`: Show help information
- `--version`: Show version information

## Model Sizes

| Model  | Parameters | Relative Speed | Relative Accuracy |
|--------|------------|----------------|-------------------|
| tiny   | 39 M       | ~32x           | Good              |
| base   | 74 M       | ~16x           | Better            |
| small  | 244 M      | ~6x            | Better            |
| medium | 769 M      | ~2x            | Best              |
| large  | 1550 M     | 1x             | Best              |

## Audio Format Requirements

- Supported format: WAV files
- Sample rate: 16 kHz (recommended)
- Channels: Mono or stereo (will be converted to mono)

## Examples

### Quick transcription with tiny model
```bash
swift run whisper-tool recording.wav
```

### High-quality transcription with progress
```bash
swift run whisper-tool --verbose --model large interview.wav
```

### JSON output for integration
```bash
swift run whisper-tool --format json --model base lecture.wav > transcription.json
```

### Spanish audio with language hint
```bash
swift run whisper-tool --language es --model medium spanish_podcast.wav
```

## JSON Output Format

When using `--format json`, the output includes:

```json
{
  "text": "The transcribed text appears here...",
  "model": "base",
  "audioFile": "path/to/audio.wav",
  "language": "en",
  "duration": 2.34
}
```

## Building

To build the tool:

```bash
swift build --product whisper-tool
```

To install locally:

```bash
swift build --product whisper-tool --configuration release
cp .build/release/whisper-tool /usr/local/bin/
```

## Performance Tips

- Use `tiny` model for quick drafts and real-time applications
- Use `base` or `small` for good balance of speed and accuracy
- Use `medium` or `large` for highest accuracy on important content
- Ensure your audio file is in WAV format for best compatibility
- Use shorter audio files (< 30 minutes) for optimal memory usage