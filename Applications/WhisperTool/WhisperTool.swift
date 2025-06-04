// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import MLX
import MLXWhisper
import MLXLMCommon
import MLXNN

@main
struct WhisperTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for transcribing audio files using Whisper",
        version: "1.0.0"
    )
    
    @Option(name: .shortAndLong, help: "Model size (tiny, base, small, medium, large)")
    var model: String = "tiny"
    
    @Argument(help: "Path to the audio file (.wav format)")
    var audioFile: String
    
    @Flag(name: .long, help: "Enable verbose output")
    var verbose: Bool = false
    
    @Option(name: .long, help: "Output format (text, json)")
    var format: String = "text"
    
    @Option(name: .long, help: "Language code (auto-detect if not specified)")
    var language: String?
    
    mutating func run() async throws {
        if verbose {
            print("🎙️ MLX Whisper Transcription Tool")
            print("Model: \(model)")
            print("Audio file: \(audioFile)")
            print("Format: \(format)")
            if let lang = language {
                print("Language: \(lang)")
            }
            print()
        }
        
        // Validate audio file exists
        guard FileManager.default.fileExists(atPath: audioFile) else {
            throw ValidationError("Audio file not found: \(audioFile)")
        }
        
        // Get model configuration
        let modelConfig = try getModelConfiguration(for: model)
        
        if verbose {
            print("📥 Loading model: \(modelConfig.name)")
        }
        
        // Create factory and load model
        let factory = WhisperModelFactory.shared
        let container = try await factory.loadContainer(configuration: modelConfig)
        
        if verbose {
            print("✅ Model loaded successfully")
            print("🎵 Transcribing audio...")
        }
        
        // Transcribe audio
        let startTime = Date()
        let transcription = try await container.transcribe(file: audioFile)
        let duration = Date().timeIntervalSince(startTime)
        
        // Output results
        switch format.lowercased() {
        case "json":
            let result = TranscriptionResult(
                text: transcription,
                model: model,
                audioFile: audioFile,
                language: language,
                duration: duration
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let jsonData = try encoder.encode(result)
            print(String(data: jsonData, encoding: .utf8)!)
            
        default: // text
            if verbose {
                print("📝 Transcription completed in \(String(format: "%.2f", duration)) seconds")
                print("📄 Result:")
                print("---")
            }
            print(transcription)
        }
    }
    
    private func getModelConfiguration(for modelName: String) throws -> ModelConfiguration {
        switch modelName.lowercased() {
        case "tiny":
            return WhisperRegistry.tiny
        case "base":
            return WhisperRegistry.base
        case "small":
            return WhisperRegistry.small
        case "medium":
            return ModelConfiguration(id: "mlx-community/whisper-medium")
        case "large":
            return ModelConfiguration(id: "mlx-community/whisper-large-v3")
        default:
            throw ValidationError("Unknown model size: \(modelName). Available: tiny, base, small, medium, large")
        }
    }
}

struct TranscriptionResult: Codable {
    let text: String
    let model: String
    let audioFile: String
    let language: String?
    let duration: TimeInterval
}