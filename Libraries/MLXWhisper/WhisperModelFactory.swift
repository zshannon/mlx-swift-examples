import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

public actor WhisperModelContainer {
    let model: Whisper
    let tokenizer: WhisperTokenizer
    public let configuration: ModelConfiguration

    init(model: Whisper, tokenizer: WhisperTokenizer, configuration: ModelConfiguration) {
        self.model = model
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    public func transcribe(file path: String) throws -> String {
        let audio = try loadAudio(path)
        print("[DEBUG] Audio array shape: \(audio.shape)")
        let mel = logMelSpectrogram(audio)
        print("[DEBUG] Mel spectrogram shape: \(mel.shape)")

        // Encode audio once
        let audioFeatures = model.embedAudio(mel[.newAxis, 0...])
        print("[DEBUG] Audio features shape: \(audioFeatures.shape)")

        // Use the WhisperTokenizer special tokens
        let specialTokens = tokenizer.specialTokens

        // Create proper SOT sequence using the tokenizer
        let initialTokens = tokenizer.createPromptTokens(language: "en", task: "transcribe")
        var tokens = MLXArray(initialTokens)
        var resultTokens: [Int] = []

        // Generate tokens autoregressively
        for step in 0 ..< 100 {
            let logits = model.logits(tokens: tokens[.newAxis, 0...], audioFeatures: audioFeatures)
            let next = argMax(logits[0..., -1], axis: -1).item(Int.self)
            print("[DEBUG] Step \(step) predicted token: \(next)")

            // Stop on end of text
            if next == specialTokens.endToken {
                break
            }

            // Skip special tokens in output
            if next < specialTokens.specialTokenBegin {
                resultTokens.append(next)
            }

            tokens = concatenated([tokens, MLXArray([next])], axis: 0)

            // Keep context manageable
            if tokens.shape[0] > 50 {
                let recent = tokens[(tokens.shape[0] - 20)...]
                tokens = concatenated([MLXArray(initialTokens), recent], axis: 0)
            }
        }

        // Decode all result tokens at once
        let result = tokenizer.decode(tokens: resultTokens)
        print("[DEBUG] Final tokens: \(resultTokens)")
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public class WhisperModelFactory: @unchecked Sendable {
    public static let shared = WhisperModelFactory()

    public let modelRegistry: AbstractModelRegistry

    public init(modelRegistry: AbstractModelRegistry = WhisperRegistry.shared) {
        self.modelRegistry = modelRegistry
    }

    public func loadContainer(
        hub: HubApi = HubApi(), configuration: ModelConfiguration,
        dtype: DType = .float16,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> WhisperModelContainer {
        let model = try await loadModel(
            hub: hub, configuration: configuration, dtype: dtype,
            progressHandler: progressHandler)

        // Use our custom WhisperTokenizer instead of loading from files
        let tokenizer = WhisperTokenizer()

        let container = WhisperModelContainer(
            model: model, tokenizer: tokenizer, configuration: configuration)
        return container
    }
}
