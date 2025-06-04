import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

public actor WhisperModelContainer {
    let model: Whisper
    let tokenizer: Tokenizer
    public let configuration: ModelConfiguration

    init(model: Whisper, tokenizer: Tokenizer, configuration: ModelConfiguration) {
        self.model = model
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    public func transcribe(file path: String) throws -> String {
        let mel = logMelSpectrogram(try loadAudio(path))
        var tokens = MLXArray([tokenizer.bosTokenId ?? 0])
        var result = ""
        for _ in 0..<448 {
            let logits = model(mel: mel[.newAxis, ...], tokens: tokens[.newAxis, ...])
            let next = Int(argmax(logits[0, -1]))
            if next == tokenizer.eosTokenId { break }
            tokens = concatenated([tokens, MLXArray([next])], axis: 0)
            result += tokenizer.decode([next])
        }
        return result
    }
}

public class WhisperModelFactory: Sendable {
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
        let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)
        return WhisperModelContainer(model: model, tokenizer: tokenizer, configuration: configuration)
    }
}
