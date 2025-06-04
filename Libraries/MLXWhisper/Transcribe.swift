import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

public func transcribe(
    _ path: String,
    configuration: ModelConfiguration = WhisperRegistry.tiny,
    dtype: DType = .float16
) async throws -> String {
    let mel = logMelSpectrogram(try loadAudio(path))
    let hub = HubApi()
    let model = try await loadModel(hub: hub, configuration: configuration, dtype: dtype)
    let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)

    var tokens = MLXArray([tokenizer.bosTokenId ?? 0])
    var result = ""
    for _ in 0 ..< 448 {
        let logits = model(mel: mel[.newAxis, ...], tokens: tokens[.newAxis, ...])
        let next = Int(argmax(logits[0, -1]))
        if next == tokenizer.eosTokenId { break }
        tokens = concatenated([tokens, MLXArray([next])], axis: 0)
        result += tokenizer.decode([next])
    }
    return result
}
