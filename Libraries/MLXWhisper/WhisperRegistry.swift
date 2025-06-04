import Foundation
import Hub
import MLXLMCommon

/// Predefined Whisper model configurations.
public class WhisperRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared = WhisperRegistry(modelConfigurations: all())

    /// Tiny Whisper model.
    public static let tiny = ModelConfiguration(id: "mlx-community/whisper-tiny")
    /// Base Whisper model.
    public static let base = ModelConfiguration(id: "mlx-community/whisper-base")
    /// Small Whisper model.
    public static let small = ModelConfiguration(id: "mlx-community/whisper-small")

    /// All predefined configurations.
    public static func all() -> [ModelConfiguration] {
        [tiny, base, small]
    }
}
