// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import Tokenizers

/// Creates a function that loads a configuration file and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _: C.Type, _ modelInit: @escaping (C) -> M
) -> (URL) throws -> M {
    { url in
        let configuration = try JSONDecoder().decode(
            C.self, from: Data(contentsOf: url)
        )
        return modelInit(configuration)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``STTModelFactory/load(hub:configuration:progressHandler:)``.
public class ModelTypeRegistry: @unchecked Sendable {
    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention.  this allows the methods
    // to remain synchronous.
    private let lock = NSLock()

    private var creators: [String: @Sendable (URL) throws -> any SpeechToTextModel] = [
        "whisper": create(WhisperConfiguration.self, WhisperModel.init),
    ]

    /// Add a new model to the type registry.
    public func registerModelType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> any SpeechToTextModel
    ) {
        lock.withLock {
            creators[type] = creator
        }
    }

    /// Given a `modelType` and configuration file instantiate a new `SpeechToTextModel`.
    public func createModel(configuration: URL, modelType: String) throws -> SpeechToTextModel {
        let creator = lock.withLock {
            creators[modelType]
        }
        guard let creator else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration)
    }
}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration.  The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class ModelRegistry: @unchecked Sendable {
    private let lock = NSLock()
    private var registry = Dictionary(uniqueKeysWithValues: all().map { ($0.name, $0) })

    public static let whisperTiny = ModelConfiguration(
        id: "mlx-community/whisper-tiny",
        tokenizerId: "openai/whisper-tiny"
    )

    private static func all() -> [ModelConfiguration] {
        [
            whisperTiny,
        ]
    }

    public func register(configurations: [ModelConfiguration]) {
        lock.withLock {
            for c in configurations {
                registry[c.name] = c
            }
        }
    }

    public func configuration(id: String) -> ModelConfiguration {
        lock.withLock {
            if let c = registry[id] {
                return c
            } else {
                return ModelConfiguration(id: id)
            }
        }
    }
}

private struct STTUserInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration: ModelConfiguration

    init(tokenizer: any Tokenizer, configuration: ModelConfiguration) {
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    func prepare(input: UserInput) throws -> STTMInput {
        let messages = input.audioURLs
//            let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
        fatalError("TODO: parse audio files")
        return STTMInput()
    }
}

/// Factory for creating new STTs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await STTModelFactory.shared.loadContainer(
///     configuration: ModelRegistry.whisperTiny)
/// ```
public class STTModelFactory: ModelFactory {
    public static let shared = STTModelFactory()

    /// registry of model type, e.g. configuration value `llama` -> configuration and init methods
    public let typeRegistry = ModelTypeRegistry()

    /// registry of model id to configuration, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`
    public let modelRegistry = ModelRegistry()

    public func configuration(id: String) -> ModelConfiguration {
        modelRegistry.configuration(id: id)
    }

    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext {
        // download weights and config
        let modelDirectory = try await downloadModel(
            hub: hub, configuration: configuration, progressHandler: progressHandler
        )

        // load the generic config to unerstand which model and how to load the weights
        let configurationURL = modelDirectory.appending(component: "config.json")
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: configurationURL)
        )
        let model = try typeRegistry.createModel(
            configuration: configurationURL, modelType: baseConfig.modelType
        )

        // apply the weights to the bare model
        try loadWeights(
            modelDirectory: modelDirectory, model: model, quantization: baseConfig.quantization
        )
        
        let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)

        return .init(
            configuration: configuration, model: model,
            processor: STTUserInputProcessor(tokenizer: tokenizer, configuration: configuration),
            tokenizer: tokenizer
        )
    }
}
