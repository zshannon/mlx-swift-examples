// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Representation of ``SpeechToTextModel`` input.
///
/// This can contain text (tokens), prepared images (`MLXArray`), or other media as
/// needed.  ``STTMInput`` is produced by ``UserInputProcessor`` in response
/// to ``UserInput``.
///
/// The ``ModelContext`` holds the ``UserInputProcessor`` associated with a
/// ``SpeechToTextModel``.
public struct STTMInput {
    
}

/// ``SpeechToTextModel`` step output.  This is consumed internally
/// by the ``TokenIterator``.
public struct STTMOutput {

    /// logits (one hot vector of probabilities for tokens)
    public let logits: MLXArray

    /// optional ``State`` to carry forward into the next step
    public let state: State?

    public struct State {
        public let crossAttentionStates: MLXArray?

        public init(crossAttentionStates: MLXArray? = nil) {
            self.crossAttentionStates = crossAttentionStates
        }
    }

    public init(logits: MLXArray, state: STTMOutput.State? = nil) {
        self.logits = logits
        self.state = state
    }
}

/// The result of the call to ``SpeechToTextModel/prepare(_:cache:windowSize:)``
public enum PrepareResult {}

/// Interface for all Speech-to-Text Models (STT).
///
/// The language model is typically called by the ``TokenIterator`` and it:
///
/// - consumes the ``STTMInput``
/// - calls ``prepare(_:cache:windowSize:)`` to initialize the KVCache and consume the prompt
/// - calls ``callAsFunction(_:cache:state:)-9kuvf`` for each token, producing an ``STTMOutput``
/// - the ``TokenIterator`` accumulates this information into a ``GenerateResult``
public protocol SpeechToTextModel: Module {

    /// Prepare the cache state and consume the ``STTMInput``.
    ///
    /// This can return:
    /// - ``PrepareResult/tokens(_:)`` if the caller should evaluate the (remaining) tokens normally
    /// - ``PrepareResult/logits(_:)`` to produce the next token from the prompt
    func prepare(_ input: STTMInput) throws -> PrepareResult

    /// Primary entry point to produce a step (single token) from the model
    func callAsFunction(_ input: STTMInput, state: STTMOutput.State?)
        -> STTMOutput

    /// Models may implement this simplified interface if they do not produce any ``STTMOutput/State``
    func callAsFunction(_ inputs: MLXArray) -> MLXArray

    /// create a new array of ``KVCache`` -- automatic implementation if self
    /// implements ``KVCacheDimensionProvider``
//    func newCache(parameters: GenerateParameters?) -> [KVCache]

    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

extension SpeechToTextModel {
    public func callAsFunction(_ input: STTMInput, state: STTMOutput.State?)
        -> STTMOutput
    {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }

    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }
}
