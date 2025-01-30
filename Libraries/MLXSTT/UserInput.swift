import Foundation

/// Container for raw user input.
///
/// A ``UserInputProcessor`` can convert this to ``STTMMInput``.
/// See also ``ModelContext``.
public struct UserInput: Sendable {
    public var audioURLs: [URL]

    public init(audio: [URL]) {
        self.audioURLs = audio
    }
}

/// Protocol for a type that can convert ``UserInput`` to ``STTMInput``.
///
/// See also ``ModelContext``.
public protocol UserInputProcessor {
    func prepare(input: UserInput) async throws -> STTMInput
}

private enum UserInputError: Error {
    case notImplemented
    case unableToLoad(URL)
    case arrayError(String)
}

/// A do-nothing ``UserInputProcessor``.
public struct StandInUserInputProcessor: UserInputProcessor {
    public init() {}

    public func prepare(input: UserInput) throws -> STTMInput {
        throw UserInputError.notImplemented
    }
}
