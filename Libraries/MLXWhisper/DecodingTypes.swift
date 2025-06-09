import Foundation
import MLX

// MARK: - Decoding Options

public struct DecodingOptions {
    public let task: String
    public let language: String?
    public let temperature: Float
    public let sampleLen: Int?
    public let bestOf: Int?
    public let beamSize: Int?
    public let patience: Float?
    public let lengthPenalty: Float?
    public let suppressBlank: Bool
    public let suppressTokens: [Int]
    public let initialPrompt: String?
    public let conditionOnPreviousText: Bool
    public let fp16: Bool
    public let compressionRatioThreshold: Float?
    public let logprobThreshold: Float?
    public let noSpeechThreshold: Float?
    public let wordTimestamps: Bool
    public let prependPunctuations: String
    public let appendPunctuations: String
    public let maxInitialTimestamp: Float?
    public let prompt: [Int]?

    public init(
        task: String = "transcribe",
        language: String? = nil,
        temperature: Float = 0.0,
        sampleLen: Int? = nil,
        bestOf: Int? = nil,
        beamSize: Int? = nil,
        patience: Float? = nil,
        lengthPenalty: Float? = nil,
        suppressBlank: Bool = true,
        suppressTokens: [Int] = [-1],
        initialPrompt: String? = nil,
        conditionOnPreviousText: Bool = true,
        fp16: Bool = true,
        compressionRatioThreshold: Float? = 2.4,
        logprobThreshold: Float? = -1.0,
        noSpeechThreshold: Float? = 0.6,
        wordTimestamps: Bool = false,
        prependPunctuations: String = "\"'\"¿([{-",
        appendPunctuations: String = "\"'.。,，!！?？:：\")]}、",
        maxInitialTimestamp: Float? = 1.0,
        prompt: [Int]? = nil
    ) {
        self.task = task
        self.language = language
        self.temperature = temperature
        self.sampleLen = sampleLen
        self.bestOf = bestOf
        self.beamSize = beamSize
        self.patience = patience
        self.lengthPenalty = lengthPenalty
        self.suppressBlank = suppressBlank
        self.suppressTokens = suppressTokens
        self.initialPrompt = initialPrompt
        self.conditionOnPreviousText = conditionOnPreviousText
        self.fp16 = fp16
        self.compressionRatioThreshold = compressionRatioThreshold
        self.logprobThreshold = logprobThreshold
        self.noSpeechThreshold = noSpeechThreshold
        self.wordTimestamps = wordTimestamps
        self.prependPunctuations = prependPunctuations
        self.appendPunctuations = appendPunctuations
        self.maxInitialTimestamp = maxInitialTimestamp
        self.prompt = prompt
    }
}

// MARK: - Decoding Result

public struct DecodingResult {
    public let audio_features: MLXArray
    public let language: String
    public let language_probs: [String: Float]?
    public let tokens: [Int]
    public let text: String
    public let avgLogprob: Float
    public let noSpeechProb: Float
    public let temperature: Float
    public let compressionRatio: Float

    public init(
        audio_features: MLXArray,
        language: String,
        language_probs: [String: Float]? = nil,
        tokens: [Int],
        text: String,
        avgLogprob: Float,
        noSpeechProb: Float,
        temperature: Float,
        compressionRatio: Float
    ) {
        self.audio_features = audio_features
        self.language = language
        self.language_probs = language_probs
        self.tokens = tokens
        self.text = text
        self.avgLogprob = avgLogprob
        self.noSpeechProb = noSpeechProb
        self.temperature = temperature
        self.compressionRatio = compressionRatio
    }
}

// MARK: - Segment

public struct TranscriptionSegment {
    public let id: Int
    public let seek: Int
    public let start: Float
    public let end: Float
    public let text: String
    public let tokens: [Int]
    public let temperature: Float
    public let avgLogprob: Float
    public let compressionRatio: Float
    public let noSpeechProb: Float
    public let words: [WordTimestamp]?

    public init(
        id: Int,
        seek: Int,
        start: Float,
        end: Float,
        text: String,
        tokens: [Int],
        temperature: Float,
        avgLogprob: Float,
        compressionRatio: Float,
        noSpeechProb: Float,
        words: [WordTimestamp]? = nil
    ) {
        self.id = id
        self.seek = seek
        self.start = start
        self.end = end
        self.text = text
        self.tokens = tokens
        self.temperature = temperature
        self.avgLogprob = avgLogprob
        self.compressionRatio = compressionRatio
        self.noSpeechProb = noSpeechProb
        self.words = words
    }
}

// MARK: - Word Timestamp

public struct WordTimestamp {
    public let word: String
    public let start: Float
    public let end: Float
    public let probability: Float

    public init(word: String, start: Float, end: Float, probability: Float) {
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability
    }
}

// MARK: - Transcription Result

public struct TranscriptionResult {
    public let text: String
    public let segments: [TranscriptionSegment]
    public let language: String

    public init(text: String, segments: [TranscriptionSegment], language: String) {
        self.text = text
        self.segments = segments
        self.language = language
    }
}

// MARK: - Language Detection

public struct LanguageDetectionResult {
    public let language: String
    public let probability: Float
    public let languageProbs: [String: Float]

    public init(language: String, probability: Float, languageProbs: [String: Float]) {
        self.language = language
        self.probability = probability
        self.languageProbs = languageProbs
    }
}
