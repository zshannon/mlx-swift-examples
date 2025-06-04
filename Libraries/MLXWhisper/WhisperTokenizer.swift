import Foundation
import Tokenizers

public struct WhisperSpecialTokens {
    public let endToken: Int = 50256
    public let startOfTranscriptToken: Int = 50257
    public let englishToken: Int = 50258
    public let transcribeToken: Int = 50359
    public let translateToken: Int = 50358
    public let noSpeechToken: Int = 50362
    public let noTimestampsToken: Int = 50363
    public let timeTokenBegin: Int = 50364
    public let specialTokenBegin: Int = 50257
    public let whitespaceToken: Int = 220

    public init() {}
}

public protocol WhisperTokenizerProtocol {
    var specialTokens: WhisperSpecialTokens { get }
    func encode(text: String) -> [Int]
    func decode(tokens: [Int]) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?
}

public class WhisperTokenizer: WhisperTokenizerProtocol {
    private let baseTokenizer: (any Tokenizer)?
    public let specialTokens = WhisperSpecialTokens()

    // Hardcoded special token mappings following WhisperKit approach
    private let specialTokenMap: [String: Int] = [
        "<|endoftext|>": 50256,
        "<|startoftranscript|>": 50257,
        "<|en|>": 50258,
        "<|zh|>": 50259,
        "<|de|>": 50260,
        "<|es|>": 50261,
        "<|ru|>": 50262,
        "<|ko|>": 50263,
        "<|fr|>": 50264,
        "<|ja|>": 50265,
        "<|pt|>": 50266,
        "<|tr|>": 50267,
        "<|pl|>": 50268,
        "<|ca|>": 50269,
        "<|nl|>": 50270,
        "<|ar|>": 50271,
        "<|sv|>": 50272,
        "<|it|>": 50273,
        "<|id|>": 50274,
        "<|hi|>": 50275,
        "<|fi|>": 50276,
        "<|vi|>": 50277,
        "<|he|>": 50278,
        "<|uk|>": 50279,
        "<|el|>": 50280,
        "<|ms|>": 50281,
        "<|cs|>": 50282,
        "<|ro|>": 50283,
        "<|da|>": 50284,
        "<|hu|>": 50285,
        "<|ta|>": 50286,
        "<|no|>": 50287,
        "<|th|>": 50288,
        "<|ur|>": 50289,
        "<|hr|>": 50290,
        "<|bg|>": 50291,
        "<|lt|>": 50292,
        "<|la|>": 50293,
        "<|mi|>": 50294,
        "<|ml|>": 50295,
        "<|cy|>": 50296,
        "<|sk|>": 50297,
        "<|te|>": 50298,
        "<|fa|>": 50299,
        "<|lv|>": 50300,
        "<|bn|>": 50301,
        "<|sr|>": 50302,
        "<|az|>": 50303,
        "<|sl|>": 50304,
        "<|kn|>": 50305,
        "<|et|>": 50306,
        "<|mk|>": 50307,
        "<|br|>": 50308,
        "<|eu|>": 50309,
        "<|is|>": 50310,
        "<|hy|>": 50311,
        "<|ne|>": 50312,
        "<|mn|>": 50313,
        "<|bs|>": 50314,
        "<|kk|>": 50315,
        "<|sq|>": 50316,
        "<|sw|>": 50317,
        "<|gl|>": 50318,
        "<|mr|>": 50319,
        "<|pa|>": 50320,
        "<|si|>": 50321,
        "<|km|>": 50322,
        "<|sn|>": 50323,
        "<|yo|>": 50324,
        "<|so|>": 50325,
        "<|af|>": 50326,
        "<|oc|>": 50327,
        "<|ka|>": 50328,
        "<|be|>": 50329,
        "<|tg|>": 50330,
        "<|sd|>": 50331,
        "<|gu|>": 50332,
        "<|am|>": 50333,
        "<|yi|>": 50334,
        "<|lo|>": 50335,
        "<|uz|>": 50336,
        "<|fo|>": 50337,
        "<|ht|>": 50338,
        "<|ps|>": 50339,
        "<|tk|>": 50340,
        "<|nn|>": 50341,
        "<|mt|>": 50342,
        "<|sa|>": 50343,
        "<|lb|>": 50344,
        "<|my|>": 50345,
        "<|bo|>": 50346,
        "<|tl|>": 50347,
        "<|mg|>": 50348,
        "<|as|>": 50349,
        "<|tt|>": 50350,
        "<|haw|>": 50351,
        "<|ln|>": 50352,
        "<|ha|>": 50353,
        "<|ba|>": 50354,
        "<|jw|>": 50355,
        "<|su|>": 50356,
        "<|yue|>": 50357,
        "<|translate|>": 50358,
        "<|transcribe|>": 50359,
        "<|startoflm|>": 50360,
        "<|startofprev|>": 50361,
        "<|nospeech|>": 50362,
        "<|notimestamps|>": 50363,
    ]

    // Reverse map for decoding
    private lazy var idToTokenMap: [Int: String] = {
        return Dictionary(uniqueKeysWithValues: specialTokenMap.map { ($1, $0) })
    }()

    public init(baseTokenizer: (any Tokenizer)? = nil) {
        self.baseTokenizer = baseTokenizer
    }

    public func encode(text: String) -> [Int] {
        // For special tokens, return the hardcoded mapping
        if let specialId = specialTokenMap[text] {
            return [specialId]
        }

        // For regular text, use the base tokenizer if available
        if let tokenizer = baseTokenizer {
            return tokenizer.encode(text: text)
        }

        // Fallback: simple character-based encoding (this is a very basic fallback)
        return text.utf8.map { Int($0) }
    }

    public func decode(tokens: [Int]) -> String {
        var result = ""
        var regularTokens: [Int] = []

        for token in tokens {
            if let specialToken = idToTokenMap[token] {
                // Decode any accumulated regular tokens first
                if !regularTokens.isEmpty {
                    if let tokenizer = baseTokenizer {
                        result += tokenizer.decode(tokens: regularTokens)
                    } else {
                        // Fallback decoding
                        result +=
                            String(
                                bytes: regularTokens.compactMap { UInt8(exactly: $0) },
                                encoding: .utf8) ?? ""
                    }
                    regularTokens.removeAll()
                }

                // Add special token (without the angle brackets for most tokens)
                if specialToken == "<|startoftranscript|>" || specialToken == "<|endoftext|>" {
                    // These tokens are typically not included in the final output
                    continue
                } else if specialToken.hasPrefix("<|") && specialToken.hasSuffix("|>") {
                    // Language tokens and other special tokens
                    continue
                } else {
                    result += specialToken
                }
            } else {
                regularTokens.append(token)
            }
        }

        // Decode any remaining regular tokens
        if !regularTokens.isEmpty {
            if let tokenizer = baseTokenizer {
                result += tokenizer.decode(tokens: regularTokens)
            } else {
                // Fallback decoding
                result +=
                    String(bytes: regularTokens.compactMap { UInt8(exactly: $0) }, encoding: .utf8)
                    ?? ""
            }
        }

        return result
    }

    public func convertTokenToId(_ token: String) -> Int? {
        return specialTokenMap[token] ?? baseTokenizer?.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        return idToTokenMap[id] ?? baseTokenizer?.convertIdToToken(id)
    }

    // Helper method to create prompt tokens for Whisper
    public func createPromptTokens(language: String? = nil, task: String = "transcribe") -> [Int] {
        var tokens: [Int] = []

        // Start of transcript
        tokens.append(specialTokens.startOfTranscriptToken)

        // Language token
        if let language = language {
            let langToken = "<|\(language)|>"
            if let langId = specialTokenMap[langToken] {
                tokens.append(langId)
            } else {
                // Default to English
                tokens.append(specialTokens.englishToken)
            }
        } else {
            // Default to English
            tokens.append(specialTokens.englishToken)
        }

        // Task token
        if task == "translate" {
            tokens.append(specialTokens.translateToken)
        } else {
            tokens.append(specialTokens.transcribeToken)
        }

        // No timestamps for now
        tokens.append(specialTokens.noTimestampsToken)

        return tokens
    }
}
