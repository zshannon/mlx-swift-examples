import Foundation
import Tokenizers

public struct WhisperSpecialTokens {
    public let endToken: Int = 50257
    public let startOfTranscriptToken: Int = 50258
    public let englishToken: Int = 50259
    public let transcribeToken: Int = 50360
    public let translateToken: Int = 50359
    public let noSpeechToken: Int = 50363
    public let noTimestampsToken: Int = 50364
    public let timeTokenBegin: Int = 50365
    public let specialTokenBegin: Int = 50258
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
    private var tokenizer: PreTrainedTokenizer?
    public let specialTokens = WhisperSpecialTokens()
    private let tokenizerQueue = DispatchQueue(label: "whisper.tokenizer.encoding")

    // Hardcoded special token mappings following WhisperKit approach
    private let specialTokenMap: [String: Int] = [
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|zh|>": 50260,
        "<|de|>": 50261,
        "<|es|>": 50262,
        "<|ru|>": 50263,
        "<|ko|>": 50264,
        "<|fr|>": 50265,
        "<|ja|>": 50266,
        "<|pt|>": 50267,
        "<|tr|>": 50268,
        "<|pl|>": 50269,
        "<|ca|>": 50270,
        "<|nl|>": 50271,
        "<|ar|>": 50272,
        "<|sv|>": 50273,
        "<|it|>": 50274,
        "<|id|>": 50275,
        "<|hi|>": 50276,
        "<|fi|>": 50277,
        "<|vi|>": 50278,
        "<|he|>": 50279,
        "<|uk|>": 50280,
        "<|el|>": 50281,
        "<|ms|>": 50282,
        "<|cs|>": 50283,
        "<|ro|>": 50284,
        "<|da|>": 50285,
        "<|hu|>": 50286,
        "<|ta|>": 50287,
        "<|no|>": 50288,
        "<|th|>": 50289,
        "<|ur|>": 50290,
        "<|hr|>": 50291,
        "<|bg|>": 50292,
        "<|lt|>": 50293,
        "<|la|>": 50294,
        "<|mi|>": 50295,
        "<|ml|>": 50296,
        "<|cy|>": 50297,
        "<|sk|>": 50298,
        "<|te|>": 50299,
        "<|fa|>": 50300,
        "<|lv|>": 50301,
        "<|bn|>": 50302,
        "<|sr|>": 50303,
        "<|az|>": 50304,
        "<|sl|>": 50305,
        "<|kn|>": 50306,
        "<|et|>": 50307,
        "<|mk|>": 50308,
        "<|br|>": 50309,
        "<|eu|>": 50310,
        "<|is|>": 50311,
        "<|hy|>": 50312,
        "<|ne|>": 50313,
        "<|mn|>": 50314,
        "<|bs|>": 50315,
        "<|kk|>": 50316,
        "<|sq|>": 50317,
        "<|sw|>": 50318,
        "<|gl|>": 50319,
        "<|mr|>": 50320,
        "<|pa|>": 50321,
        "<|si|>": 50322,
        "<|km|>": 50323,
        "<|sn|>": 50324,
        "<|yo|>": 50325,
        "<|so|>": 50326,
        "<|af|>": 50327,
        "<|oc|>": 50328,
        "<|ka|>": 50329,
        "<|be|>": 50330,
        "<|tg|>": 50331,
        "<|sd|>": 50332,
        "<|gu|>": 50333,
        "<|am|>": 50334,
        "<|yi|>": 50335,
        "<|lo|>": 50336,
        "<|uz|>": 50337,
        "<|fo|>": 50338,
        "<|ht|>": 50339,
        "<|ps|>": 50340,
        "<|tk|>": 50341,
        "<|nn|>": 50342,
        "<|mt|>": 50343,
        "<|sa|>": 50344,
        "<|lb|>": 50345,
        "<|my|>": 50346,
        "<|bo|>": 50347,
        "<|tl|>": 50348,
        "<|mg|>": 50349,
        "<|as|>": 50350,
        "<|tt|>": 50351,
        "<|haw|>": 50352,
        "<|ln|>": 50353,
        "<|ha|>": 50354,
        "<|ba|>": 50355,
        "<|jw|>": 50356,
        "<|su|>": 50357,
        "<|yue|>": 50358,
        "<|translate|>": 50359,
        "<|transcribe|>": 50360,
        "<|startoflm|>": 50361,
        "<|startofprev|>": 50362,
        "<|nospeech|>": 50363,
        "<|notimestamps|>": 50364,
    ]

    // Reverse map for decoding
    private lazy var idToTokenMap: [Int: String] = {
        return Dictionary(uniqueKeysWithValues: specialTokenMap.map { ($1, $0) })
    }()

    public init() {
        // Initialize with a basic encoding - we'll use GPT-2 as the base
        // since Whisper uses a similar tokenization approach
        self.initializeTokenizer()
    }

    private func initializeTokenizer() {
        tokenizerQueue.async {
            Task {
                do {
                    if let tok = try await AutoTokenizer.from(pretrained: "gpt2") as? PreTrainedTokenizer {
                        self.tokenizer = tok
                    }
                } catch {
                    print("Failed to load encoding: \(error)")
                }
            }
        }
    }

    private func waitForTokenizer() -> PreTrainedTokenizer? {
        // Simple synchronous wait for encoding to be available
        let semaphore = DispatchSemaphore(value: 0)
        var result: PreTrainedTokenizer?

        if tokenizer != nil {
            return tokenizer
        }

        tokenizerQueue.async {
            Task {
                do {
                    if let tok = try await AutoTokenizer.from(pretrained: "gpt2") as? PreTrainedTokenizer {
                        result = tok
                    }
                    semaphore.signal()
                } catch {
                    print("Failed to load encoding: \(error)")
                    semaphore.signal()
                }
            }
        }

        semaphore.wait()
        tokenizer = result
        return result
    }

    public func encode(text: String) -> [Int] {
        if let specialId = specialTokenMap[text] {
            return [specialId]
        }
        let tok = tokenizer ?? waitForTokenizer()
        let ids = tok?.encode(text: text) ?? []
        print("[DEBUG] Encoding '\(text)' -> \(ids)")
        return ids
    }

    public func decode(tokens: [Int]) -> String {
        let tok = tokenizer ?? waitForTokenizer()
        var result = ""
        var regular: [Int] = []
        for token in tokens {
            if let special = idToTokenMap[token] {
                if !regular.isEmpty {
                    result += tok?.decode(tokens: regular) ?? ""
                    regular.removeAll()
                }
                if special == "<|startoftranscript|>" || special == "<|endoftext|>" {
                    continue
                } else if special.hasPrefix("<|") && special.hasSuffix("|>") {
                    continue
                } else {
                    result += special
                }
            } else {
                regular.append(token)
            }
        }
        if !regular.isEmpty {
            result += tok?.decode(tokens: regular) ?? ""
        }
        print("[DEBUG] Decoding tokens \(tokens) -> \(result)")
        return result
    }

    public func convertTokenToId(_ token: String) -> Int? {
        let tok = tokenizer ?? waitForTokenizer()
        let id = specialTokenMap[token] ?? tok?.convertTokenToId(token)
        print("[DEBUG] convertTokenToId '\(token)' -> \(String(describing: id))")
        return id
    }

    public func convertIdToToken(_ id: Int) -> String? {
        if let special = idToTokenMap[id] {
            return special
        }
        let tok = tokenizer ?? waitForTokenizer()
        let token = tok?.convertIdToToken(id)
        print("[DEBUG] convertIdToToken \(id) -> \(String(describing: token))")
        return token
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
