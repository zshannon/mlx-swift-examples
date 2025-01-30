import AVFAudio
import Foundation
import Hub
import NaturalLanguage
import Tokenizers

public func loadTokenizer(
    configuration: ModelConfiguration,
    hub: HubApi
) async throws -> WhisperTokenizer {
    let tokenizerName = configuration.tokenizerId ?? configuration.name

    // Attempt to load tokenizer from local folder if specified
    let resolvedTokenizerFolder = hub.localRepoLocation(HubApi.Repo(id: tokenizerName))
    let tokenizerConfigPath = resolvedTokenizerFolder.appendingPathComponent("tokenizer.json")

    // Check if 'tokenizer.json' exists in the folder
    if FileManager.default.fileExists(atPath: tokenizerConfigPath.path) {
        do {
            let localConfig = LanguageModelConfigurationFromHub(modelFolder: resolvedTokenizerFolder, hubApi: hub)
            if let tokenizerConfig = try await localConfig.tokenizerConfig {
                let tokenizerData = try await localConfig.tokenizerData
                let whisperTokenizer = try PreTrainedTokenizer(
                    tokenizerConfig: tokenizerConfig,
                    tokenizerData: tokenizerData
                )
//                print("Loading tokenizer from local folder")
                return WhisperTokenizerWrapper(tokenizer: whisperTokenizer)
            } else {
                // tokenizerConfig is nil, fall through to load from Hub
//                print("Tokenizer configuration not found in local config")
            }
        } catch {
            // Error during the local loading process and fall through to load from Hub
//            print("Error loading local tokenizer: \(error)")
            throw error
        }
    }

    // Fallback to loading from the Hub if local loading is not possible or fails
//    print("Loading tokenizer from Hub")
    return try await WhisperTokenizerWrapper(
        tokenizer: AutoTokenizer.from(
            pretrained: tokenizerName,
            hubApi: hub
        )
    )
}

public struct SpecialTokens {
    public let endToken: Int
    public let englishToken: Int
    public let noSpeechToken: Int
    public let noTimestampsToken: Int
    public let specialTokenBegin: Int
    public let startOfPreviousToken: Int
    public let startOfTranscriptToken: Int
    public let timeTokenBegin: Int
    public let transcribeToken: Int
    public let translateToken: Int
    public let whitespaceToken: Int

    public init(
        endToken: Int,
        englishToken: Int,
        noSpeechToken: Int,
        noTimestampsToken: Int,
        specialTokenBegin: Int,
        startOfPreviousToken: Int,
        startOfTranscriptToken: Int,
        timeTokenBegin: Int,
        transcribeToken: Int,
        translateToken: Int,
        whitespaceToken: Int
    ) {
        self.endToken = endToken
        self.englishToken = englishToken
        self.noSpeechToken = noSpeechToken
        self.noTimestampsToken = noTimestampsToken
        self.specialTokenBegin = specialTokenBegin
        self.startOfPreviousToken = startOfPreviousToken
        self.startOfTranscriptToken = startOfTranscriptToken
        self.timeTokenBegin = timeTokenBegin
        self.transcribeToken = transcribeToken
        self.translateToken = translateToken
        self.whitespaceToken = whitespaceToken
    }
}

public protocol WhisperTokenizer: Tokenizer {
    var specialTokens: SpecialTokens { get }
    var allLanguageTokens: Set<Int> { get }

    func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]])
}

struct WhisperTokenizerWrapper: WhisperTokenizer {
    let tokenizer: any Tokenizer
    let specialTokens: SpecialTokens
    let allLanguageTokens: Set<Int>

    init(tokenizer: any Tokenizer) {
        let specialTokens = SpecialTokens(
            endToken: tokenizer.convertTokenToId("<|endoftext|>") ?? Self.defaultEndToken,
            englishToken: tokenizer.convertTokenToId("<|en|>") ?? Self.defaultEnglishToken,
            noSpeechToken: tokenizer.convertTokenToId("<|nospeech|>") ?? Self.defaultNoSpeechToken,
            noTimestampsToken: tokenizer.convertTokenToId("<|notimestamps|>") ?? Self.defaultNoTimestampsToken,
            specialTokenBegin: tokenizer.convertTokenToId("<|endoftext|>") ?? Self.defaultSpecialTokenBegin,
            startOfPreviousToken: tokenizer.convertTokenToId("<|startofprev|>") ?? Self.defaultStartOfPreviousToken,
            startOfTranscriptToken: tokenizer.convertTokenToId("<|startoftranscript|>") ?? Self
                .defaultStartOfTranscriptToken,
            timeTokenBegin: tokenizer.convertTokenToId("<|0.00|>") ?? Self.defaultTimeTokenBegin,
            transcribeToken: tokenizer.convertTokenToId("<|transcribe|>") ?? Self.defaultTranscribeToken,
            translateToken: tokenizer.convertTokenToId("<|translate|>") ?? Self.defaultTranslateToken,
            whitespaceToken: tokenizer.convertTokenToId(" ") ?? Self.defaultWhitespaceToken
        )
        self.tokenizer = tokenizer
        self.specialTokens = specialTokens
        allLanguageTokens = Set(
            Constants.languages
                .compactMap { tokenizer.convertTokenToId("<|\($0.value)|>") }
                .filter { $0 > specialTokens.specialTokenBegin }
        )
    }

    private func splitTokensOnUnicode(tokens: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        let decodedFull = tokenizer.decode(tokens: tokens)
        let replacementString = "\u{fffd}"

        var words: [String] = []
        var wordTokens: [[Int]] = []
        var currentTokens: [Int] = []
        var unicodeOffset = 0

        for token in tokens {
            currentTokens.append(token)
            let decoded = tokenizer.decode(tokens: currentTokens)

            var hasUnicodeInFullString = false
            if let range = decoded.range(of: replacementString) {
                hasUnicodeInFullString = decodedFull[range] == replacementString
            }

            if !decoded.contains(replacementString) || hasUnicodeInFullString {
                words.append(decoded)
                wordTokens.append(currentTokens)
                currentTokens = []
                unicodeOffset += decoded.count
            }
        }

        return (words, wordTokens)
    }

    private func splitTokensOnSpaces(tokens: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        let (subwords, subwordTokensList) = splitTokensOnUnicode(tokens: tokens)
        var words: [String] = []
        var wordTokens: [[Int]] = []

        for (subword, subwordTokens) in zip(subwords, subwordTokensList) {
            let special = subwordTokens.first! >= specialTokens.specialTokenBegin
            let withSpace = subword.hasPrefix(" ")
            var punctuation = false
            if let strippedSubword = UnicodeScalar(subword.trimmingCharacters(in: .whitespaces)) {
                punctuation = CharacterSet.punctuationCharacters.contains(strippedSubword)
            }
            if special || withSpace || punctuation || words.isEmpty {
                words.append(subword)
                wordTokens.append(subwordTokens)
            } else {
                words[words.count - 1] += subword
                wordTokens[words.count - 1].append(contentsOf: subwordTokens)
            }
        }

        return (words, wordTokens)
    }

    private func isPunctuation(_ text: String, tokenRange: Range<String.Index>, tag: NLTag?) -> Bool {
        let punctuationCharacters = CharacterSet.punctuationCharacters
        let token = String(text[tokenRange])
        if let tag = tag, tag == .punctuation {
            return true
        } else if token.unicodeScalars.allSatisfy({ punctuationCharacters.contains($0) }) {
            return true
        }
        return false
    }

    /// Decodes token ids into individual words and per-word subtokens
    /// - Parameter tokenIds: Array of tokens to decode and then split
    /// - Returns: Tuple containing and array of the split words and all tokens for each word
    func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        let decodedWords = tokenizer.decode(tokens: tokenIds.filter { $0 < specialTokens.specialTokenBegin })

        // Detect language of input text
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(decodedWords)
        let languageCode = recognizer.dominantLanguage?.rawValue

        if ["zh", "ja", "th", "lo", "my", "yue"].contains(languageCode) {
            return splitTokensOnUnicode(tokens: tokenIds)
        } else {
            return splitTokensOnSpaces(tokens: tokenIds)
        }
    }
}

extension WhisperTokenizerWrapper: Tokenizer {
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func applyChatTemplate(messages: [[String: String]]) throws -> [Int] {
        try tokenizer.applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(messages: [[String: String]],
                           chatTemplate: Tokenizers.ChatTemplateArgument) throws -> [Int]
    {
        try tokenizer.applyChatTemplate(messages: messages, chatTemplate: chatTemplate)
    }

    func applyChatTemplate(messages: [[String: String]], chatTemplate: String) throws -> [Int] {
        try tokenizer.applyChatTemplate(messages: messages, chatTemplate: chatTemplate)
    }

    func applyChatTemplate(
        messages: [[String: String]],
        chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [[String: Any]]?
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: chatTemplate,
            addGenerationPrompt: addGenerationPrompt,
            truncation: truncation,
            maxLength: maxLength,
            tools: tools
        )
    }

    func tokenize(text: String) -> [String] {
        tokenizer.tokenize(text: text)
    }

    func encode(text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        tokenizer.convertIdToToken(id)
    }

    var bosToken: String? {
        tokenizer.bosToken
    }

    var bosTokenId: Int? {
        tokenizer.bosTokenId
    }

    var eosToken: String? {
        tokenizer.eosToken
    }

    var eosTokenId: Int? {
        tokenizer.eosTokenId
    }

    var unknownToken: String? {
        tokenizer.unknownToken
    }

    var unknownTokenId: Int? {
        tokenizer.unknownTokenId
    }
}

extension WhisperTokenizerWrapper {
    /// Default values for each token, using base vocab
    static var defaultWhitespaceToken: Int { 220 }
    static var defaultSpecialTokenBegin: Int { 50257 }
    static var defaultEndToken: Int { 50257 }
    static var defaultStartOfPreviousToken: Int { 50361 }
    static var defaultStartOfTranscriptToken: Int { 50258 }
    static var defaultEnglishToken: Int { 50259 }
    static var defaultTranscribeToken: Int { 50359 }
    static var defaultTranslateToken: Int { 50358 }
    static var defaultNoSpeechToken: Int { 50362 }
    static var defaultNoTimestampsToken: Int { 50363 }
    static var defaultTimeTokenBegin: Int { 50364 }
}

@frozen
public enum Constants {
    enum Logging {
        static let subsystem = "com.argmax.whisperkit"
    }

    static let specialTokenCharacters = CharacterSet(charactersIn: "<|>")

    public static let maxTokenContext = Int(448 / 2)
    public static let languages: [String: String] =
        [
            "english": "en",
            "chinese": "zh",
            "german": "de",
            "spanish": "es",
            "russian": "ru",
            "korean": "ko",
            "french": "fr",
            "japanese": "ja",
            "portuguese": "pt",
            "turkish": "tr",
            "polish": "pl",
            "catalan": "ca",
            "dutch": "nl",
            "arabic": "ar",
            "swedish": "sv",
            "italian": "it",
            "indonesian": "id",
            "hindi": "hi",
            "finnish": "fi",
            "vietnamese": "vi",
            "hebrew": "he",
            "ukrainian": "uk",
            "greek": "el",
            "malay": "ms",
            "czech": "cs",
            "romanian": "ro",
            "danish": "da",
            "hungarian": "hu",
            "tamil": "ta",
            "norwegian": "no",
            "thai": "th",
            "urdu": "ur",
            "croatian": "hr",
            "bulgarian": "bg",
            "lithuanian": "lt",
            "latin": "la",
            "maori": "mi",
            "malayalam": "ml",
            "welsh": "cy",
            "slovak": "sk",
            "telugu": "te",
            "persian": "fa",
            "latvian": "lv",
            "bengali": "bn",
            "serbian": "sr",
            "azerbaijani": "az",
            "slovenian": "sl",
            "kannada": "kn",
            "estonian": "et",
            "macedonian": "mk",
            "breton": "br",
            "basque": "eu",
            "icelandic": "is",
            "armenian": "hy",
            "nepali": "ne",
            "mongolian": "mn",
            "bosnian": "bs",
            "kazakh": "kk",
            "albanian": "sq",
            "swahili": "sw",
            "galician": "gl",
            "marathi": "mr",
            "punjabi": "pa",
            "sinhala": "si",
            "khmer": "km",
            "shona": "sn",
            "yoruba": "yo",
            "somali": "so",
            "afrikaans": "af",
            "occitan": "oc",
            "georgian": "ka",
            "belarusian": "be",
            "tajik": "tg",
            "sindhi": "sd",
            "gujarati": "gu",
            "amharic": "am",
            "yiddish": "yi",
            "lao": "lo",
            "uzbek": "uz",
            "faroese": "fo",
            "haitian creole": "ht",
            "pashto": "ps",
            "turkmen": "tk",
            "nynorsk": "nn",
            "maltese": "mt",
            "sanskrit": "sa",
            "luxembourgish": "lb",
            "myanmar": "my",
            "tibetan": "bo",
            "tagalog": "tl",
            "malagasy": "mg",
            "assamese": "as",
            "tatar": "tt",
            "hawaiian": "haw",
            "lingala": "ln",
            "hausa": "ha",
            "bashkir": "ba",
            "javanese": "jw",
            "sundanese": "su",
            "cantonese": "yue",
            "burmese": "my",
            "valencian": "ca",
            "flemish": "nl",
            "haitian": "ht",
            "letzeburgesch": "lb",
            "pushto": "ps",
            "panjabi": "pa",
            "moldavian": "ro",
            "moldovan": "ro",
            "sinhalese": "si",
            "castilian": "es",
            "mandarin": "zh",
        ]

    public static let languageCodes: Set<String> = Set(languages.values)

    public static let defaultLanguageCode: String = "en"

    public static let defaultAudioReadFrameSize: AVAudioFrameCount =
        1_323_000 // 30s of audio at commonly found 44.1khz sample rate

    public static let defaultWindowSamples: Int =
        480_000 // 30s of audio at 16khz sample rate default for Whisper models
}

public protocol StreamingDetokenizer: IteratorProtocol<String> {

    mutating func append(token: Int)

}

public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating public func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokens: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokens: segmentTokens)
        let new = newSegment.suffix(newSegment.count - segment.count)

        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return String(new)
    }

}
