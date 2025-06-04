import Foundation
@preconcurrency import Tiktoken

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
    private let encoding: Encoding
    public let specialTokens = WhisperSpecialTokens()

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
        let bundle = Bundle.module
        let fileURL = bundle.url(forResource: "multilingual", withExtension: "tiktoken")!
        let data = try! Data(contentsOf: fileURL)
        let ranks = FileDecoder().decode(data)
        let regex = try! NSRegularExpression(
            pattern:
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
            options: []
        )
        self.encoding = Encoding(
            name: "whisper-multilingual", regex: regex, mergeableRanks: ranks, specialTokens: [:])
    }

    public func encode(text: String) -> [Int] {
        if let specialId = specialTokenMap[text] {
            return [specialId]
        }
        let ids = encoding.encode(value: text)
        print("[DEBUG] Encoding '\(text)' -> \(ids)")
        return ids
    }

    public func decode(tokens: [Int]) -> String {
        var result = ""
        var regular: [Int] = []
        for token in tokens {
            if let special = idToTokenMap[token] {
                if !regular.isEmpty {
                    result += encoding.decode(value: regular)
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
            let decodedRegular = encoding.decode(value: regular)
            result += decodedRegular
            
            // Debug specific problematic tokens
            if regular.contains(0) {
                print("[DEBUG] Token 0 decoded to: '\(encoding.decode(value: [0]))'")
            }
        }
        print("[DEBUG] Decoding tokens \(tokens) -> '\(result)'")
        return result
    }

    public func convertTokenToId(_ token: String) -> Int? {
        let id = specialTokenMap[token] ?? encoding.encode(value: token).first
        print("[DEBUG] convertTokenToId '\(token)' -> \(String(describing: id))")
        return id
    }

    public func convertIdToToken(_ id: Int) -> String? {
        if let special = idToTokenMap[id] {
            return special
        }
        let token = encoding.decode(value: [id])
        print("[DEBUG] convertIdToToken \(id) -> \(token)")
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

struct FileDecoder {
    func decode(_ data: Data) -> [[UInt8]: Int] {
        guard let decoded = String(data: data, encoding: .utf8) else { return [:] }
        var result: [[UInt8]: Int] = .init()
        decoded.split(separator: "\n").forEach({
            let lineSplit = $0.split(separator: " ")
            guard let first = lineSplit.first,
                let key = String(first).base64Decoded(),
                let value = lineSplit.last
            else {
                return
            }
            result[key.uInt8] = Int(value)
        })
        return result
    }
}

public class Encoding {

    //mergeable_ranks: dict[bytes, int],
    //special_tokens: dict[str, int],
    //explicit_n_vocab: Optional[int] = None,

    //    let name: String
    //    let explicitNVocab: Int?
    //    let pattern: String
    //    let mergeableRanks: [[UInt8]: Int]
    //    let specialTokens: [String: Int] // TODO: Map to [UInt8]

    private let name: String
    private let regex: NSRegularExpression  // Regex
    let mergeableRanks: [[UInt8]: Int]
    let specialTokens: [String: Int]
    private let maxValueToken: Int

    private let coreBpe: CoreBPE

    init(
        name: String, regex: NSRegularExpression, mergeableRanks: [[UInt8]: Int],
        specialTokens: [String: Int], explicitNVocab: Int? = nil
    ) {
        self.name = name
        self.regex = regex
        self.mergeableRanks = mergeableRanks
        self.specialTokens = specialTokens
        self.maxValueToken = max(mergeableRanks.values.max() ?? 0, specialTokens.values.max() ?? 0)

        // Assert validation

        //        if explicit_n_vocab:
        //            assert len(mergeable_ranks) + len(special_tokens) == explicit_n_vocab
        //            assert self.max_token_value == explicit_n_vocab - 1

        let decoder = mergeableRanks.inverted
        self.coreBpe = .init(encoder: mergeableRanks, decoder: decoder, regexTls: [regex])
    }

    public func encode(value: String) -> [Int] {
        coreBpe.encodeOrdinaryNative(text: value)
    }

    public func decode(value: [Int]) -> String {
        coreBpe.decodeNative(tokens: value)
    }
}

class CoreBPE {
    private let encoder: [[UInt8]: Int]
    private let specialTokensEncoder: [String: Int]
    private let decoder: [Int: [UInt8]]
    private let specialTokensDecoder: [Int: Data]
    private let regexTls: [NSRegularExpression]
    private let specialRegexTls: [NSRegularExpression]
    private let sortedTokenBytes: [Data]

    init(
        encoder: [[UInt8]: Int] = .init(),
        specialTokensEncoder: [String: Int] = .init(),
        decoder: [Int: [UInt8]] = .init(),
        specialTokensDecoder: [Int: Data] = .init(),
        regexTls: [NSRegularExpression] = .init(),
        specialRegexTls: [NSRegularExpression] = .init(),
        sortedTokenBytes: [Data] = .init()
    ) {
        self.encoder = encoder
        self.specialTokensEncoder = specialTokensEncoder
        self.decoder = decoder
        self.specialTokensDecoder = specialTokensDecoder
        self.regexTls = regexTls
        self.specialRegexTls = specialRegexTls
        self.sortedTokenBytes = sortedTokenBytes
    }

    func encodeOrdinaryNative(text: String) -> [Int] {
        let regex = regexTls.first!
        var ret = [Int]()
        for mat in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            if let range = Range(mat.range, in: text) {
                let piece = Array(text[range].utf8)
                if let token = encoder[piece] {
                    ret.append(token)
                    continue
                }
                let encoded = bytePairEncode([UInt8](piece), encoder)
                ret.append(contentsOf: encoded)
            }
        }
        return ret
    }

    func decodeNative(tokens: [Int]) -> String {
        let data = tokens.reduce(
            into: Data(),
            {
                if let tokenBytes = decoder[$1] {
                    $0.append(contentsOf: tokenBytes)
                }
            })
        return String(data: data, encoding: .utf8) ?? ""
    }
}

extension CoreBPE {
    fileprivate func increaseLastPieceTokenLen(tokens: [Int], lastPieceTokenLen: Int) -> (
        [Int], Int
    ) {
        func tokenIsAllSpace(_ token: Int) -> Bool {
            guard let tokenBytes = decoder[token] else { return false }
            return tokenBytes.reversed().allSatisfy { [32, 10, 9].contains($0) }  // WARNING: .all(|&b| [b' ', b'\n', b'\t'].contains(&b))
        }

        var lastPieceTokenLen = lastPieceTokenLen
        if lastPieceTokenLen > 0 && tokenIsAllSpace(tokens[tokens.count - lastPieceTokenLen]) {
            while lastPieceTokenLen < tokens.count
                && tokenIsAllSpace(tokens[tokens.count - lastPieceTokenLen - 1])
            {
                lastPieceTokenLen += 1
            }
        }

        assert(lastPieceTokenLen <= tokens.count)
        return (tokens, lastPieceTokenLen)
    }
}

// MARK: - Merges

extension CoreBPE {
    fileprivate func bytePairMerge<T>(
        _ piece: [UInt8], _ ranks: [[UInt8]: Int], completion: (Range<Int>) -> T
    ) -> [T] {
        // This is a vector of (start, rank).
        // The rank is of the byte pair starting at position start.
        // The rank of the last item in the vector is not a valid value.
        var parts = (0 ..< piece.count + 1).map { ($0, Int.max) }

        let getRank: ([(Int, Int)], Int, Int) -> Int? = { parts, startIdx, skip in
            let calculatedIndex = startIdx + skip + 2
            if calculatedIndex < parts.count {
                let range = parts[startIdx].0 ..< parts[calculatedIndex].0
                let subPiece = Array(piece[range])
                return ranks[subPiece]
            } else {
                return nil
            }
        }

        // We look up the ranks once in the beginning and iteratively update
        // them during each merge, which reduces the number of rank lookups.
        for i in 0 ..< (parts.count - 2) {
            if let rank = getRank(parts, i, 0) {
                assert(rank != Int.max)
                parts[i].1 = rank
            }
        }

        // If you have n parts and m merges, this does O(mn) work.
        // We could do something with a heap and do O(m log n) work.
        // It is important to consider that n is often small (<100), and as such
        // the cache-locality benefits outweigh the algorithmic complexity downsides
        // of the `parts` vector data structure above.

        // Note that we hash bytes, not token pairs. As long as we train BPE the way we
        // currently do, this is equivalent. An easy way to break this would be to decouple
        // merge priority from token index or to prevent specific token merges.
        while parts.count > 1 {
            // usize::MAX is a sentinel rank value allowing us to
            // take the min more quickly
            var minRank = (Int.max, 0)
            for (i, (_, rank)) in parts.enumerated() {
                if rank < minRank.0 {
                    minRank = (rank, i)
                }
            }

            if minRank.0 != Int.max {
                let i = minRank.1

                // NOTE: We are about to remove parts[i + 1]. We do not do it
                // yet because there are cache-locality benefits to updating
                // parts[i] and parts[i-1] before removing, which could thrash
                // the cache. Thus, we update the rank calculation by skipping over
                // parts[i + 1], by invoking `get_rank!` with `skip = 1`.
                parts[i].1 = getRank(parts, i, 1) ?? Int.max
                if i > 0 {
                    parts[i - 1].1 = getRank(parts, i - 1, 1) ?? Int.max
                }
                parts.remove(at: i + 1)
            } else {
                break
            }
        }

        // TODO: Use ranks
        return parts.prevCurrent({ completion($0.0 ..< $1.0) })
    }

    fileprivate func bytePairEncode(_ piece: [UInt8], _ ranks: [[UInt8]: Int]) -> [Int] {
        if piece.count == 1 {
            return [ranks[piece]!]
        }
        return bytePairMerge(
            piece, ranks,
            completion: { p in
                let chunk = Array(piece[p])
                return ranks[chunk] ?? 0
            })
    }

    //    func bytePairSplit(_ piece: [UInt8], _ ranks: [[UInt8]: Int]) -> [[UInt8]] {
    //        if piece.count == 1 {
    //            return [piece]
    //        }
    //        return bytePairMerge(piece, ranks, completion: { Array(piece[$0]) })
    //    }
}

extension Array {
    func prevCurrent<T>(_ body: (Element, Element) throws -> T) rethrows -> [T] {
        enumerated().compactMap({ index, element in
            guard index > 0 else { return nil }
            let prev = self[index - 1]
            return try? body(prev, element)
        })
    }
}

typealias Ranks = [[UInt8]: Int]
extension Ranks {
    var inverted: [Int: [UInt8]] {
        reduce(into: [:], { $0[$1.value] = $1.key })
    }
}

extension String {
    func base64Encoded() -> String? {
        data(using: .utf8)?.base64EncodedString()
    }

    func base64Decoded() -> String? {
        guard let data = Data(base64Encoded: self) else { return nil }
        return String(data: data, encoding: .ascii)
    }
}
extension String {
    var uInt8: [UInt8] { utf16.map({ UInt8($0) }) }
}
