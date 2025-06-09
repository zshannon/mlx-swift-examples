import Compression
import Foundation
import MLX
import MLXFast
import MLXRandom

// MARK: - Whisper Decoder

public class WhisperDecoder {
    private let model: Whisper
    private let tokenizer: WhisperTokenizerProtocol

    public init(model: Whisper, tokenizer: WhisperTokenizerProtocol) {
        self.model = model
        self.tokenizer = tokenizer
    }

    // MARK: - Language Detection

    public func detectLanguage(_ mel: MLXArray) -> LanguageDetectionResult {
        let audioFeatures = model.embedAudio(mel)
        let tokens = MLXArray([tokenizer.specialTokens.startOfTranscriptToken])
        let logits = model.logits(
            tokens: tokens.expandedDimensions(axis: 0), audioFeatures: audioFeatures)

        // Get language token probabilities
        let languageTokenStart = tokenizer.specialTokens.startOfTranscriptToken + 1
        let languageTokenEnd = languageTokenStart + model.numLanguages

        let languageLogits = logits[0, 0, languageTokenStart ..< languageTokenEnd]
        let languageProbs = softmax(languageLogits, axis: -1)

        // Convert to language codes and probabilities
        var languageProbsDict: [String: Float] = [:]
        let languages = [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
            "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
            "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
            "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
            "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
            "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
            "su",
        ]

        for (i, lang) in languages.enumerated() {
            if i < languageProbs.size {
                languageProbsDict[lang] = languageProbs[i].item(Float.self)
            }
        }

        // Find most probable language
        let maxLang = languageProbsDict.max { $0.value < $1.value }
        let detectedLanguage = maxLang?.key ?? "en"
        let detectedProb = maxLang?.value ?? 0.0

        return LanguageDetectionResult(
            language: detectedLanguage,
            probability: detectedProb,
            languageProbs: languageProbsDict
        )
    }

    // MARK: - Main Decode Function

    public func decode(_ segment: MLXArray, options: DecodingOptions) -> DecodingResult {
        print("[DECODER DEBUG] Input segment shape: \(segment.shape)")
        print("[DECODER DEBUG] Model dimensions - nMels: \(model.dims.nMels), nAudioCtx: \(model.dims.nAudioCtx), nAudioState: \(model.dims.nAudioState), nVocab: \(model.dims.nVocab)")
        
        let audioFeatures = model.embedAudio(segment)
        print("[DECODER DEBUG] Audio features shape: \(audioFeatures.shape)")
        print("[DECODER DEBUG] Audio features sample: \(audioFeatures[0, 0, 0..<5])")
        
        // Check for NaN or infinite values in audio features
        let audioMin = audioFeatures.min().item(Float.self)
        let audioMax = audioFeatures.max().item(Float.self)
        print("[DECODER DEBUG] Audio features range - min: \(audioMin), max: \(audioMax)")
        
        if audioMin.isNaN || audioMax.isNaN || audioMin.isInfinite || audioMax.isInfinite {
            print("[DECODER DEBUG] WARNING: Audio features contain NaN or Inf values!")
        }

        // Create initial tokens
        var initialTokens = [tokenizer.specialTokens.startOfTranscriptToken]
        print("[DECODER DEBUG] Start token: \(tokenizer.specialTokens.startOfTranscriptToken)")

        // Add language token if specified
        if let language = options.language {
            if let langToken = getLanguageToken(language) {
                initialTokens.append(langToken)
                print("[DECODER DEBUG] Added language token for \(language): \(langToken)")
            }
        }

        // Add task token
        let taskToken =
            options.task == "translate"
            ? tokenizer.specialTokens.translateToken : tokenizer.specialTokens.transcribeToken
        initialTokens.append(taskToken)
        print("[DECODER DEBUG] Added task token: \(taskToken)")

        // Add timestamp token if not suppressed
        if !options.wordTimestamps {
            initialTokens.append(tokenizer.specialTokens.noTimestampsToken)
            print("[DECODER DEBUG] Added no timestamps token: \(tokenizer.specialTokens.noTimestampsToken)")
        }

        // Add prompt tokens if provided
        if let prompt = options.prompt, !prompt.isEmpty {
            initialTokens.append(contentsOf: prompt)
            print("[DECODER DEBUG] Added prompt tokens: \(prompt)")
        }

        print("[DECODER DEBUG] Initial tokens: \(initialTokens)")

        let sampleLen = options.sampleLen ?? (model.dims.nTextCtx / 2)
        let maxTokens = sampleLen - initialTokens.count
        print("[DECODER DEBUG] Max tokens to generate: \(maxTokens)")

        // Generate tokens
        let (tokens, avgLogprob, noSpeechProb) = sampleTokens(
            audioFeatures: audioFeatures,
            initialTokens: initialTokens,
            maxTokens: maxTokens,
            temperature: options.temperature,
            suppressBlank: options.suppressBlank,
            suppressTokens: options.suppressTokens
        )

        print("[DECODER DEBUG] Generated tokens: \(tokens)")
        print("[DECODER DEBUG] Average logprob: \(avgLogprob)")
        print("[DECODER DEBUG] No speech prob: \(noSpeechProb)")

        // Decode text
        let textTokens = tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        print("[DECODER DEBUG] Text tokens (filtered): \(textTokens)")
        let text = tokenizer.decode(tokens: textTokens)
        print("[DECODER DEBUG] Decoded text: '\(text)'")

        // Calculate compression ratio
        let compressionRatio = calculateCompressionRatio(text: text)

        return DecodingResult(
            audio_features: audioFeatures,
            language: options.language ?? "en",
            tokens: tokens,
            text: text,
            avgLogprob: avgLogprob,
            noSpeechProb: noSpeechProb,
            temperature: options.temperature,
            compressionRatio: compressionRatio
        )
    }

    // MARK: - Token Sampling

    private func sampleTokens(
        audioFeatures: MLXArray,
        initialTokens: [Int],
        maxTokens: Int,
        temperature: Float,
        suppressBlank: Bool,
        suppressTokens: [Int]
    ) -> ([Int], Float, Float) {
        var tokens = initialTokens
        var logprobs: [Float] = []
        var noSpeechProb: Float = 0.0

        print("[SAMPLE DEBUG] Starting token sampling")
        print("[SAMPLE DEBUG] Initial tokens: \(initialTokens)")
        print("[SAMPLE DEBUG] Max tokens: \(maxTokens)")
        print("[SAMPLE DEBUG] Temperature: \(temperature)")

        // Get no-speech probability from first prediction
        let initialTokensArray = MLXArray(initialTokens)
        print("[SAMPLE DEBUG] Initial tokens array shape: \(initialTokensArray.shape)")
        let initialLogits = model.logits(
            tokens: initialTokensArray.expandedDimensions(axis: 0),
            audioFeatures: audioFeatures
        )
        print("[SAMPLE DEBUG] Initial logits shape: \(initialLogits.shape)")
        
        // Check initial logits for problems
        let logitsMin = initialLogits.min().item(Float.self)
        let logitsMax = initialLogits.max().item(Float.self)
        print("[SAMPLE DEBUG] Initial logits range - min: \(logitsMin), max: \(logitsMax)")
        
        if logitsMin.isNaN || logitsMax.isNaN || logitsMin.isInfinite || logitsMax.isInfinite {
            print("[SAMPLE DEBUG] WARNING: Initial logits contain NaN or Inf values!")
        }

        let noSpeechLogit = initialLogits[0, -1, tokenizer.specialTokens.noSpeechToken]
        let allLogits = initialLogits[0, -1, 0...]
        let noSpeechLogitNormalized = noSpeechLogit - logSumExp(allLogits, axis: -1)
        noSpeechProb = exp(noSpeechLogitNormalized).item(Float.self)
        print("[SAMPLE DEBUG] No speech probability: \(noSpeechProb)")

        // Generate tokens autoregressively
        var consecutiveZeros = 0
        var repetitionCount = 0
        for step in 0 ..< maxTokens {
            let currentTokens = MLXArray(tokens)
            print("[SAMPLE DEBUG] Step \(step): Current tokens: \(tokens)")
            let logits = model.logits(
                tokens: currentTokens.expandedDimensions(axis: 0),
                audioFeatures: audioFeatures
            )
            print("[SAMPLE DEBUG] Step \(step): Logits shape: \(logits.shape)")

            var nextLogits = logits[0, -1, 0...]
            print("[SAMPLE DEBUG] Step \(step): Next logits shape: \(nextLogits.shape)")

            // Apply suppressions
            if suppressBlank && tokens.count == initialTokens.count {
                // Suppress blank at the beginning
                nextLogits[tokenizer.specialTokens.whitespaceToken] = MLXArray(-Float.infinity)
                print("[SAMPLE DEBUG] Step \(step): Suppressed blank token")
            }

            for suppressToken in suppressTokens {
                if suppressToken >= 0 && suppressToken < nextLogits.size {
                    nextLogits[suppressToken] = MLXArray(-Float.infinity)
                    print("[SAMPLE DEBUG] Step \(step): Suppressed token \(suppressToken)")
                }
            }
            
            // Suppress repetitive tokens
            if tokens.count >= 3 {
                let lastThreeTokens = Array(tokens.suffix(3))
                let uniqueTokens = Set(lastThreeTokens)
                if uniqueTokens.count == 1 && uniqueTokens.first! < tokenizer.specialTokens.specialTokenBegin {
                    let repeatedToken = uniqueTokens.first!
                    if repeatedToken >= 0 && repeatedToken < nextLogits.size {
                        nextLogits[repeatedToken] = MLXArray(-Float.infinity)
                        print("[SAMPLE DEBUG] Step \(step): Suppressed repetitive token \(repeatedToken)")
                    }
                }
            }

            // Sample next token
            let nextToken: Int
            let logprob: Float

            // Debug: Check logits before sampling
            let logitsStats = (
                min: nextLogits.min().item(Float.self),
                max: nextLogits.max().item(Float.self),
                mean: nextLogits.mean().item(Float.self)
            )
            print("[SAMPLE DEBUG] Step \(step): Logits stats - min: \(logitsStats.min), max: \(logitsStats.max), mean: \(logitsStats.mean)")
            
            // Check top-k tokens
            let topK = 5
            let sortedIndices = argSort(nextLogits, axis: -1)
            let vocabSize = nextLogits.size
            let startIdx = max(0, vocabSize - topK)
            let topIndices = sortedIndices[startIdx..<vocabSize]
            let topValues = nextLogits[topIndices]
            print("[SAMPLE DEBUG] Step \(step): Top \(topK) token indices: \(topIndices)")
            print("[SAMPLE DEBUG] Step \(step): Top \(topK) token values: \(topValues)")

            if temperature > 0 {
                let scaledLogits = nextLogits / temperature
                nextToken = MLXRandom.categorical(scaledLogits).item(Int.self)
                let probs = softmax(scaledLogits, axis: -1)
                logprob = log(probs[nextToken]).item(Float.self)
                print("[SAMPLE DEBUG] Step \(step): Sampled token \(nextToken) with probability sampling, temp=\(temperature)")
                print("[SAMPLE DEBUG] Step \(step): Token \(nextToken) probability: \(probs[nextToken].item(Float.self))")
            } else {
                nextToken = argMax(nextLogits, axis: -1).item(Int.self)
                logprob =
                    nextLogits[nextToken].item(Float.self)
                    - logSumExp(nextLogits, axis: -1).item(Float.self)
                print("[SAMPLE DEBUG] Step \(step): Selected token \(nextToken) with greedy sampling")
                print("[SAMPLE DEBUG] Step \(step): Token \(nextToken) logit value: \(nextLogits[nextToken].item(Float.self))")
            }

            print("[SAMPLE DEBUG] Step \(step): Next token: \(nextToken), logprob: \(logprob)")
            
            // Validate token bounds
            if nextToken < 0 || nextToken >= model.dims.nVocab {
                print("[SAMPLE DEBUG] Step \(step): ERROR - Token \(nextToken) is out of vocabulary bounds (0..<\(model.dims.nVocab))")
                break
            }
            
            // Debug: Check if this is a special token
            if nextToken >= tokenizer.specialTokens.specialTokenBegin {
                print("[SAMPLE DEBUG] Step \(step): Generated special token \(nextToken)")
            } else if nextToken == 0 {
                consecutiveZeros += 1
                print("[SAMPLE DEBUG] Step \(step): Generated token 0 (potentially problematic), consecutive zeros: \(consecutiveZeros)")
                // Check if we're generating too many zeros in a row
                let recentTokens = Array(tokens.suffix(5))
                let zeroCount = recentTokens.filter { $0 == 0 }.count
                print("[SAMPLE DEBUG] Step \(step): Recent tokens: \(recentTokens), zero count: \(zeroCount)")
                if consecutiveZeros >= 10 {
                    print("[SAMPLE DEBUG] Step \(step): Too many consecutive zeros (\(consecutiveZeros)), breaking to prevent infinite loop")
                    break
                }
            } else {
                consecutiveZeros = 0
            }
            
            // Check for repetitive patterns
            if tokens.count >= 6 {
                let lastSix = Array(tokens.suffix(6))
                let pattern = Array(lastSix.prefix(3))
                let nextPattern = Array(lastSix.suffix(3))
                if pattern == nextPattern {
                    repetitionCount += 1
                    print("[SAMPLE DEBUG] Step \(step): Detected repetitive pattern: \(pattern), count: \(repetitionCount)")
                    if repetitionCount >= 2 {
                        print("[SAMPLE DEBUG] Step \(step): Breaking due to excessive repetition")
                        break
                    }
                } else {
                    repetitionCount = 0
                }
            }

            // Check for end token
            if nextToken == tokenizer.specialTokens.endToken {
                print("[SAMPLE DEBUG] Step \(step): Hit end token, stopping")
                break
            }

            tokens.append(nextToken)
            logprobs.append(logprob)
            
            // Additional safety check: if we're generating only zeros or whitespace tokens
            if step > 10 {
                let lastTenTokens = Array(tokens.suffix(10))
                let problemTokens = lastTenTokens.filter { $0 == 0 || $0 == tokenizer.specialTokens.whitespaceToken }
                if problemTokens.count >= 8 {
                    print("[SAMPLE DEBUG] Step \(step): Too many problem tokens in last 10 tokens, breaking")
                    break
                }
            }
        }

        let avgLogprob = logprobs.isEmpty ? 0.0 : logprobs.reduce(0, +) / Float(logprobs.count)
        print("[SAMPLE DEBUG] Final tokens: \(tokens)")
        print("[SAMPLE DEBUG] Average logprob: \(avgLogprob)")

        return (tokens, avgLogprob, noSpeechProb)
    }

    // MARK: - Helper Functions

    private func getLanguageToken(_ language: String) -> Int? {
        let languageToToken: [String: Int] = [
            "en": tokenizer.specialTokens.englishToken
            // Add more language mappings as needed
        ]
        return languageToToken[language]
    }

    private func calculateCompressionRatio(text: String) -> Float {
        guard let textData = text.data(using: .utf8) else { return 1.0 }
        let compressedData = textData.compressed(using: .zlib)
        return Float(textData.count) / Float(compressedData.count)
    }
}

// MARK: - Data Compression Extension

extension Data {
    func compressed(using algorithm: NSData.CompressionAlgorithm) -> Data {
        return self.withUnsafeBytes { bytes in
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
            defer { buffer.deallocate() }

            let compressedSize = compression_encode_buffer(
                buffer, count,
                bytes.bindMemory(to: UInt8.self).baseAddress!, count,
                nil, compression_algorithm(UInt32(algorithm.rawValue))
            )

            guard compressedSize > 0 else { return self }
            return Data(bytes: buffer, count: compressedSize)
        }
    }
}

// MARK: - Compression Algorithm Extension

// extension NSData.CompressionAlgorithm {
//     static let zlib = NSData.CompressionAlgorithm(rawValue: 1)!
// }

// MARK: - MLX Utilities

private func logSumExp(_ x: MLXArray, axis: Int) -> MLXArray {
    let maxVal = x.max(axes: [axis], keepDims: true)
    return maxVal + log(exp(x - maxVal).sum(axes: [axis], keepDims: true))
}


