import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

public actor WhisperModelContainer {
    let model: Whisper
    let tokenizer: WhisperTokenizer
    public let configuration: ModelConfiguration

    init(model: Whisper, tokenizer: WhisperTokenizer, configuration: ModelConfiguration) {
        self.model = model
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    public func transcribe(
        file path: String,
        language: String? = nil,
        task: String = "transcribe",
        temperature: [Float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compressionRatioThreshold: Float? = 2.4,
        logprobThreshold: Float? = -1.0,
        noSpeechThreshold: Float? = 0.6,
        conditionOnPreviousText: Bool = true,
        initialPrompt: String? = nil,
        wordTimestamps: Bool = false,
        verbose: Bool = false
    ) throws -> TranscriptionResult {
        let audio = try loadAudio(path)
        print("[DEBUG] Audio array shape: \(audio.shape)")

        // Create mel spectrogram with proper padding for windowing
        let mel = logMelSpectrogram(audio, nMels: model.dims.nMels, padding: nSamples)
        print("[DEBUG] Mel spectrogram shape: \(mel.shape)")

        let totalFrames = mel.shape[1]
        let contentFrames = max(0, totalFrames - nFrames)
        let contentDuration = Float(contentFrames * hopLength) / Float(sampleRate)
        
        print("[DEBUG] nFrames: \(nFrames)")
        print("[DEBUG] totalFrames: \(totalFrames)")
        print("[DEBUG] contentFrames: \(contentFrames)")
        print("[DEBUG] contentDuration: \(contentDuration)")
        print("[DEBUG] hopLength: \(hopLength)")
        print("[DEBUG] sampleRate: \(sampleRate)")

        // Language detection if not specified
        let detectedLanguage: String = "en"
        // if let lang = language {
        //     detectedLanguage = lang
        // } else {
        //     if !model.isMultilingual {
        //         detectedLanguage = "en"
        //     } else {
        //         if verbose {
        //             print("Detecting language using up to the first 30 seconds...")
        //         }
        //         let melSegment = padOrTrim(mel[0..., 0 ..< nFrames].T, length: nFrames).T
        //         let decoder = WhisperDecoder(model: model, tokenizer: tokenizer)
        //         let detection = decoder.detectLanguage(melSegment.expandedDimensions(axis: 0))
        //         detectedLanguage = detection.language
        //         if verbose {
        //             print("Detected language: \(detectedLanguage)")
        //         }
        //     }
        // }

        // Initialize variables for windowed processing
        var seek = 0
        var allTokens: [Int] = []
        var allSegments: [TranscriptionSegment] = []
        var promptResetSince = 0
        
        print("[DEBUG] Starting windowed processing with contentFrames: \(contentFrames)")

        // Add initial prompt tokens if provided
        if let prompt = initialPrompt {
            let promptTokens = tokenizer.encode(
                text: " " + prompt.trimmingCharacters(in: .whitespaces))
            allTokens.append(contentsOf: promptTokens)
        }

        let decoder = WhisperDecoder(model: model, tokenizer: tokenizer)

        // Process audio in 30-second windows
        while seek < totalFrames {
            print("[DEBUG] Processing window at seek: \(seek), totalFrames: \(totalFrames)")
            let timeOffset = Float(seek * hopLength) / Float(sampleRate)
            let segmentSize = min(nFrames, totalFrames - seek)
            print("[DEBUG] Time offset: \(timeOffset), segment size: \(segmentSize)")

            // Extract mel segment
            let melStart = seek
            let melEnd = min(seek + segmentSize, mel.shape[1])
            print("[DEBUG] Extracting mel segment from \(melStart) to \(melEnd)")
            var melSegment = mel[0..., melStart..<melEnd]
            print("[DEBUG] Extracted mel segment shape: \(melSegment.shape)")

            // Pad to nFrames if necessary
            if melSegment.shape[1] < nFrames {
                let padding = nFrames - melSegment.shape[1]
                melSegment = concatenated(
                    [melSegment, MLXArray.zeros([melSegment.shape[0], padding])], axis: 1)
            }

            melSegment = melSegment.expandedDimensions(axis: 0)
            print("[DEBUG] Mel segment shape for decoding: \(melSegment.shape)")
            
            // Test encoder directly
            let audioFeatures = model.embedAudio(melSegment)
            print("[DEBUG] Audio features shape: \(audioFeatures.shape)")
            print("[DEBUG] Audio features sample values: \(audioFeatures[0, 0, 0..<5])")
            
            // Prepare decoding options
            let prompt = conditionOnPreviousText ? Array(allTokens[promptResetSince...]) : []
            print("[DEBUG] Using prompt tokens: \(prompt)")
            
            // Try different temperatures until success
            var result: DecodingResult?
            for temp in temperature {
                print("[DEBUG] Trying temperature: \(temp)")
                let options = DecodingOptions(
                    task: task,
                    language: detectedLanguage,
                    temperature: temp,
                    suppressBlank: true,
                    conditionOnPreviousText: conditionOnPreviousText,
                    prompt: prompt.isEmpty ? nil : prompt
                )
                
                let tempResult = decoder.decode(melSegment, options: options)
                print("[DEBUG] Decode result - tokens: \(tempResult.tokens.count), avgLogprob: \(tempResult.avgLogprob), noSpeechProb: \(tempResult.noSpeechProb)")

                // Check quality thresholds
                var needsFallback = false

                if let threshold = compressionRatioThreshold,
                    tempResult.compressionRatio > threshold
                {
                    needsFallback = true  // Too repetitive
                }

                if let threshold = logprobThreshold,
                    tempResult.avgLogprob < threshold
                {
                    needsFallback = true  // Average log probability too low
                }

                if let threshold = noSpeechThreshold,
                    tempResult.noSpeechProb > threshold
                {
                    needsFallback = false  // Silence is okay
                }

                if !needsFallback {
                    result = tempResult
                    print("[DEBUG] Accepted result with temperature \(temp)")
                    break
                } else {
                    print("[DEBUG] Rejected result with temperature \(temp): compression=\(tempResult.compressionRatio), logprob=\(tempResult.avgLogprob), noSpeech=\(tempResult.noSpeechProb)")
                }
            }

            guard let finalResult = result else {
                print("[DEBUG] No good result found, skipping segment")
                // Skip if no good result found
                seek += segmentSize
                print("[DEBUG] Updated seek to: \(seek)")
                continue
            }
            
            print("[DEBUG] Final result tokens: \(finalResult.tokens)")
            print("[DEBUG] Final result text: '\(finalResult.text)'")
            
            // Check for no speech
            if let threshold = noSpeechThreshold,
               finalResult.noSpeechProb > threshold {
                print("[DEBUG] Skipping segment due to no speech threshold: \(finalResult.noSpeechProb) > \(threshold)")
                seek += segmentSize
                print("[DEBUG] Updated seek to: \(seek)")
                continue
            }

            // Process tokens to find segments
            let tokens = finalResult.tokens
            let segmentDuration = Float(segmentSize * hopLength) / Float(sampleRate)

            // Look for timestamp tokens to create segments
            let timestampTokens = tokens.enumerated().map { (index, token) in
                (index, token >= tokenizer.specialTokens.timeTokenBegin)
            }

            let timestampIndices = timestampTokens.compactMap { (index, isTimestamp) in
                isTimestamp ? index : nil
            }

            if timestampIndices.count >= 2 {
                // Create segments based on timestamps
                for i in 0 ..< (timestampIndices.count - 1) {
                    let startIdx = timestampIndices[i]
                    let endIdx = timestampIndices[i + 1]

                    let segmentTokens = Array(tokens[startIdx ..< endIdx])
                    let startToken = tokens[startIdx]
                    let endToken = tokens[endIdx]

                    let startTime =
                        timeOffset + Float(startToken - tokenizer.specialTokens.timeTokenBegin)
                        * 0.02
                    let endTime =
                        timeOffset + Float(endToken - tokenizer.specialTokens.timeTokenBegin) * 0.02

                    let textTokens = segmentTokens.filter {
                        $0 < tokenizer.specialTokens.specialTokenBegin
                    }
                    let text = tokenizer.decode(tokens: textTokens)

                    let segment = TranscriptionSegment(
                        id: allSegments.count,
                        seek: seek,
                        start: startTime,
                        end: endTime,
                        text: text,
                        tokens: segmentTokens,
                        temperature: finalResult.temperature,
                        avgLogprob: finalResult.avgLogprob,
                        compressionRatio: finalResult.compressionRatio,
                        noSpeechProb: finalResult.noSpeechProb
                    )

                    allSegments.append(segment)
                }

                // Update seek based on last timestamp
                if let lastTimestampIdx = timestampIndices.last {
                    let lastToken = tokens[lastTimestampIdx]
                    let timestampPos = lastToken - tokenizer.specialTokens.timeTokenBegin
                    seek += timestampPos * 2  // 2 frames per timestamp token
                }
            } else {
                // No timestamps, create single segment for entire window
                let textTokens = tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                let text = tokenizer.decode(tokens: textTokens)

                let segment = TranscriptionSegment(
                    id: allSegments.count,
                    seek: seek,
                    start: timeOffset,
                    end: timeOffset + segmentDuration,
                    text: text,
                    tokens: tokens,
                    temperature: finalResult.temperature,
                    avgLogprob: finalResult.avgLogprob,
                    compressionRatio: finalResult.compressionRatio,
                    noSpeechProb: finalResult.noSpeechProb
                )

                allSegments.append(segment)
                seek += segmentSize
            }

            // Add tokens to history
            allTokens.append(contentsOf: tokens)

            // Reset prompt if temperature was high or condition disabled
            if !conditionOnPreviousText || finalResult.temperature > 0.5 {
                promptResetSince = allTokens.count
            }

            if verbose {
                if let lastSegment = allSegments.last {
                    print(
                        "[\(formatTimestamp(lastSegment.start)) --> \(formatTimestamp(lastSegment.end))] \(lastSegment.text)"
                    )
                }
            }
        }

        // Combine all text
        let finalText = allSegments.map { $0.text }.joined()
        
        print("[DEBUG] Final processing complete. Total segments: \(allSegments.count)")
        print("[DEBUG] Final text length: \(finalText.count)")
        
        return TranscriptionResult(
            text: finalText.trimmingCharacters(in: .whitespacesAndNewlines),
            segments: allSegments,
            language: detectedLanguage
        )
    }

    private func formatTimestamp(_ seconds: Float) -> String {
        let totalMilliseconds = Int(seconds * 1000)
        let hours = totalMilliseconds / 3_600_000
        let minutes = (totalMilliseconds % 3_600_000) / 60_000
        let secs = (totalMilliseconds % 60_000) / 1_000
        let milliseconds = totalMilliseconds % 1_000

        let hoursMarker = hours > 0 ? String(format: "%02d:", hours) : ""
        return String(format: "%@%02d:%02d.%03d", hoursMarker, minutes, secs, milliseconds)
    }
}

public class WhisperModelFactory: @unchecked Sendable {
    public static let shared = WhisperModelFactory()

    public let modelRegistry: AbstractModelRegistry

    public init(modelRegistry: AbstractModelRegistry = WhisperRegistry.shared) {
        self.modelRegistry = modelRegistry
    }

    public func loadContainer(
        hub: HubApi = HubApi(), configuration: ModelConfiguration,
        dtype: DType = .float16,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> WhisperModelContainer {
        let model = try await loadModel(
            hub: hub, configuration: configuration, dtype: dtype,
            progressHandler: progressHandler)

        // Use our custom WhisperTokenizer instead of loading from files
        let tokenizer = WhisperTokenizer()

        let container = WhisperModelContainer(
            model: model, tokenizer: tokenizer, configuration: configuration)
        return container
    }
}
