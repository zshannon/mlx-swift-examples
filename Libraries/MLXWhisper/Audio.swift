import Foundation
import MLX
import MLXFFT
import MLXFast
#if canImport(AVFoundation)
import AVFoundation
#endif

public let sampleRate = 16000
public let nFFT = 400
public let hopLength = 160
public let chunkLength = 30
public let nSamples = chunkLength * sampleRate
public let nFrames = nSamples / hopLength
public let nSamplesPerToken = hopLength * 2
public let framesPerSecond = sampleRate / hopLength
public let tokensPerSecond = sampleRate / nSamplesPerToken

#if canImport(AVFoundation)
public enum WhisperAudioError: Error {
    case unableToRead
}

public func loadAudio(_ file: String, sr: Int = sampleRate) throws -> MLXArray {
    let url = URL(fileURLWithPath: file)
    let audioFile = try AVAudioFile(forReading: url)

    let inputFormat = audioFile.processingFormat
    let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(sr),
        channels: 1,
        interleaved: false)!

    let frameCapacity = AVAudioFrameCount(audioFile.length)
    guard let inputBuffer = AVAudioPCMBuffer(
        pcmFormat: inputFormat, frameCapacity: frameCapacity)
    else { throw WhisperAudioError.unableToRead }
    try audioFile.read(into: inputBuffer)

    guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat),
          let outputBuffer = AVAudioPCMBuffer(
              pcmFormat: outputFormat,
              frameCapacity: AVAudioFrameCount(
                  Double(frameCapacity) * outputFormat.sampleRate / inputFormat.sampleRate))
    else { throw WhisperAudioError.unableToRead }

    var error: NSError?
    let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
        outStatus.pointee = .haveData
        return inputBuffer
    }
    converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
    if let error { throw error }

    let samples = Array(
        UnsafeBufferPointer(start: outputBuffer.floatChannelData![0], count: Int(outputBuffer.frameLength)))
    let array = MLXArray(samples)
    print("[DEBUG] Loaded audio samples shape: \(array.shape)")
    return array
}
#else
public enum WhisperAudioError: Error { case unsupported }
public func loadAudio(_ file: String, sr: Int = sampleRate) throws -> MLXArray {
    throw WhisperAudioError.unsupported
}
#endif

public func padOrTrim(_ array: MLXArray, length: Int = nSamples, axis: Int = -1) -> MLXArray {
    var a = array
    if a.shape[axis] > length {
        // Simple slicing for the trimming case
        a = a[0..<length]
    }
    if a.shape[axis] < length {
        // Simple padding for 1D case
        a = padded(a, widths: [[0, length - a.shape[axis]]])
    }
    print("[DEBUG] padOrTrim result shape: \(a.shape)")
    return a
}

private let melCacheActor = ActorContainer([Int: MLXArray]())

actor ActorContainer<T> {
    var value: T
    
    init(_ initialValue: T) {
        self.value = initialValue
    }
    
    func get() -> T {
        return value
    }
    
    func set(_ newValue: T) {
        value = newValue
    }
    
    func update<R>(_ operation: (inout T) -> R) -> R {
        return operation(&value)
    }
}

func hzToMel(_ f: Float) -> Float {
    2595.0 * log10(1 + f / 700.0)
}

func melToHz(_ m: Float) -> Float {
    700.0 * (pow(10, m / 2595.0) - 1)
}

public func melFilters(nMels: Int, nFft: Int = nFFT) -> MLXArray {
    // Try to load from bundle resource first
    if let url = Bundle.module.url(forResource: "mel_filters", withExtension: "npz"),
       let arrays = try? loadArrays(url: url),
       let arr = arrays["mel_\(nMels)"] {
        return arr
    }
    

    
    // Simple fallback: return a basic mel filter bank approximation
    // This creates a basic triangular filter bank for mel spectrograms
    let fftBins = nFft / 2 + 1  // Only positive frequencies
    
    // Create a simple mel filter bank with correct dimensions
    var filterData: [Double] = []
    for i in 0..<nMels {
        for j in 0..<fftBins {
            // Simple triangular filters spread across frequency range
            let centerFreq = Double(i + 1) * Double(fftBins) / Double(nMels + 1)
            let bandwidth = Double(fftBins) / Double(nMels)
            let distance = abs(Double(j) - centerFreq)
            let value = max(0.0, 1.0 - distance / bandwidth)
            filterData.append(value)
        }
    }
    
    return MLXArray(converting: filterData, [nMels, fftBins])
}

func hanning(_ size: Int) -> MLXArray {
    let n = MLXArray(0..<(size + 1))
    let m = Float(size)
    let window = 0.5 - 0.5 * MLX.cos(2 * Float.pi * n / m)
    return window[0..<size]
}

public func stft(_ x: MLXArray, window: MLXArray, nperseg: Int = 256, noverlap: Int? = nil, nfft: Int? = nil, axis: Int = -1, padMode: String = "reflect") -> MLXArray {
    let nfft = nfft ?? nperseg
    let noverlap = noverlap ?? nfft / 4
    func pad(_ x: MLXArray, padding: Int, mode: String = "constant") -> MLXArray {
        if mode == "constant" { 
            return padded(x, widths: [[padding, padding]])
        }
        if mode == "reflect" {
            let prefix = x[1..<(padding+1)]
            let suffix = x[(x.size-padding-1)..<(x.size-1)]
            return concatenated([prefix, x, suffix])
        }
        fatalError("invalid pad mode")
    }
    let padding = nperseg / 2
    let x = pad(x, padding: padding, mode: padMode)
    
    // Create windowed frames manually
    let t = (x.size - nperseg + noverlap) / noverlap
    var frames: [MLXArray] = []
    for i in 0..<t {
        let start = i * noverlap
        let end = start + nperseg
        if end <= x.size {
            let frame = x[start..<end] * window
            frames.append(frame)
        }
    }
    
    // Stack frames and apply FFT
    let stackedFrames = stacked(frames, axis: 0)
    
    // Apply FFT to each frame using MLX FFT
    let fftResult = fft(stackedFrames, n: nfft, axis: -1)
    
    // Take only positive frequencies (first half + DC and Nyquist)
    let fftBins = nfft / 2 + 1
    let positiveFreqs = fftResult[0..., 0..<fftBins]
    
    return positiveFreqs
}

public func logMelSpectrogram(_ audio: MLXArray, nMels: Int = 80, padding: Int = 0) -> MLXArray {
    var a = audio
    if padding > 0 { a = padded(a, widths: [[0, padding]]) }
    let window = hanning(nFFT)
    let freqs = stft(a, window: window, nperseg: nFFT, noverlap: hopLength)
    let magnitudes = MLX.pow(abs(freqs), 2)
    let filters = melFilters(nMels: nMels, nFft: nFFT)
    let melSpec = matmul(magnitudes, filters.T)
    var logSpec = maximum(melSpec, MLXArray(1e-10)).log10()
    logSpec = maximum(logSpec, logSpec.max() - 8.0)
    logSpec = (logSpec + 4.0) / 4.0
    
    // Ensure the output has the correct shape for Whisper: [nMels, nFrames]
    // and pad/trim to expected length
    let targetFrames = nFrames  // 3000 frames for 30 seconds of audio
    let currentFrames = logSpec.shape[0]
    
    if currentFrames < targetFrames {
        // Pad with zeros
        let padding = targetFrames - currentFrames
        logSpec = concatenated([logSpec, MLXArray.zeros([padding, nMels])], axis: 0)
    } else if currentFrames > targetFrames {
        // Trim to target size
        logSpec = logSpec[0..<targetFrames, 0...]
    }
    
    // Transpose to get [nMels, nFrames] shape expected by Whisper
    let result = logSpec.T
    print("[DEBUG] logMelSpectrogram result shape: \(result.shape)")
    return result
}
