import Foundation
import MLX
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
    return MLXArray(samples)
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
        let sl: [Slice?] = Array(repeating: nil, count: a.ndim).enumerated().map { i, _ in i == axis ? Slice(0..<length) : nil }
        a = a[sl]
    }
    if a.shape[axis] < length {
        var pad = Array(repeating: (0,0), count: a.ndim)
        pad[axis] = (0, length - a.shape[axis])
        a = padArray(a, pad)
    }
    return a
}

private var melCache = [Int: MLXArray]()

func hzToMel(_ f: Float) -> Float {
    2595.0 * log10(1 + f / 700.0)
}

func melToHz(_ m: Float) -> Float {
    700.0 * (pow(10, m / 2595.0) - 1)
}

public func melFilters(nMels: Int) -> MLXArray {
    if let m = melCache[nMels] { return m }

    let sr = Float(sampleRate)
    let fMax = sr / 2.0
    let mMin = hzToMel(0)
    let mMax = hzToMel(fMax)

    // mel scale points
    var mels = [Float]()
    for i in 0..<(nMels + 2) {
        let mel = mMin + Float(i) / Float(nMels + 1) * (mMax - mMin)
        mels.append(melToHz(mel))
    }

    // fft frequencies
    let nFftBins = nFFT / 2 + 1
    var fftFreqs = [Float](repeating: 0, count: nFftBins)
    for i in 0..<nFftBins {
        fftFreqs[i] = sr * Float(i) / Float(nFFT)
    }

    var weights = [Float](repeating: 0, count: nMels * nFftBins)

    for i in 0..<nMels {
        let lower = mels[i]
        let center = mels[i + 1]
        let upper = mels[i + 2]
        let fdiff1 = center - lower
        let fdiff2 = upper - center
        for j in 0..<nFftBins {
            let freq = fftFreqs[j]
            let l = (freq - lower) / fdiff1
            let u = (upper - freq) / fdiff2
            let idx = i * nFftBins + j
            weights[idx] = max(0, min(l, u))
        }
        let enorm = 2.0 / (upper - lower)
        for j in 0..<nFftBins {
            weights[i * nFftBins + j] *= enorm
        }
    }

    let arr = MLXArray(converting: weights, [nMels, nFftBins])
    melCache[nMels] = arr
    return arr
}

func hanning(_ size: Int) -> MLXArray {
    let n = MLXArray(arange: size + 1, type: .float32)
    let m = Float(size)
    let window = 0.5 - 0.5 * cos(2 * .pi * n / m)
    return window[0..<size]
}

public func stft(_ x: MLXArray, window: MLXArray, nperseg: Int = 256, noverlap: Int? = nil, nfft: Int? = nil, axis: Int = -1, padMode: String = "reflect") -> MLXArray {
    let nfft = nfft ?? nperseg
    let noverlap = noverlap ?? nfft / 4
    func pad(_ x: MLXArray, padding: Int, mode: String = "constant") -> MLXArray {
        if mode == "constant" { return padArray(x, [(padding,padding)]) }
        if mode == "reflect" {
            let prefix = x[1..<(padding+1)][reversed: true]
            let suffix = x[(x.count-padding-1)..<(x.count-1)][reversed: true]
            return concatenated([prefix,x,suffix])
        }
        fatalError("invalid pad mode")
    }
    let padding = nperseg / 2
    var x = pad(x, padding: padding, mode: padMode)
    let strides = [noverlap, 1]
    let t = (x.size - nperseg + noverlap) / noverlap
    let shape = [t, nfft]
    x = asStrided(x, shape: shape, strides: strides)
    return rfft(x * window)
}

public func logMelSpectrogram(_ audio: MLXArray, nMels: Int = 80, padding: Int = 0) -> MLXArray {
    var a = audio
    if padding > 0 { a = padArray(a, [(0,padding)]) }
    let window = hanning(nFFT)
    let freqs = stft(a, window: window, nperseg: nFFT, noverlap: hopLength)
    var magnitudes = abs(freqs[:-1, :]) ** 2
    let filters = melFilters(nMels: nMels)
    var melSpec = magnitudes @ filters.T
    var logSpec = max(melSpec, 1e-10).log10()
    logSpec = max(logSpec, logSpec.max() - 8.0)
    logSpec = (logSpec + 4.0) / 4.0
    return logSpec
}
