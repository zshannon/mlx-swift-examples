import Foundation
@preconcurrency import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import Compression

/// Custom loadArrays function that supports both .safetensors and .npz files
private func loadArraysWhisper(url: URL) throws -> [String: MLXArray] {
    switch url.pathExtension {
    case "safetensors":
        return try loadArrays(url: url)
    case "npz":
        return try loadNPZ(url: url)
    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
}

/// Load arrays from NPZ file (zip archive containing .npy files)
private func loadNPZ(url: URL) throws -> [String: MLXArray] {
    let data = try Data(contentsOf: url)
    var result: [String: MLXArray] = [:]
    
    // Create a temporary directory to extract NPZ contents
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("whisper_npz_\(UUID().uuidString)")
    
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer {
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    // Write NPZ data to temporary file
    let tempNPZ = tempDir.appendingPathComponent("temp.npz")
    try data.write(to: tempNPZ)
    
    // Use unzip to extract (NPZ is a zip file)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
    process.arguments = ["-q", tempNPZ.path, "-d", tempDir.path]
    
    do {
        try process.run()
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            throw NSError(domain: "NPZError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to extract NPZ file"])
        }
    } catch {
        throw NSError(domain: "NPZError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to run unzip: \(error)"])
    }
    
    // Load all .npy files from extracted directory
    let enumerator = FileManager.default.enumerator(at: tempDir, includingPropertiesForKeys: nil)!
    for case let fileURL as URL in enumerator {
        if fileURL.pathExtension == "npy" {
            let arrayName = fileURL.deletingPathExtension().lastPathComponent
            let array = try loadArray(url: fileURL)
            result[arrayName] = array
        }
    }
    
    return result
}

/// Map Python-style parameter names to Swift MLX convention
private func mapParameterName(_ pythonName: String) -> String {
    var name = pythonName
    
    // Map decoder/encoder structure
    if name.hasPrefix("decoder.") {
        name = name.replacingOccurrences(of: "decoder.", with: "_decoder.")
    } else if name.hasPrefix("encoder.") {
        name = name.replacingOccurrences(of: "encoder.", with: "_encoder.")
    }
    
    // Map block indices from Python to Swift array syntax
    name = name.replacingOccurrences(of: "blocks.", with: "blocks[")
    
    // Close array indices and map subsequent dots
    let blockPattern = #"blocks\[(\d+)\]\.(.+)"#
    if let regex = try? NSRegularExpression(pattern: blockPattern, options: []) {
        let range = NSRange(location: 0, length: name.count)
        name = regex.stringByReplacingMatches(
            in: name,
            options: [],
            range: range,
            withTemplate: "blocks[$1].$2"
        )
    }
    
    // Map attention layer names
    name = name.replacingOccurrences(of: ".attn.", with: "._attn.")
    name = name.replacingOccurrences(of: ".cross_attn.", with: "._crossAttn.")
    name = name.replacingOccurrences(of: ".attn_ln.", with: "._attnLn.")
    name = name.replacingOccurrences(of: ".cross_attn_ln.", with: "._crossAttnLn.")
    name = name.replacingOccurrences(of: ".mlp_ln.", with: "._mlpLn.")
    
    // Map query/key/value/out to MLX naming
    name = name.replacingOccurrences(of: ".query.", with: "._query.")
    name = name.replacingOccurrences(of: ".key.", with: "._key.")
    name = name.replacingOccurrences(of: ".value.", with: "._value.")
    name = name.replacingOccurrences(of: ".out.", with: "._out.")
    
    // Map MLP layers
    name = name.replacingOccurrences(of: ".mlp1.", with: "._mlp1.")
    name = name.replacingOccurrences(of: ".mlp2.", with: "._mlp2.")
    
    // Map conv layers
    name = name.replacingOccurrences(of: ".conv1.", with: "._conv1.")
    name = name.replacingOccurrences(of: ".conv2.", with: "._conv2.")
    
    // Map layer norm names
    name = name.replacingOccurrences(of: ".ln_post.", with: "._lnPost.")
    name = name.replacingOccurrences(of: ".ln.", with: "._ln.")
    
    // Map token embedding
    name = name.replacingOccurrences(of: ".token_embedding.", with: "._tokenEmbedding.")
    
    return name
}

private enum LoadSaveError: Error {
    case unknownExtension(String)
}

extension LoadSaveError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .unknownExtension(let ext):
            return "Unknown extension \(ext)"
        }
    }
}

/// Download the whisper model using the `HubApi`.
///
/// This will download `*.safetensors`, `*.npz` and `*.json` files if the ``ModelConfiguration``
/// represents a Hub id, e.g. `mlx-community/whisper-tiny`.
///
/// - Parameters:
///   - hub: HubApi instance
///   - configuration: the model identifier
///   - progressHandler: callback for progress
/// - Returns: URL for the directory containing downloaded files
public func downloadModel(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    do {
        switch configuration.id {
        case .id(let id):
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors", "*.npz", "*.json"]
            
            let result = try await hub.snapshot(
                from: repo, matching: modelFiles, progressHandler: progressHandler)
            return result

        case .directory(let directory):
            return directory
        }

    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        let localDir = configuration.modelDirectory(hub: hub)
        return localDir

    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            // Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."
            // fall back to the local directory
            let localDir = configuration.modelDirectory(hub: hub)
            return localDir
        } else {
            throw error
        }
    }
}

/// Load whisper model weights.
///
/// This function loads all `safetensor` and `npz` files in the given `modelDirectory`,
/// and updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: Whisper,
    dtype: DType = .float16
) throws {
    // load the weights
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" || url.pathExtension == "npz" {
            let w = try loadArraysWhisper(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }
    
    // Map Python-style parameter names to Swift MLX convention
    var mappedWeights: [String: MLXArray] = [:]
    for (key, value) in weights {
        let mappedKey = mapParameterName(key)
        mappedWeights[mappedKey] = value.asType(dtype)
    }
    
    let parameters = ModuleParameters.unflattened(mappedWeights)
    
    try model.update(parameters: parameters, verify: [])
    eval(model)
}

/// Load tokenizer for whisper model.
/// Creates a Whisper-specific tokenizer with proper special tokens.
///
/// - Parameters:
///   - configuration: model configuration
///   - hub: HubApi instance
/// - Returns: loaded tokenizer
public func loadTokenizer(configuration: ModelConfiguration, hub: HubApi = HubApi()) async throws -> Tokenizer {
    do {
        // Try to use the standard tokenizer loading from MLXLMCommon
        let tokenizer = try await MLXLMCommon.loadTokenizer(configuration: configuration, hub: hub)
        return tokenizer
    } catch {
        // For Whisper models, create a proper Whisper tokenizer
        // Whisper uses specific special tokens for transcription
        let whisperTokenizer = try await AutoTokenizer.from(pretrained: "openai/whisper-tiny")
        return whisperTokenizer
    }
}

public func loadModel(path: URL, dtype: DType = .float16) throws -> Whisper {
    let url = path

    let configURL = url.appending(component: "config.json")
    let data = try Data(contentsOf: configURL)
    var json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    
    json.removeValue(forKey: "model_type")
    json.removeValue(forKey: "quantization")
    
    let dims = ModelDimensions(
        nMels: json["n_mels"] as! Int,
        nAudioCtx: json["n_audio_ctx"] as! Int,
        nAudioState: json["n_audio_state"] as! Int,
        nAudioHead: json["n_audio_head"] as! Int,
        nAudioLayer: json["n_audio_layer"] as! Int,
        nVocab: json["n_vocab"] as! Int,
        nTextCtx: json["n_text_ctx"] as! Int,
        nTextState: json["n_text_state"] as! Int,
        nTextHead: json["n_text_head"] as! Int,
        nTextLayer: json["n_text_layer"] as! Int
    )
    
    let model = Whisper(dims: dims, dtype: dtype)
    try loadWeights(modelDirectory: url, model: model, dtype: dtype)
    
    return model
}

func prepareModelDirectory(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> URL {
    let result = try await downloadModel(hub: hub, configuration: configuration, progressHandler: progressHandler)
    return result
}

public func loadModel(
    hub: HubApi = HubApi(), configuration: ModelConfiguration,
    dtype: DType = .float16,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> Whisper {
    let dir = try await prepareModelDirectory(
        hub: hub, configuration: configuration, progressHandler: progressHandler)
    
    let model = try loadModel(path: dir, dtype: dtype)
    return model
}
