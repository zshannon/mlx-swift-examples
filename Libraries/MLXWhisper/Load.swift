import Foundation
@preconcurrency import Hub
import MLX
import MLXNN

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
    var weightsURL = url.appending(component: "weights.safetensors")
    if !FileManager.default.fileExists(atPath: weightsURL.path) {
        weightsURL = url.appending(component: "weights.npz")
    }
    let weights = try MLXArray.load(contentsOf: weightsURL)
    try model.update(
        parameters: ModuleParameters.unflattened(weights.mapValues { $0.asType(dtype) }),
        verify: .none)
    mx.eval(model.parameters())
    return model
}

func prepareModelDirectory(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> URL {
    do {
        switch configuration.id {
        case .id(let id):
            let repo = Hub.Repo(id: id)
            let modelFiles = ["weights.safetensors", "weights.npz", "config.json"]
            return try await hub.snapshot(
                from: repo, matching: modelFiles, progressHandler: progressHandler)

        case .directory(let directory):
            return directory
        }
    } catch Hub.HubClientError.authorizationRequired {
        return configuration.modelDirectory(hub: hub)
    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            return configuration.modelDirectory(hub: hub)
        } else {
            throw error
        }
    }
}

public func loadModel(
    hub: HubApi = HubApi(), configuration: ModelConfiguration,
    dtype: DType = .float16,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> Whisper {
    let dir = try await prepareModelDirectory(
        hub: hub, configuration: configuration, progressHandler: progressHandler)
    return try loadModel(path: dir, dtype: dtype)
}
