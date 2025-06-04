import MLX
import MLXNN

/// Dimensions for a Whisper model
public struct ModelDimensions: Sendable {
    public var nMels: Int
    public var nAudioCtx: Int
    public var nAudioState: Int
    public var nAudioHead: Int
    public var nAudioLayer: Int
    public var nVocab: Int
    public var nTextCtx: Int
    public var nTextState: Int
    public var nTextHead: Int
    public var nTextLayer: Int
}

func sinusoids(length: Int, channels: Int, maxTimescale: Float = 10000) -> MLXArray {
    precondition(channels % 2 == 0)
    let logTimescaleIncrement = log(maxTimescale) / Float(channels / 2 - 1)
    let invTimescales = exp(-logTimescaleIncrement * MLXArray(arange: channels/2))
    let scaledTime = MLXArray(arange: length)[..., .newAxis] * invTimescales[.newAxis, ...]
    return concatenated([sin(scaledTime), cos(scaledTime)], axis: 1)[0]
}

class MultiHeadAttention: Module {
    let nState: Int
    let nHead: Int
    @ModuleInfo var query: Linear
    @ModuleInfo var key: Linear
    @ModuleInfo var value: Linear
    @ModuleInfo var out: Linear

    init(nState: Int, nHead: Int) {
        self.nState = nState
        self.nHead = nHead
        self._query = Linear(nState, nState)
        self._key = Linear(nState, nState, bias: false)
        self._value = Linear(nState, nState)
        self._out = Linear(nState, nState)
    }

    func callAsFunction(_ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil, kvCache: (MLXArray, MLXArray)? = nil) -> (MLXArray, (MLXArray, MLXArray), MLXArray) {
        var q = query(x)
        var k: MLXArray
        var v: MLXArray
        if xa == nil {
            k = key(x)
            v = value(x)
            if let cache = kvCache {
                k = concatenated([cache.0, k], axis: 1)
                v = concatenated([cache.1, v], axis: 1)
            }
        } else if kvCache == nil {
            k = key(xa!)
            v = value(xa!)
        } else {
            k = kvCache!.0
            v = kvCache!.1
        }
        let (wv, qk) = qkvAttention(q: q, k: k, v: v, mask: mask)
        return (out(wv), (k,v), qk)
    }

    func qkvAttention(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let nBatch = q.shape[0]
        let nCtx = q.shape[1]
        let scale = pow(Float(nState / nHead), -0.25)
        var q = q.reshaped([nBatch, nCtx, nHead, -1]).transposed(0,2,1,3) * scale
        var k = k.reshaped([nBatch, -1, nHead, -1]).transposed(0,2,3,1) * scale
        var v = v.reshaped([nBatch, -1, nHead, -1]).transposed(0,2,1,3)
        var qk = q @ k
        if let m = mask { qk += m[0..<nCtx, 0..<nCtx] }
        let w = softmax(qk, axis: -1)
        var out = (w @ v).transposed(0,2,1,3)
        out = out.reshaped([nBatch, nCtx, nState])
        return (out, qk)
    }
}

class ResidualAttentionBlock: Module {
    @ModuleInfo var attn: MultiHeadAttention
    @ModuleInfo var attnLn: LayerNorm
    @ModuleInfo var crossAttn: MultiHeadAttention?
    @ModuleInfo var crossAttnLn: LayerNorm?
    @ModuleInfo var mlp1: Linear
    @ModuleInfo var mlp2: Linear
    @ModuleInfo var mlpLn: LayerNorm

    init(nState: Int, nHead: Int, crossAttention: Bool = false) {
        self._attn = MultiHeadAttention(nState: nState, nHead: nHead)
        self._attnLn = LayerNorm(nState)
        if crossAttention {
            self._crossAttn = MultiHeadAttention(nState: nState, nHead: nHead)
            self._crossAttnLn = LayerNorm(nState)
        }
        let nMlp = nState * 4
        self._mlp1 = Linear(nState, nMlp)
        self._mlp2 = Linear(nMlp, nState)
        self._mlpLn = LayerNorm(nState)
    }

    func callAsFunction(_ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil, kvCache: (MLXArray, MLXArray)? = nil) -> (MLXArray, (MLXArray, MLXArray)?, MLXArray?) {
        var kvCache = kvCache
        var y: MLXArray
        var crossQK: MLXArray? = nil
        (y, kvCache, _) = attn(attnLn(x), mask: mask, kvCache: kvCache)
        var out = x + y
        if let cross = crossAttn {
            var crossKVCache: (MLXArray, MLXArray)? = nil
            (y, crossKVCache, crossQK) = cross(crossAttnLn!(out), xa: xa, kvCache: kvCache)
            out += y
            kvCache = crossKVCache
        }
        out += mlp2(gelu(mlp1(mlpLn(out))))
        return (out, kvCache, crossQK)
    }
}

class AudioEncoder: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    var positionalEmbedding: MLXArray
    var blocks: [ResidualAttentionBlock]
    @ModuleInfo var lnPost: LayerNorm

    init(nMels: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int, dtype: DType = .float16) {
        self._conv1 = Conv1d(inChannels: nMels, outChannels: nState, kernelSize: 3, padding: 1)
        self._conv2 = Conv1d(inChannels: nState, outChannels: nState, kernelSize: 3, stride: 2, padding: 1)
        positionalEmbedding = sinusoids(length: nCtx, channels: nState, maxTimescale: 10000).asType(dtype)
        blocks = (0..<nLayer).map { _ in ResidualAttentionBlock(nState: nState, nHead: nHead) }
        self._lnPost = LayerNorm(nState)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = gelu(conv1(x))
        x = gelu(conv2(x))
        assert(x.shape[1] == positionalEmbedding.shape[0])
        x += positionalEmbedding
        for block in blocks {
            (x, _, _) = block(x)
        }
        return lnPost(x)
    }
}

class TextDecoder: Module {
    @ModuleInfo var tokenEmbedding: Embedding
    var positionalEmbedding: MLXArray
    var blocks: [ResidualAttentionBlock]
    @ModuleInfo var ln: LayerNorm
    var mask: MLXArray

    init(nVocab: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int, dtype: DType = .float16) {
        self._tokenEmbedding = Embedding(nVocab, nState)
        self.positionalEmbedding = MLXArray(zeros: [nCtx, nState])
        blocks = (0..<nLayer).map { _ in ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: true) }
        self._ln = LayerNorm(nState)
        self.mask = MultiHeadAttention.createAdditiveCausalMask(length: nCtx).asType(dtype)
    }

    func callAsFunction(_ x: MLXArray, xa: MLXArray, kvCache: [(MLXArray, MLXArray)]? = nil) -> (MLXArray, [(MLXArray, MLXArray)], [MLXArray]) {
        let offset = kvCache?.first?.0.shape[1] ?? 0
        var x = tokenEmbedding(x) + positionalEmbedding[offset..<offset+x.shape[1]]
        var cache = kvCache ?? Array(repeating: (MLXArray(), MLXArray()), count: blocks.count)
        var crossQK: [MLXArray] = []
        for i in 0..<blocks.count {
            let (y, newKV, qk) = blocks[i](x, xa: xa, mask: mask, kvCache: cache[i])
            x = y
            cache[i] = newKV ?? cache[i]
            crossQK.append(qk ?? MLXArray())
        }
        x = ln(x)
        let logits = tokenEmbedding.asLinear(x)
        return (logits, cache, crossQK)
    }
}

public class Whisper: Module {
    public let dims: ModelDimensions
    @ModuleInfo var encoder: AudioEncoder
    @ModuleInfo var decoder: TextDecoder
    public var alignmentHeads: MLXArray

    public init(dims: ModelDimensions, dtype: DType = .float16) {
        self.dims = dims
        self._encoder = AudioEncoder(nMels: dims.nMels, nCtx: dims.nAudioCtx, nState: dims.nAudioState, nHead: dims.nAudioHead, nLayer: dims.nAudioLayer, dtype: dtype)
        self._decoder = TextDecoder(nVocab: dims.nVocab, nCtx: dims.nTextCtx, nState: dims.nTextState, nHead: dims.nTextHead, nLayer: dims.nTextLayer, dtype: dtype)
        let allHeads = MLXArray(zeros: [dims.nTextLayer, dims.nTextHead], type: .bool)
        var arr = allHeads
        let start = dims.nTextLayer / 2
        if start < dims.nTextLayer {
            arr[start..., ...] = MLXArray(repeating: 1, shape: [dims.nTextLayer - start, dims.nTextHead], type: .bool)
        }
        self.alignmentHeads = MLXArray(nonZeroIndices: arr)
    }

    public var isMultilingual: Bool { dims.nVocab >= 51865 }
    public var numLanguages: Int { dims.nVocab - 51765 - (isMultilingual ? 1 : 0) }

    public func embedAudio(_ mel: MLXArray) -> MLXArray { encoder(mel) }
    public func logits(tokens: MLXArray, audioFeatures: MLXArray) -> MLXArray { decoder(tokens, xa: audioFeatures).0 }
    public func forwardWithCrossQK(mel: MLXArray, tokens: MLXArray) -> (MLXArray, [MLXArray]) {
        let (logits, _, qk) = decoder(tokens, xa: encoder(mel))
        return (logits, qk)
    }
    public func callAsFunction(mel: MLXArray, tokens: MLXArray) -> MLXArray { decoder(tokens, xa: encoder(mel)).0 }
}

