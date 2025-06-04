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
    let logTimescaleIncrement = MLX.log(MLXArray(maxTimescale)) / Float(channels / 2 - 1)
    let invTimescales = MLX.exp(-logTimescaleIncrement * MLXArray(stride(from: 0, to: channels/2, by: 1)))
    let scaledTime = MLXArray(stride(from: 0, to: length, by: 1))[0..., .newAxis] * invTimescales[.newAxis, 0...]
    return concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: 1)
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
        
        super.init()
        
        self._query.wrappedValue = Linear(nState, nState)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState)
        self._out.wrappedValue = Linear(nState, nState)
    }

    func callAsFunction(_ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil, kvCache: (MLXArray, MLXArray)? = nil) -> (MLXArray, (MLXArray, MLXArray), MLXArray) {
        let q = query(x)
        var k: MLXArray
        var v: MLXArray
        if xa == nil {
            k = key(x)
            v = value(x)
            if let cache = kvCache {
                if cache.0.size > 0 {
                    k = concatenated([cache.0, k], axis: 1)
                }
                if cache.1.size > 0 {
                    v = concatenated([cache.1, v], axis: 1)
                }
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
        let scale = MLX.pow(MLXArray(Float(nState / nHead)), MLXArray(-0.25))
        let q = q.reshaped([nBatch, nCtx, nHead, nState / nHead]).transposed(0,2,1,3) * scale
        let k = k.reshaped([nBatch, k.shape[1], nHead, nState / nHead]).transposed(0,2,3,1) * scale
        let v = v.reshaped([nBatch, v.shape[1], nHead, nState / nHead]).transposed(0,2,1,3)
        let qk = matmul(q, k)
        if let m = mask { qk += m[0..<nCtx, 0..<nCtx] }
        let w = softmax(qk, axis: -1)
        let out = matmul(w, v).transposed(0,2,1,3).reshaped([nBatch, nCtx, nState])
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
        let nMlp = nState * 4
        
        super.init()
        
        self._attn.wrappedValue = MultiHeadAttention(nState: nState, nHead: nHead)
        self._attnLn.wrappedValue = LayerNorm(dimensions: nState)
        
        if crossAttention {
            self._crossAttn.wrappedValue = MultiHeadAttention(nState: nState, nHead: nHead)
            self._crossAttnLn.wrappedValue = LayerNorm(dimensions: nState)
        }
        
        self._mlp1.wrappedValue = Linear(nState, nMlp)
        self._mlp2.wrappedValue = Linear(nMlp, nState)
        self._mlpLn.wrappedValue = LayerNorm(dimensions: nState)
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
    private let nCtx: Int
    private let nState: Int
    private let dtype: DType
    private var _positionalEmbedding: MLXArray?
    var blocks: [ResidualAttentionBlock]
    @ModuleInfo var lnPost: LayerNorm
    
    var positionalEmbedding: MLXArray {
        if let cached = _positionalEmbedding {
            return cached
        }
        let embedding = sinusoids(length: nCtx, channels: nState, maxTimescale: 10000).asType(dtype)
        _positionalEmbedding = embedding
        return embedding
    }

    init(nMels: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int, dtype: DType = .float16) {
        self.nCtx = nCtx
        self.nState = nState
        self.dtype = dtype
        self.blocks = (0..<nLayer).map { _ in ResidualAttentionBlock(nState: nState, nHead: nHead) }
        
        super.init()
        
        self._conv1.wrappedValue = Conv1d(inputChannels: nMels, outputChannels: nState, kernelSize: 3, padding: 1)
        self._conv2.wrappedValue = Conv1d(inputChannels: nState, outputChannels: nState, kernelSize: 3, stride: 2, padding: 1)
        self._lnPost.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLX Conv1d expects input in NLC format: [batch, length, channels]
        var x = x
        
        // Convert mel spectrogram to proper format for MLX Conv1d
        if x.shape.count == 2 && x.shape[0] == 80 {
            // x is [nMels, nFrames], transpose to [nFrames, nMels], then add batch
            x = x.T  // [3000, 80]
            x = expandedDimensions(x, axis: 0)  // [1, 3000, 80]
        } else if x.shape.count == 2 {
            // Generic 2D case: add batch dimension
            x = expandedDimensions(x, axis: 0)
        }
        
        x = gelu(conv1(x))
        x = gelu(conv2(x))
        
        // x is now [batch, seq_len, nState] which is correct for transformer blocks
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
    private let nCtx: Int
    private let nState: Int
    private let dtype: DType
    private var _positionalEmbedding: MLXArray?
    private var _mask: MLXArray?
    var blocks: [ResidualAttentionBlock]
    @ModuleInfo var ln: LayerNorm
    
    var positionalEmbedding: MLXArray {
        if let cached = _positionalEmbedding {
            return cached
        }
        let embedding = MLXArray.zeros([nCtx, nState])
        _positionalEmbedding = embedding
        return embedding
    }
    
    var mask: MLXArray {
        if let cached = _mask {
            return cached
        }
        let indices = MLXArray(0 ..< nCtx)
        var mask = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
        mask = mask.asType(dtype) * -1e9
        _mask = mask
        return mask
    }

    init(nVocab: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int, dtype: DType = .float16) {
        self.nCtx = nCtx
        self.nState = nState
        self.dtype = dtype
        self.blocks = (0..<nLayer).map { _ in ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: true) }
        
        super.init()
        
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: nVocab, dimensions: nState)
        self._ln.wrappedValue = LayerNorm(dimensions: nState)
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
    private var _alignmentHeads: MLXArray?
    
    public var alignmentHeads: MLXArray {
        if let cached = _alignmentHeads {
            return cached
        }
        let allHeads = MLXArray.zeros([dims.nTextLayer, dims.nTextHead])
        let start = dims.nTextLayer / 2
        let heads: MLXArray
        if start < dims.nTextLayer {
            let onesSection = MLXArray.ones([dims.nTextLayer - start, dims.nTextHead])
            let topSection = allHeads[..<start, 0...]
            heads = concatenated([topSection, onesSection], axis: 0).asType(.bool)
        } else {
            heads = allHeads.asType(.bool)
        }
        _alignmentHeads = heads
        return heads
    }

    public init(dims: ModelDimensions, dtype: DType = .float16) {
        self.dims = dims
        
        super.init()
        
        self._encoder.wrappedValue = AudioEncoder(nMels: dims.nMels, nCtx: dims.nAudioCtx, nState: dims.nAudioState, nHead: dims.nAudioHead, nLayer: dims.nAudioLayer, dtype: dtype)
        self._decoder.wrappedValue = TextDecoder(nVocab: dims.nVocab, nCtx: dims.nTextCtx, nState: dims.nTextState, nHead: dims.nTextHead, nLayer: dims.nTextLayer, dtype: dtype)
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

