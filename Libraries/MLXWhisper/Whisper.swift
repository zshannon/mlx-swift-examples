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
    let invTimescales = MLX.exp(
        -logTimescaleIncrement * MLXArray(stride(from: 0, to: channels / 2, by: 1)))
    let scaledTime =
        MLXArray(stride(from: 0, to: length, by: 1))[0..., .newAxis] * invTimescales[.newAxis, 0...]
    return concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: 1)
}

// class MultiHeadAttention: Module {
//     let nState: Int
//     let nHead: Int
//     @ModuleInfo var query: Linear
//     @ModuleInfo var key: Linear
//     @ModuleInfo var value: Linear
//     @ModuleInfo var out: Linear

//     init(nState: Int, nHead: Int) {
//         self.nState = nState
//         self.nHead = nHead

//         super.init()

//         self._query.wrappedValue = Linear(nState, nState)
//         self._key.wrappedValue = Linear(nState, nState, bias: false)
//         self._value.wrappedValue = Linear(nState, nState)
//         self._out.wrappedValue = Linear(nState, nState)
//     }

//     func callAsFunction(_ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil, kvCache: (MLXArray, MLXArray)? = nil) -> (MLXArray, (MLXArray, MLXArray), MLXArray) {
//         print("[ATTN DEBUG] Input x range: [\(x.min().item(Float.self)), \(x.max().item(Float.self))]")
//         if let xa = xa {
//             print("[ATTN DEBUG] Input xa range: [\(xa.min().item(Float.self)), \(xa.max().item(Float.self))]")
//         }

//         let q = query(x)
//         let qMin = q.min().item(Float.self)
//         let qMax = q.max().item(Float.self)
//         print("[ATTN DEBUG] Query range: [\(qMin), \(qMax)]")
//         if qMin.isNaN || qMax.isNaN || qMin.isInfinite || qMax.isInfinite {
//             print("[ATTN DEBUG] WARNING: Query contains NaN/Inf!")
//         }

//         var k: MLXArray
//         var v: MLXArray
//         if xa == nil {
//             k = key(x)
//             v = value(x)
//             if let cache = kvCache {
//                 if cache.0.size > 0 {
//                     k = concatenated([cache.0, k], axis: 1)
//                 }
//                 if cache.1.size > 0 {
//                     v = concatenated([cache.1, v], axis: 1)
//                 }
//             }
//         } else if kvCache == nil {
//             k = key(xa!)
//             v = value(xa!)
//         } else {
//             k = kvCache!.0
//             v = kvCache!.1
//         }

//         let kMin = k.min().item(Float.self)
//         let kMax = k.max().item(Float.self)
//         print("[ATTN DEBUG] Key range: [\(kMin), \(kMax)]")
//         if kMin.isNaN || kMax.isNaN || kMin.isInfinite || kMax.isInfinite {
//             print("[ATTN DEBUG] WARNING: Key contains NaN/Inf!")
//         }

//         let vMin = v.min().item(Float.self)
//         let vMax = v.max().item(Float.self)
//         print("[ATTN DEBUG] Value range: [\(vMin), \(vMax)]")
//         if vMin.isNaN || vMax.isNaN || vMin.isInfinite || vMax.isInfinite {
//             print("[ATTN DEBUG] WARNING: Value contains NaN/Inf!")
//         }

//         let (wv, qk) = qkvAttention(q: q, k: k, v: v, mask: mask)
//         let wvMin = wv.min().item(Float.self)
//         let wvMax = wv.max().item(Float.self)
//         print("[ATTN DEBUG] Attention output range: [\(wvMin), \(wvMax)]")
//         if wvMin.isNaN || wvMax.isNaN || wvMin.isInfinite || wvMax.isInfinite {
//             print("[ATTN DEBUG] WARNING: Attention output contains NaN/Inf!")
//         }

//         let result = out(wv)
//         let resultMin = result.min().item(Float.self)
//         let resultMax = result.max().item(Float.self)
//         print("[ATTN DEBUG] Final output range: [\(resultMin), \(resultMax)]")
//         if resultMin.isNaN || resultMax.isNaN || resultMin.isInfinite || resultMax.isInfinite {
//             print("[ATTN DEBUG] WARNING: Final output contains NaN/Inf!")
//         }

//         return (result, (k,v), qk)
//     }

//     func qkvAttention(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray? = nil) -> (MLXArray, MLXArray) {
//         let nBatch = q.shape[0]
//         let nCtx = q.shape[1]
//         print("[QKV DEBUG] Batch: \(nBatch), Context: \(nCtx), nState: \(nState), nHead: \(nHead)")

//         let scale = MLX.pow(MLXArray(Float(nState / nHead)), MLXArray(-0.25))
//         print("[QKV DEBUG] Scale value: \(scale.item(Float.self))")

//         let q = q.reshaped([nBatch, nCtx, nHead, nState / nHead]).transposed(0,2,1,3) * scale
//         let qScaledMin = q.min().item(Float.self)
//         let qScaledMax = q.max().item(Float.self)
//         print("[QKV DEBUG] Scaled Q range: [\(qScaledMin), \(qScaledMax)]")
//         if qScaledMin.isNaN || qScaledMax.isNaN || qScaledMin.isInfinite || qScaledMax.isInfinite {
//             print("[QKV DEBUG] WARNING: Scaled Q contains NaN/Inf!")
//         }

//         let k = k.reshaped([nBatch, k.shape[1], nHead, nState / nHead]).transposed(0,2,3,1) * scale
//         let kScaledMin = k.min().item(Float.self)
//         let kScaledMax = k.max().item(Float.self)
//         print("[QKV DEBUG] Scaled K range: [\(kScaledMin), \(kScaledMax)]")
//         if kScaledMin.isNaN || kScaledMax.isNaN || kScaledMin.isInfinite || kScaledMax.isInfinite {
//             print("[QKV DEBUG] WARNING: Scaled K contains NaN/Inf!")
//         }

//         let v = v.reshaped([nBatch, v.shape[1], nHead, nState / nHead]).transposed(0,2,1,3)
//         let vReshapedMin = v.min().item(Float.self)
//         let vReshapedMax = v.max().item(Float.self)
//         print("[QKV DEBUG] Reshaped V range: [\(vReshapedMin), \(vReshapedMax)]")

//         let qk = matmul(q, k)
//         let qkMin = qk.min().item(Float.self)
//         let qkMax = qk.max().item(Float.self)
//         print("[QKV DEBUG] QK scores range: [\(qkMin), \(qkMax)]")
//         if qkMin.isNaN || qkMax.isNaN || qkMin.isInfinite || qkMax.isInfinite {
//             print("[QKV DEBUG] WARNING: QK scores contain NaN/Inf!")
//         }

//         if let m = mask {
//             print("[QKV DEBUG] Applying mask")
//             qk += m[0..<nCtx, 0..<nCtx]
//             let qkMaskedMin = qk.min().item(Float.self)
//             let qkMaskedMax = qk.max().item(Float.self)
//             print("[QKV DEBUG] Masked QK range: [\(qkMaskedMin), \(qkMaskedMax)]")
//             if qkMaskedMin.isNaN || qkMaskedMax.isNaN || qkMaskedMin.isInfinite || qkMaskedMax.isInfinite {
//                 print("[QKV DEBUG] WARNING: Masked QK contains NaN/Inf!")
//             }
//         }

//         let w = softmax(qk, axis: -1)
//         let wMin = w.min().item(Float.self)
//         let wMax = w.max().item(Float.self)
//         print("[QKV DEBUG] Softmax weights range: [\(wMin), \(wMax)]")
//         if wMin.isNaN || wMax.isNaN || wMin.isInfinite || wMax.isInfinite {
//             print("[QKV DEBUG] WARNING: Softmax weights contain NaN/Inf!")
//         }

//         let out = matmul(w, v).transposed(0,2,1,3).reshaped([nBatch, nCtx, nState])
//         let outMin = out.min().item(Float.self)
//         let outMax = out.max().item(Float.self)
//         print("[QKV DEBUG] Final attention output range: [\(outMin), \(outMax)]")
//         if outMin.isNaN || outMax.isNaN || outMin.isInfinite || outMax.isInfinite {
//             print("[QKV DEBUG] WARNING: Final attention output contains NaN/Inf!")
//         }

//         return (out, qk)
//     }
// }

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

        self._attn.wrappedValue = MultiHeadAttention(dimensions: nState, numHeads: nHead, bias: false)
        self._attnLn.wrappedValue = LayerNorm(dimensions: nState)

        if crossAttention {
            self._crossAttn.wrappedValue = MultiHeadAttention(dimensions: nState, numHeads: nHead, bias: false)
            self._crossAttnLn.wrappedValue = LayerNorm(dimensions: nState)
        }

        self._mlp1.wrappedValue = Linear(nState, nMlp)
        self._mlp2.wrappedValue = Linear(nMlp, nState)
        self._mlpLn.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(
        _ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?, MLXArray?) {
        print(
            "[BLOCK DEBUG] Input x range: [\(x.min().item(Float.self)), \(x.max().item(Float.self))]"
        )

        var out = x

        // Self attention
        let attnLnOut = attnLn(x)
        let attnLnMin = attnLnOut.min().item(Float.self)
        let attnLnMax = attnLnOut.max().item(Float.self)
        print("[BLOCK DEBUG] After attnLn: [\(attnLnMin), \(attnLnMax)]")
        if attnLnMin.isNaN || attnLnMax.isNaN || attnLnMin.isInfinite || attnLnMax.isInfinite {
            print("[BLOCK DEBUG] WARNING: attnLn output contains NaN/Inf!")
        }

        // Use MLXNN MultiHeadAttention with self-attention (same input for Q, K, V)
        let selfAttnOut = attn(attnLnOut, keys: attnLnOut, values: attnLnOut, mask: mask)
        let attnMin = selfAttnOut.min().item(Float.self)
        let attnMax = selfAttnOut.max().item(Float.self)
        print("[BLOCK DEBUG] After self attention: [\(attnMin), \(attnMax)]")
        if attnMin.isNaN || attnMax.isNaN || attnMin.isInfinite || attnMax.isInfinite {
            print("[BLOCK DEBUG] WARNING: Self attention output contains NaN/Inf!")
        }

        out = out + selfAttnOut
        let afterSelfAttnMin = out.min().item(Float.self)
        let afterSelfAttnMax = out.max().item(Float.self)
        print(
            "[BLOCK DEBUG] After self attention residual: [\(afterSelfAttnMin), \(afterSelfAttnMax)]"
        )
        if afterSelfAttnMin.isNaN || afterSelfAttnMax.isNaN || afterSelfAttnMin.isInfinite
            || afterSelfAttnMax.isInfinite
        {
            print("[BLOCK DEBUG] WARNING: After self attention residual contains NaN/Inf!")
        }

        // Cross attention
        if let cross = crossAttn, let xa = xa {
            let crossAttnLnOut = crossAttnLn!(out)
            let crossAttnLnMin = crossAttnLnOut.min().item(Float.self)
            let crossAttnLnMax = crossAttnLnOut.max().item(Float.self)
            print("[BLOCK DEBUG] After crossAttnLn: [\(crossAttnLnMin), \(crossAttnLnMax)]")
            if crossAttnLnMin.isNaN || crossAttnLnMax.isNaN || crossAttnLnMin.isInfinite
                || crossAttnLnMax.isInfinite
            {
                print("[BLOCK DEBUG] WARNING: crossAttnLn output contains NaN/Inf!")
            }

            // Cross attention: queries from decoder, keys/values from encoder
            let crossAttnOut = cross(crossAttnLnOut, keys: xa, values: xa, mask: nil)
            let crossAttnMin = crossAttnOut.min().item(Float.self)
            let crossAttnMax = crossAttnOut.max().item(Float.self)
            print("[BLOCK DEBUG] After cross attention: [\(crossAttnMin), \(crossAttnMax)]")
            if crossAttnMin.isNaN || crossAttnMax.isNaN || crossAttnMin.isInfinite
                || crossAttnMax.isInfinite
            {
                print("[BLOCK DEBUG] WARNING: Cross attention output contains NaN/Inf!")
            }

            out = out + crossAttnOut
            let afterCrossAttnMin = out.min().item(Float.self)
            let afterCrossAttnMax = out.max().item(Float.self)
            print(
                "[BLOCK DEBUG] After cross attention residual: [\(afterCrossAttnMin), \(afterCrossAttnMax)]"
            )
            if afterCrossAttnMin.isNaN || afterCrossAttnMax.isNaN || afterCrossAttnMin.isInfinite
                || afterCrossAttnMax.isInfinite
            {
                print("[BLOCK DEBUG] WARNING: After cross attention residual contains NaN/Inf!")
            }
        }

        // MLP
        let mlpLnOut = mlpLn(out)
        let mlpLnMin = mlpLnOut.min().item(Float.self)
        let mlpLnMax = mlpLnOut.max().item(Float.self)
        print("[BLOCK DEBUG] After mlpLn: [\(mlpLnMin), \(mlpLnMax)]")
        if mlpLnMin.isNaN || mlpLnMax.isNaN || mlpLnMin.isInfinite || mlpLnMax.isInfinite {
            print("[BLOCK DEBUG] WARNING: mlpLn output contains NaN/Inf!")
        }

        let mlp1Out = mlp1(mlpLnOut)
        let mlp1Min = mlp1Out.min().item(Float.self)
        let mlp1Max = mlp1Out.max().item(Float.self)
        print("[BLOCK DEBUG] After mlp1: [\(mlp1Min), \(mlp1Max)]")
        if mlp1Min.isNaN || mlp1Max.isNaN || mlp1Min.isInfinite || mlp1Max.isInfinite {
            print("[BLOCK DEBUG] WARNING: mlp1 output contains NaN/Inf!")
        }

        let geluOut = gelu(mlp1Out)
        let geluMin = geluOut.min().item(Float.self)
        let geluMax = geluOut.max().item(Float.self)
        print("[BLOCK DEBUG] After gelu: [\(geluMin), \(geluMax)]")
        if geluMin.isNaN || geluMax.isNaN || geluMin.isInfinite || geluMax.isInfinite {
            print("[BLOCK DEBUG] WARNING: gelu output contains NaN/Inf!")
        }

        let mlp2Out = mlp2(geluOut)
        let mlp2Min = mlp2Out.min().item(Float.self)
        let mlp2Max = mlp2Out.max().item(Float.self)
        print("[BLOCK DEBUG] After mlp2: [\(mlp2Min), \(mlp2Max)]")
        if mlp2Min.isNaN || mlp2Max.isNaN || mlp2Min.isInfinite || mlp2Max.isInfinite {
            print("[BLOCK DEBUG] WARNING: mlp2 output contains NaN/Inf!")
        }

        out = out + mlp2Out
        let finalMin = out.min().item(Float.self)
        let finalMax = out.max().item(Float.self)
        print("[BLOCK DEBUG] Final output: [\(finalMin), \(finalMax)]")
        if finalMin.isNaN || finalMax.isNaN || finalMin.isInfinite || finalMax.isInfinite {
            print("[BLOCK DEBUG] WARNING: Final output contains NaN/Inf!")
        }

        // Note: KV caching not implemented yet for simplicity
        return (out, nil, nil)
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
        self.blocks = (0 ..< nLayer).map { _ in ResidualAttentionBlock(nState: nState, nHead: nHead)
        }

        super.init()

        self._conv1.wrappedValue = Conv1d(
            inputChannels: nMels, outputChannels: nState, kernelSize: 3, padding: 1)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: nState, outputChannels: nState, kernelSize: 3, stride: 2, padding: 1)
        self._lnPost.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLX Conv1d expects input in NLC format: [batch, length, channels]
        var x = x
        print("[ENCODER DEBUG] Input shape: \(x.shape)")
        let inputMin = x.min().item(Float.self)
        let inputMax = x.max().item(Float.self)
        print("[ENCODER DEBUG] Input range: min=\(inputMin), max=\(inputMax)")

        // Convert mel spectrogram to proper format for MLX Conv1d
        if x.shape.count == 2 && x.shape[0] == 80 {
            // x is [nMels, nFrames], transpose to [nFrames, nMels], then add batch
            x = x.T  // [3000, 80]
            x = expandedDimensions(x, axis: 0)  // [1, 3000, 80]
        } else if x.shape.count == 3 && x.shape[1] == 80 {
            // x is [batch, nMels, nFrames], transpose to [batch, nFrames, nMels]
            x = x.transposed(0, 2, 1)  // [batch, nFrames, nMels]
        } else if x.shape.count == 2 {
            // Generic 2D case: add batch dimension
            x = expandedDimensions(x, axis: 0)
        }
        print("[ENCODER DEBUG] After reshape: \(x.shape)")

        x = gelu(conv1(x))
        let conv1Min = x.min().item(Float.self)
        let conv1Max = x.max().item(Float.self)
        print(
            "[ENCODER DEBUG] After conv1+gelu: shape=\(x.shape), range=[\(conv1Min), \(conv1Max)]")
        if conv1Min.isNaN || conv1Max.isNaN || conv1Min.isInfinite || conv1Max.isInfinite {
            print("[ENCODER DEBUG] WARNING: Conv1 output contains NaN/Inf!")
        }

        x = gelu(conv2(x))
        let conv2Min = x.min().item(Float.self)
        let conv2Max = x.max().item(Float.self)
        print(
            "[ENCODER DEBUG] After conv2+gelu: shape=\(x.shape), range=[\(conv2Min), \(conv2Max)]")
        if conv2Min.isNaN || conv2Max.isNaN || conv2Min.isInfinite || conv2Max.isInfinite {
            print("[ENCODER DEBUG] WARNING: Conv2 output contains NaN/Inf!")
        }

        // x is now [batch, seq_len, nState] which is correct for transformer blocks
        print(
            "[ENCODER DEBUG] Before positional embedding: x.shape[1]=\(x.shape[1]), positionalEmbedding.shape[0]=\(positionalEmbedding.shape[0])"
        )
        assert(x.shape[1] == positionalEmbedding.shape[0])
        x += positionalEmbedding
        let posEmbMin = x.min().item(Float.self)
        let posEmbMax = x.max().item(Float.self)
        print("[ENCODER DEBUG] After positional embedding: range=[\(posEmbMin), \(posEmbMax)]")
        if posEmbMin.isNaN || posEmbMax.isNaN || posEmbMin.isInfinite || posEmbMax.isInfinite {
            print("[ENCODER DEBUG] WARNING: After positional embedding contains NaN/Inf!")
        }

        for (i, block) in blocks.enumerated() {
            (x, _, _) = block(x)
            let blockMin = x.min().item(Float.self)
            let blockMax = x.max().item(Float.self)
            print("[ENCODER DEBUG] After block \(i): range=[\(blockMin), \(blockMax)]")
            if blockMin.isNaN || blockMax.isNaN || blockMin.isInfinite || blockMax.isInfinite {
                print("[ENCODER DEBUG] WARNING: Block \(i) output contains NaN/Inf!")
                break
            }
        }

        let result = lnPost(x)
        let finalMin = result.min().item(Float.self)
        let finalMax = result.max().item(Float.self)
        print(
            "[ENCODER DEBUG] Final output: shape=\(result.shape), range=[\(finalMin), \(finalMax)]")
        if finalMin.isNaN || finalMax.isNaN || finalMin.isInfinite || finalMax.isInfinite {
            print("[ENCODER DEBUG] WARNING: Final output contains NaN/Inf!")
        }

        return result
    }
}

class TextDecoder: Module {
    @ModuleInfo var tokenEmbedding: Embedding
    @ModuleInfo var positionalEmbedding: Embedding
    private let nCtx: Int
    private let nState: Int
    private let dtype: DType
    private var _mask: MLXArray?
    var blocks: [ResidualAttentionBlock]
    @ModuleInfo var ln: LayerNorm

    func createCausalMask(sequenceLength: Int) -> MLXArray {
        let indices = MLXArray(0 ..< sequenceLength)
        var mask = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
        mask = mask.asType(dtype) * -1e9
        return mask
    }

    init(nVocab: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int, dtype: DType = .float16) {
        self.nCtx = nCtx
        self.nState = nState
        self.dtype = dtype
        self.blocks = (0 ..< nLayer).map { _ in
            ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: true)
        }

        super.init()

        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: nVocab, dimensions: nState)
        self._positionalEmbedding.wrappedValue = Embedding(embeddingCount: nCtx, dimensions: nState)
        self._ln.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(_ x: MLXArray, xa: MLXArray, kvCache: [(MLXArray, MLXArray)]? = nil) -> (
        MLXArray, [(MLXArray, MLXArray)], [MLXArray]
    ) {
        print("[DECODER DEBUG] Input tokens shape: \(x.shape)")
        print("[DECODER DEBUG] Audio features shape: \(xa.shape)")
        let xaMin = xa.min().item(Float.self)
        let xaMax = xa.max().item(Float.self)
        print("[DECODER DEBUG] Audio features range: [\(xaMin), \(xaMax)]")

        let offset = kvCache?.first?.0.shape[1] ?? 0
        let tokenEmb = tokenEmbedding(x)
        let tokenEmbMin = tokenEmb.min().item(Float.self)
        let tokenEmbMax = tokenEmb.max().item(Float.self)
        print("[DECODER DEBUG] Token embedding range: [\(tokenEmbMin), \(tokenEmbMax)]")
        if tokenEmbMin.isNaN || tokenEmbMax.isNaN || tokenEmbMin.isInfinite
            || tokenEmbMax.isInfinite
        {
            print("[DECODER DEBUG] WARNING: Token embedding contains NaN/Inf!")
        }

        let posIndices = MLXArray(offset ..< offset + x.shape[1])
        let posEmb = positionalEmbedding(posIndices)
        let posEmbMin = posEmb.min().item(Float.self)
        let posEmbMax = posEmb.max().item(Float.self)
        print("[DECODER DEBUG] Positional embedding range: [\(posEmbMin), \(posEmbMax)]")

        var x = tokenEmb + posEmb
        let inputMin = x.min().item(Float.self)
        let inputMax = x.max().item(Float.self)
        print("[DECODER DEBUG] After token+pos embedding: [\(inputMin), \(inputMax)]")
        if inputMin.isNaN || inputMax.isNaN || inputMin.isInfinite || inputMax.isInfinite {
            print("[DECODER DEBUG] WARNING: Combined embedding contains NaN/Inf!")
        }

        var cache = kvCache ?? Array(repeating: (MLXArray(), MLXArray()), count: blocks.count)
        var crossQK: [MLXArray] = []
        for i in 0..<blocks.count {
            let currentMask = createCausalMask(sequenceLength: x.shape[1])
            let (y, _, _) = blocks[i](x, xa: xa, mask: currentMask, kvCache: cache[i])
            x = y
            let blockMin = x.min().item(Float.self)
            let blockMax = x.max().item(Float.self)
            print("[DECODER DEBUG] After decoder block \(i): [\(blockMin), \(blockMax)]")
            if blockMin.isNaN || blockMax.isNaN || blockMin.isInfinite || blockMax.isInfinite {
                print("[DECODER DEBUG] WARNING: Decoder block \(i) output contains NaN/Inf!")
                break
            }
            // KV caching temporarily disabled
            crossQK.append(MLXArray())
        }

        x = ln(x)
        let lnMin = x.min().item(Float.self)
        let lnMax = x.max().item(Float.self)
        print("[DECODER DEBUG] After layer norm: [\(lnMin), \(lnMax)]")
        if lnMin.isNaN || lnMax.isNaN || lnMin.isInfinite || lnMax.isInfinite {
            print("[DECODER DEBUG] WARNING: Layer norm output contains NaN/Inf!")
        }

        let logits = tokenEmbedding.asLinear(x)
        let logitsMin = logits.min().item(Float.self)
        let logitsMax = logits.max().item(Float.self)
        print("[DECODER DEBUG] Final logits range: [\(logitsMin), \(logitsMax)]")
        if logitsMin.isNaN || logitsMax.isNaN || logitsMin.isInfinite || logitsMax.isInfinite {
            print("[DECODER DEBUG] WARNING: Final logits contain NaN/Inf!")
        }

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

        self._encoder.wrappedValue = AudioEncoder(
            nMels: dims.nMels, nCtx: dims.nAudioCtx, nState: dims.nAudioState,
            nHead: dims.nAudioHead, nLayer: dims.nAudioLayer, dtype: dtype)
        self._decoder.wrappedValue = TextDecoder(
            nVocab: dims.nVocab, nCtx: dims.nTextCtx, nState: dims.nTextState,
            nHead: dims.nTextHead, nLayer: dims.nTextLayer, dtype: dtype)
    }

    public var isMultilingual: Bool { dims.nVocab >= 51865 }
    public var numLanguages: Int { dims.nVocab - 51765 - (isMultilingual ? 1 : 0) }

    public func embedAudio(_ mel: MLXArray) -> MLXArray { encoder(mel) }
    public func logits(tokens: MLXArray, audioFeatures: MLXArray) -> MLXArray {
        decoder(tokens, xa: audioFeatures).0
    }
    public func forwardWithCrossQK(mel: MLXArray, tokens: MLXArray) -> (MLXArray, [MLXArray]) {
        let (logits, _, qk) = decoder(tokens, xa: encoder(mel))
        return (logits, qk)
    }
    public func callAsFunction(mel: MLXArray, tokens: MLXArray) -> MLXArray {
        decoder(tokens, xa: encoder(mel)).0
    }
}
