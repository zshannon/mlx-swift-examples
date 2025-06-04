// Copyright © 2024 Apple Inc.

import MLX
import MLXNN

class StubMultiHeadAttention: Module {
    let nState: Int
    let nHead: Int

    init(nState: Int, nHead: Int) {
        self.nState = nState
        self.nHead = nHead
        super.init()
    }
}

class StubResidualAttentionBlock: Module {
    let nState: Int
    let nHead: Int
    let crossAttention: Bool

    init(nState: Int, nHead: Int, crossAttention: Bool = false) {
        self.nState = nState
        self.nHead = nHead
        self.crossAttention = crossAttention
        super.init()
    }
}

class StubAudioEncoder: Module {
    let nMels: Int
    let nCtx: Int
    let nState: Int
    let nHead: Int
    let nLayer: Int
    let dtype: DType
    let blocks: [StubResidualAttentionBlock]

    init(nMels: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int, dtype: DType = .float16) {
        self.nMels = nMels
        self.nCtx = nCtx
        self.nState = nState
        self.nHead = nHead
        self.nLayer = nLayer
        self.dtype = dtype
        self.blocks = (0..<nLayer).map { _ in StubResidualAttentionBlock(nState: nState, nHead: nHead) }
        super.init()
    }
}

class StubTextDecoder: Module {
    let nVocab: Int
    let nCtx: Int
    let nState: Int
    let nHead: Int
    let nLayer: Int
    let dtype: DType
    let blocks: [StubResidualAttentionBlock]

    init(nVocab: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int, dtype: DType = .float16) {
        self.nVocab = nVocab
        self.nCtx = nCtx
        self.nState = nState
        self.nHead = nHead
        self.nLayer = nLayer
        self.dtype = dtype
        self.blocks = (0..<nLayer).map { _ in StubResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: true) }
        super.init()
    }
}

public class WhisperStub: Module {
    public let dims: ModelDimensions
    let encoder: StubAudioEncoder
    let decoder: StubTextDecoder

    public init(dims: ModelDimensions, dtype: DType = .float16) {
        self.dims = dims
        self.encoder = StubAudioEncoder(nMels: dims.nMels, nCtx: dims.nAudioCtx, nState: dims.nAudioState, nHead: dims.nAudioHead, nLayer: dims.nAudioLayer, dtype: dtype)
        self.decoder = StubTextDecoder(nVocab: dims.nVocab, nCtx: dims.nTextCtx, nState: dims.nTextState, nHead: dims.nTextHead, nLayer: dims.nTextLayer, dtype: dtype)
        
        super.init()
    }

    public var isMultilingual: Bool { dims.nVocab >= 51865 }
    public var numLanguages: Int { dims.nVocab - 51765 - (isMultilingual ? 1 : 0) }
}