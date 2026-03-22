module VisionTransformers

import Flux

using Match, ArgCheck, Statistics
using Pipe: @pipe


include("layers/utils.jl")
export img2seq, seq2img

include("layers/base.jl")
export MLP, SeparableConv, Tokens, StripTokens

include("layers/embeddings.jl")
export AbsolutePositionEmbedding, RelativePositionEmbedding, VariablePositionEmbedding

include("layers/attention.jl")
export MultiHeadAttention, ConvAttention

include("models/utils.jl")
include("models/vit.jl")
include("models/cvt.jl")
include("models/pvt.jl")
include("models/swin.jl")
include("models/twins.jl")
export ViT, ViTBlock, CvT, CvTBlock, PVT, PVTBlock, SWIN, SWINBlock, Twins, TwinsBlock

end