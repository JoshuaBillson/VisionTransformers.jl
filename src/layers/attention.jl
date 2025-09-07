struct LinearProjection{D<:Flux.Dense}
    qkv::D
end

Flux.@layer :expand LinearProjection

function LinearProjection(dim::Integer; qkv_bias=false)
    return LinearProjection(Flux.Dense(dim=>dim*3; bias=qkv_bias))
end

(m::LinearProjection)(x::AbstractArray) = Flux.chunk(m.qkv(x), 3; dims=1)

"""
    MultiHeadAttentionModule(qkv, proj; nheads=8, attn_dropout_prob=0.0, proj_dropout_prob=0.0)

"""
struct MultiHeadAttention{Q,P,D1,D2}
    nheads::Int
    qkv::Q
    proj::P
    attn_drop::D1
    proj_drop::D2
end

Flux.@layer :expand MultiHeadAttention

function MultiHeadAttention(dim::Integer; nheads=8, attn_dropout_prob=0.0, proj_dropout_prob=0.0, qkv_bias=false)
    return MultiHeadAttention(
        nheads, 
        Flux.Dense(dim => dim*3; bias=qkv_bias),
        Flux.Dense(dim=>dim),
        Flux.Dropout(attn_dropout_prob), 
        Flux.Dropout(proj_dropout_prob)
    )
end

function (m::MultiHeadAttention)(x::AbstractArray{<:Real,3})
    q, k, v = Flux.chunk(m.qkv(x), 3; dims=1)
    y, α = Flux.dot_product_attention(q, k, v; m.nheads, fdrop=m.attn_drop)
    return m.proj(y) |> m.proj_drop
end

"""
struct SeparableConv{D,N,C<:Flux.Conv{D},P}
    conv::C
    proj::P
    norm::N
end

Flux.@layer :expand SeparableConv

function SeparableConv(dim, kernel::NTuple{N,Int}, stride::NTuple{N,Int}; norm=:BN, bias=false) where N
    @argcheck norm in (:LN, :BN)
    return SeparableConv(
        Flux.DepthwiseConv(kernel, dim=>dim; pad=Flux.SamePad(), stride, bias=false), 
        Flux.Dense(dim=>dim; bias),
        norm == :BN ? Flux.BatchNorm(dim) : Flux.LayerNorm(dim)
    )
end

function (m::SeparableConv{D,<:Flux.BatchNorm})(x::AbstractArray{<:Real,N}) where {D,N}
    C = size(x, D+1)
    Obs = size(x, N)
    @pipe m.conv(x) |> m.norm |> permutedims(_, (D+1, 1:D..., N)) |> m.proj |> reshape(_, (C,:,Obs))
end

function (m::SeparableConv{D,<:Flux.LayerNorm})(x::AbstractArray{<:Real,N}) where {D,N}
    C = size(x, D+1)
    Obs = size(x, N)
    @pipe m.conv(x) |> permutedims(_, (D+1, 1:D..., N)) |> m.norm |> m.proj |> reshape(_, (C,:,Obs))
end
"""

function SeparableConv(dim, kernel::NTuple{D,Int}, stride::NTuple{D,Int}; bias=false) where D
    Flux.Chain(
        Flux.DepthwiseConv(kernel, dim=>dim; pad=Flux.SamePad(), stride, bias=false), 
        Flux.BatchNorm(dim),
        Base.Fix2(permutedims, (D+1, 1:D..., D+2)),
        Flux.Dense(dim=>dim; bias)
    )
end

struct ConvProjection{Q,K,V}
    q::Q
    k::K
    v::V
end

Flux.@layer :expand ConvProjection

function ConvProjection(dim::Integer; kernel=(3,3), q_stride=(1,1), kv_stride=(1,1), qkv_bias=false)
    return ConvProjection(
        SeparableConv(dim, kernel, q_stride; bias=qkv_bias),
        SeparableConv(dim, kernel, kv_stride; bias=qkv_bias),
        SeparableConv(dim, kernel, kv_stride; bias=qkv_bias),
    )
end

function (m::ConvProjection)(x::AbstractArray{T,N}) where {T,N}
    D = N - 2  # Spatial Dimensions
    C = size(x,1)  # Channel Dimension
    O = size(x,N)  # Observation Dimension
    x_permuted = permutedims(x, (2:D+1...,1,N))  # CWHN -> WHCN
    return map(layer -> reshape(layer(x_permuted), (C,:,O)), (m.q,m.k,m.v))
end

"""
    ConvAttention(dim::Int; kernel=(3,3), q_stride=(1,1), kv_stride=(1,1), nheads=8, attn_dropout_prob=0.0, proj_dropout_prob=0.0, norm=:BN, qkv_bias=false)
    ConvAttention(dim::Int; kernel=(3,3), q_stride=(1,1), kv_stride=(1,1), nheads=8, attn_dropout_prob=0.0, proj_dropout_prob=0.0, qkv_bias=false)

A convolutional attention layer as proposed in CVT.

# Parameters
- `dim`: The dimension of the feature embedding.
- `nheads`: The number of heads to use for self attention.
- `kernel`: The kernel size used in the convolutional projection layers.
- `attn_dropout_prob`: Dropout probability in the attention block.
- `proj_dropout_prob`: Dropout probability in the projection block.
- `q_stride`: Convolutional stride used to compute the query.
- `kv_stride`: Convolutional stride used to compute the key and value.
- `qkv_bias`: Whether to include a bias term in the convolutional projection layers.
"""
mutable struct ConvAttention{Q,K,V,P,D}
    nheads::Int
    q::Q
    k::K
    v::V
    proj::P
    attn_drop::D
end

Flux.@layer :expand ConvAttention

function ConvAttention(dim::Int; kernel=(3,3), q_stride=(1,1), kv_stride=(1,1), nheads=8, attn_dropout_prob=0.0, proj_dropout_prob=0.0, qkv_bias=false)
    @argcheck length(kernel) == length(kv_stride) == length(q_stride)
    @argcheck nheads > 0
    @argcheck 0 <= attn_dropout_prob <= 1
    @argcheck 0 <= proj_dropout_prob <= 1
    return ConvAttention(
        nheads, 
        SeparableConv(dim, kernel, q_stride; bias=qkv_bias),
        SeparableConv(dim, kernel, kv_stride; bias=qkv_bias),
        SeparableConv(dim, kernel, kv_stride; bias=qkv_bias),
        Flux.Chain(Flux.Dense(dim, dim), Flux.Dropout(proj_dropout_prob)), 
        Flux.Dropout(attn_dropout_prob)
    )
end

function (m::ConvAttention)(x::AbstractArray{<:Real,N}) where N
    D = N - 2  # Spatial Dimensions
    C = size(x,1)  # Channel Dimension
    O = size(x,N)  # Observation Dimension
    x_permuted = permutedims(x, (2:D+1...,1,N))  # CWHN -> WHCN
    q, k, v = map(layer -> reshape(layer(x_permuted), (C,:,O)), (m.q,m.k,m.v))
    y, α = Flux.NNlib.dot_product_attention(q, k, v; nheads=m.nheads, fdrop=m.attn_drop)
    return reshape(m.proj(y), size(x))
end

struct SRProjection{SR,Q,KV}
    sr::SR
    q::Q
    kv::KV
end

Flux.@layer :expand SRProjection

function SRProjection(dim::Integer; qkv_bias=false, sr_ratio=(1,1), sr_method=:conv)
    # Validate Arguments
    @argcheck all(x -> x >= 1, sr_ratio)
    @argcheck sr_method in (:conv, :pool)

    # Construct SR Layer
    sr = identity
    D = length(sr_ratio)  # Spatial Dimensions
    if any(sr_ratio .> 1)
        sr = Flux.Chain(
            Base.Fix2(permutedims, (2:D+1...,1,D+2)), 
            sr_method == :conv ? Flux.Conv(sr_ratio, dim=>dim, stride=sr_ratio) : Flux.MeanPool(sr_ratio),
            Base.Fix2(permutedims, (D+1,1:D...,D+2)),
            Flux.LayerNorm(dim)
        )
    end

    # Build SRProjection
    return SRProjection(
        sr,
        Flux.Dense(dim, dim; bias=qkv_bias), 
        Flux.Dense(dim, dim*2; bias=qkv_bias), 
    )
end

function (m::SRProjection)(x::AbstractArray{<:Real,N}) where N
    C = size(x, 1)  # Channel Dimension
    O = size(x, N)  # Obs Dimension
    k, v = @pipe m.sr(x) |> reshape(_, (C,:,O)) |> m.kv |> Flux.chunk(_, 2; dims=1)
    q = reshape(m.q(x), (C,:,O))
    return (q, k, v)
end

struct SRAttention{SR,Q,KV,P,AD}
    nheads::Int
    sr::SR
    q::Q
    kv::KV
    proj::P
    attn_drop::AD
end

Flux.@layer :expand SRAttention

function SRAttention(dim::Int; nheads=8, qkv_bias=false, attn_dropout_prob=0.0, proj_dropout_prob=0.0, sr_ratio=(1,1), sr_method=:conv)
    # Validate Arguments
    @argcheck all(x -> x >= 1, sr_ratio)
    @argcheck sr_method in (:conv, :pool)

    # Construct SR Layer
    sr = identity
    D = length(sr_ratio)  # Spatial Dimensions
    if any(sr_ratio .> 1)
        sr = Flux.Chain(
            Base.Fix2(permutedims, (2:D+1...,1,D+2)), 
            sr_method == :conv ? Flux.Conv(sr_ratio, dim=>dim, stride=sr_ratio) : Flux.MeanPool(sr_ratio),
            Base.Fix2(permutedims, (D+1,1:D...,D+2)),
            Flux.LayerNorm(dim)
        )
    end

    # Build SRAttention
    return SRAttention(
        nheads, 
        sr,
        Flux.Dense(dim, dim; bias=qkv_bias), 
        Flux.Dense(dim, dim*2; bias=qkv_bias), 
        Flux.Chain(Flux.Dense(dim, dim), Flux.Dropout(proj_dropout_prob)), 
        Flux.Dropout(attn_dropout_prob)
    )
end

function (m::SRAttention)(x::AbstractArray{<:Real,N}) where N
    C = size(x, 1)  # Channel Dimension
    O = size(x, N)  # Obs Dimension
    k, v = @pipe m.sr(x) |> reshape(_, (C,:,O)) |> m.kv |> Flux.chunk(_, 2; dims=1)
    q = reshape(m.q(x), (C,:,O))
    y, α = Flux.NNlib.dot_product_attention(q, k, v; nheads=m.nheads, fdrop=m.attn_drop)
    return reshape(m.proj(y), size(x))
end

struct WindowedAttention{D,E,W<:NTuple{D},QKV,AD,P}
    nheads::Int
    window_size::W
    shift_size::W
    position_embedding::E
    qkv_layer::QKV
    attn_drop::AD
    projection::P
end

Flux.@layer :expand WindowedAttention

function WindowedAttention(dim::Integer; window_size=(7,7), shift_size=(0,0), position_embedding=false, nheads=8, qkv_bias=false, attn_dropout_prob=0.0, proj_dropout_prob=0.0)
    @assert dim % nheads==0 "planes should be divisible by nheads"

    # Initialize Layers
    qkv_layer = Flux.Dense(dim, dim * 3; bias = qkv_bias)
    attn_drop = Flux.Dropout(attn_dropout_prob)
    proj = Flux.Chain(Flux.Dense(dim, dim), Flux.Dropout(proj_dropout_prob))

    # Initialize Positional Embedding
    relative_position_embedding = position_embedding ? RelativePositionEmbedding(dim, nheads, window_size) : nothing

    # Construct Layer
    return WindowedAttention(
        nheads,
        window_size,
        shift_size,
        relative_position_embedding,
        qkv_layer, 
        attn_drop, 
        proj
    )
end

function (m::WindowedAttention{D})(x::AbstractArray{<:Number,N}) where {D,N}
    # Forward Shift
    x = _forward_shift_windows(x, m.shift_size)

    # Partition Into Windows (CxW*WxN)
    windows = window_partition(x, m.window_size)

    # Get Position Bias
    relative_position_bias = m.position_embedding

    # Get Attention Mask
    #attention_mask = nothing
    wL = prod(m.window_size)
    #attention_mask1 = Flux.ones_like(x, Bool, (wL,wL,1,size(x,4)))
    #@info typeof(attention_mask1) size(attention_mask1)
    attention_mask2 = _window_attention_mask(x, m.window_size, m.shift_size, m.nheads)
    #@info typeof(attention_mask2) size(attention_mask2)
    #@info all(attention_mask1 .== attention_mask2)
    attention_mask = attention_mask2

    # Compute Attention
    qkv = m.qkv_layer(windows)
    q, k, v = Flux.chunk(qkv, 3, dims=1)
    y, α = Flux.NNlib.dot_product_attention(q, k, v, relative_position_bias; nheads=m.nheads, fdrop=m.attn_drop, mask=attention_mask)
    y = m.projection(y)

    # Reverse Windows
    y = window_reverse(y, m.window_size)

    # Undo Shift
    return _reverse_shift_windows(y, m.shift_size)
end

_forward_shift_windows(x::AbstractArray{<:Real}, window_shift::Tuple) = circshift(x, (0,(-1 .* window_shift)...,0))

_reverse_shift_windows(x::AbstractArray{<:Real}, window_shift::Tuple) = circshift(x, (0,window_shift...,0))

function _window_attention_mask(x, window_size::NTuple{2,Int}, window_shift::NTuple{2,Int}, nheads)
    # Create Attention Mask From Window Regions
    N = size(x,4)
    wL = prod(window_size)
    nW, nH = size(x)[2:3] .÷ window_size
    region_mask = _region_mask(x, window_shift)
    mask_windows = window_partition(region_mask, window_size)  # [1 x wL x nW x nH x 1]
    mask_windows = dropdims(mask_windows, dims=1)  # [wL x nW x nH x 1]
    attn_mask = Flux.unsqueeze(mask_windows, dims=2) .== Flux.unsqueeze(mask_windows, dims=1) # [wL x wL x nW x nH x 1]

    # Extend Mask to Match nheads and batchsize
    attn_mask = Flux.unsqueeze(attn_mask, dims=3)
    attn_mask = reshape(attn_mask, (wL,wL,1,nW,nH,1))
    return reshape(repeat(attn_mask, 1, 1, 1, 1, 1, size(x,4)), (wL,wL,1,:))
    #attn_mask = Flux.zeros_like(x, Bool, (wL,wL,nheads,nW,nH,N)) .| reshape(attn_mask, (wL,wL,1,nW,nH,1))
    #return Bool.(reshape(attn_mask, (wL,wL,nheads,:)))
end

function _region_mask(x::AbstractArray{<:Real,4}, window_shift::NTuple{2,Int})
    W, H = size(x)[2:3]
    sW, sH = window_shift
    mask1 = Flux.pad_constant(Flux.zeros_like(x, (W-sW,H)), (0,sW), 1, dims=1)
    mask2 = Flux.pad_constant(Flux.zeros_like(x, (W,H-sH)), (0,sH), 2, dims=2)
    return reshape(mask1 .+ mask2, (1,W,H,1))
end

function compute_attention_mask(x, window_size::Int, feature_size::Tuple{Int,Int}, nheads)
    N = size(x,4)
    W, H = feature_size
    shift_size = window_size ÷ 2
    img_mask = zeros(UInt8, 1, W, H, 1)
    w_slices = [1:(W-window_size), (W-window_size+1):(W-shift_size), (W-shift_size+1):W]
    h_slices = [1:(H-window_size), (H-window_size+1):(H-shift_size), (H-shift_size+1):H]
    cnt = 0
    for w in w_slices
        for h in h_slices
            img_mask[:, w, h, :] .= cnt
            cnt += 1
        end
    end

    mask_windows = window_partition2(img_mask, window_size)  # 1, window_size, window_size, nW
    mask_windows = reshape(mask_windows, (window_size^2, :))  # window_size * window_size, nW
    attn_mask = Flux.unsqueeze(mask_windows, dims=2) .- Flux.unsqueeze(mask_windows, dims=1)
    attention_mask = attn_mask .== 0

    nW = size(attention_mask, 3)  # number of windows
    wL = window_size ^ 2  # length of flattened window 
    attn_mask = Flux.zeros_like(attention_mask, Bool, (wL,wL,nheads,nW,N)) .| reshape(attention_mask, (wL,wL,1,nW,1))
    return reshape(attn_mask, (wL,wL,nheads,:))
end

function window_partition2(x::AbstractArray{<:Number,4}, window_size)
    C, W, H, B = size(x)
    @pipe reshape(x, (C, window_size, W ÷ window_size, window_size, H ÷ window_size, B)) |> 
    permutedims(_, (1,2,4,3,5,6)) |> 
    reshape(_, (C, window_size, window_size, :))
end

function window_partition(x::AbstractArray{<:Any,4}, window_size::NTuple{2,Int})
    wW, wH = window_size
    C, W, H, N = size(x)
    nW, nH = (W,H) .÷ (wW,wH)
    @pipe reshape(x, (C, wW, nW, wH, nH, N)) |> 
    permutedims(_, (1,2,4,3,5,6)) |> 
    reshape(_, (C, wW*wH, nW, nH, N))
end

function window_partition(x::AbstractArray{<:Any,5}, window_size::NTuple{3,Int})
    wW, wH, wD = window_size
    C, W, H, D, N = size(x)
    nW, nH, nD = (W,H,D) .÷ (wW,wH,wD)
    @pipe reshape(x, (C, wW, nW, wH, nH, wD, nD, N)) |> 
    permutedims(_, (1,2,4,6,3,5,7,8)) |> 
    reshape(_, (C, wW*wH*wD, nW, nH, nD, N))
end

function window_reverse(x::AbstractArray{<:Any,5}, window_size::NTuple{2,Int})
    C, _, nW, nH, N = size(x)
    wW, wH = window_size
    W, H = (wW,wH) .* (nW,nH)
    @pipe reshape(x, (C,wW,wH,nW,nH,N)) |> permutedims(_, (1,2,4,3,5,6)) |> reshape(_, (C,W,H,N))
end

function window_reverse(x::AbstractArray{<:Any,6}, window_size::NTuple{3,Int})
    C, _, nW, nH, nD, N = size(x)
    wW, wH, wD = window_size
    W, H, D = (wW,wH,wD) .* (nW,nH,nD)
    @pipe reshape(x, (C,wW,wH,wD,nW,nH,nD,N)) |> permutedims(_, (1,2,5,3,6,4,7,8)) |> reshape(_, (C,W,H,D,N))
end
