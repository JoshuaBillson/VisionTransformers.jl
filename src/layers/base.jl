"""
    MLP(indims, hiddendims, outdims; dropout=0.0, act=Flux.gelu)

Build a 2-layer multi-layer-perceptron.

# Parameters
- `indims`: The dimension of the input features.
- `hiddendims`: The dimension of the hidden features.
- `outdims`: The dimension of the output features.
- `dropout`: The dropout probability following each Dense layer.
- `act`: The activation function following the first Dense layer.
"""
function MLP(indims, hiddendims, outdims; dropout=0.0, act=Flux.gelu)
    @argcheck indims > 0
    @argcheck hiddendims > 0
    @argcheck outdims > 0
    @argcheck 0 <= dropout < 1
    return Flux.Chain(
        Flux.Dense(indims => hiddendims, act), 
        Flux.Dropout(dropout), 
        Flux.Dense(hiddendims => outdims), 
        Flux.Dropout(dropout)
    )
end

"""
    SeparableConv(dim, kernel::NTuple{N,Int}, stride::NTuple{N,Int}; bias=false) where N

Build a separable convolution layer consisting of a depthwise convolution followed by a pointwise convolution.

# Parameters
- `dim`: The number of input and output channels.
- `kernel`: The size of the convolution kernel.
- `stride`: The stride of the convolution.
- `bias`: Whether to include a bias term in the pointwise convolution.
"""
function SeparableConv(dim, kernel::NTuple{N,Int}, stride::NTuple{N,Int}; bias=false) where N
    @argcheck dim > 0
    @argcheck all(kernel .> 0)
    @argcheck all(stride .> 0)
    return Flux.Chain(
        Flux.DepthwiseConv(kernel, dim=>dim; pad=Flux.SamePad(), stride, bias=false), 
        Flux.Conv(tuple([1 for _ in 1:N]...), dim=>dim; pad=Flux.SamePad(), stride=tuple([1 for _ in 1:N]...), bias),
        Flux.BatchNorm(dim)
    )
end

"""
    Tokens(dim, ntokens; init=rand32)

Learnable token embedding layer that prepends a specified number of learnable tokens to the input sequence.

# Parameters
- `dim`: The dimensionality of the tokens.
- `ntokens`: The number of learnable tokens to prepend.
- `init`: A function to initialize the token embeddings (default is `rand32`).

# Input
A tensor of shape `CLN` where C is the embedding dimension, L is the sequence length, and N is the batch size. 
The output will have shape `C(L+ntokens)N` with the learnable tokens prepended to the input sequence.
"""
struct Tokens{T}
    tokens::T
end

Flux.@layer :expand Tokens

Tokens(dim::Int, ntokens::Int; init=rand32) = Tokens(init((dim, ntokens, 1)))

function (m::Tokens)(x::AbstractArray{T,3}) where {T}
    tokens = m.tokens .* Flux.ones_like(x, T, (1, 1, size(x, 3)))
    return hcat(tokens, x)
end

"""
    StripTokens(ntokens)

Layer that removes a specified number of tokens from the beginning of the input sequence.

# Parameters
- `ntokens`: The number of tokens to remove from the beginning of the sequence.
"""
struct StripTokens
    ntokens::Int
end

Flux.@layer :expand StripTokens

(m::StripTokens)(x::AbstractArray{<:Real,3}) = x[:, m.ntokens+1:end, :]

struct PatchMerging{D,N}
    reduction::D
    norm::N
end

Flux.@layer :expand PatchMerging

function PatchMerging(dim::Integer)
    return PatchMerging(
        Flux.Dense(dim*4=>dim*2, bias=false),
        Flux.LayerNorm(dim*4)
    )
end

function (m::PatchMerging)(x::AbstractArray{<:Real,4})
    # Slice into Quadrants
    x1 = @view x[:,1:2:end,1:2:end,:]
    x2 = @view x[:,2:2:end,1:2:end,:]
    x3 = @view x[:,1:2:end,2:2:end,:]
    x4 = @view x[:,2:2:end,2:2:end,:]

    # Concatenate Slices
    return @pipe cat(x1, x2, x3, x4, dims=1) |> m.norm |> m.reduction
end

function TransformerBlock(dim::Int, attention, mlp; drop_path=0.0, path_dim=3)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                attention,
                Flux.Dropout(drop_path, dims=path_dim)
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                mlp,
                Flux.Dropout(drop_path, dims=path_dim)
            ), 
            +
        )
    )
end