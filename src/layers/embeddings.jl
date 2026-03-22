"""
    AbsolutePositionEmbedding(dim::Int, imsize::Tuple, patchsize::Tuple; init=zeros32, extra_tokens=0)

Absolute position embedding layer that adds learnable positional embeddings to the input tensor.

# Parameters
- `dim`: The dimensionality of the embeddings.
- `imsize`: A tuple representing the size of the input image (height, width[, depth]).
- `patchsize`: A tuple representing the size of the patches (height, width[, depth]).
- `init`: A function to initialize the positional embeddings (default is `zeros32`).
- `extra_tokens`: The number of extra tokens to add to the positional embeddings (default is 0).

# Input
An input tensor of shape `WHCN` or `WHDCN` where C is the embedding dimension and N is the batch size.
The embedding will be resampled via bilinear interpolation if the input size does not match the initialized size.
"""
struct AbsolutePositionEmbedding{E}
    embedding::E
end

Flux.@layer :expand AbsolutePositionEmbedding

function AbsolutePositionEmbedding(dim::Int, imsize::NTuple{N,Int}, patchsize::NTuple{N,Int}; init=zeros32, extra_tokens=0) where N
    @argcheck all(imsize .% patchsize .== 0)
    @argcheck all(imsize .> 0)
    @argcheck all(patchsize .> 0)
    @argcheck dim > 0
    @argcheck extra_tokens >= 0
    ntokens = prod(imsize .÷ patchsize) + extra_tokens
    return AbsolutePositionEmbedding(init(dim, ntokens, 1))
end

function (m::AbsolutePositionEmbedding)(x::AbstractArray{<:Real,3})
    return m.embedding .+ x
end

"""
    VariablePositionEmbedding(dim::Int, imsize::Tuple, patchsize::Tuple; init=zeros32)

Variable position embedding layer that adds learnable positional embeddings to the input tensor.

# Parameters
- `dim`: The dimensionality of the embeddings.
- `imsize`: A tuple representing the size of the input image (height, width[, depth]).
- `patchsize`: A tuple representing the size of the patches (height, width[, depth]).
- `init`: A function to initialize the positional embeddings (default is `zeros32`).

# Input
An input tensor of shape `WHCN` or `WHDCN` where C is the embedding dimension and N is the batch size.
The embedding will be resampled via bilinear interpolation if the input size does not match the initialized size.
"""
struct VariablePositionEmbedding{E}
    embedding::E
end

Flux.@layer :expand VariablePositionEmbedding

function VariablePositionEmbedding(dim::Int, imsize::NTuple{N,Int}, patchsize::NTuple{N,Int}; init=zeros32) where N
    @argcheck all(imsize .% patchsize .== 0)
    @argcheck all(imsize .> 0)
    @argcheck all(patchsize .> 0)
    @argcheck dim > 0
    return VariablePositionEmbedding(init((imsize .÷ patchsize)..., dim, 1))
end

function (m::VariablePositionEmbedding)(x::AbstractArray{<:Real,N}) where N
    D = N - 2  # Determine if 2D or 3D based on input dimensions
    input_imsize = size(x)[2:D+1]
    pos_embed = m.embedding
    if input_imsize != imsize(m.embedding)
        pos_embed = Flux.upsample_bilinear(pos_embed; size=input_imsize)  # Resample positional embedding to match input size
    end
    pos_embed = permutedims(pos_embed, (D+1, 1:D..., D+2))  # WHCN -> CWHN or WHDCN -> CWHDN
    return pos_embed .+ x
end

"""
    PEG(dim::Int; kernel=(3,3), bias=true)

Position Embedding Generator (PEG) layer that applies a depthwise convolution to the input tensor to encode positional information.

# Parameters
- `dim`: The number of input and output channels.
- `kernel`: The size of the convolution kernel (default is `(3,3)`).
- `bias`: Whether to include a bias term in the convolution (default is `true`).
"""
function PEG(dim::Int; kernel=(3,3), bias=true)
    return Flux.SkipConnection(
        Flux.DepthwiseConv(kernel, dim=>dim; pad=Flux.SamePad(), stride=(1,1), bias),
        +
    )
end

"""
    RelativePositionEmbedding(dim::Int, nheads::Int, window_size::Tuple; init=zeros32)

Relative position embedding layer that adds learnable relative positional biases to the attention scores.

# Parameters
- `dim`: The dimensionality of the embeddings.
- `nheads`: The number of attention heads.
- `window_size`: A tuple representing the size of the attention window (height, width[, depth]).
- `init`: A function to initialize the relative positional biases (default is `zeros32`).

# Input
An attention scores tensor of shape `[Ww*Wh x Ww*Wh x nheads x 1]` where Ww and Wh are the window 
width and height, respectively. For 3D data, the shape is `[Ww*Wh*Wd x Ww*Wh*Wd x nheads x 1]` where 
Wd is the window depth. 

# Note
We define an overloaded version of `Flux.NNlib.apply_attn_bias` to apply the relative position embedding
to the attention scores. Thus, it is sufficient to pass this layer to `Flux.NNlib.dot_product_attention` as
the bias argument.

# Example
```julia
q, k, v = Flux.chunk(qkv(x), 3; dims=1)
y, α = Flux.dot_product_attention(q, k, v, position_embedding; nheads)
```
"""
struct RelativePositionEmbedding{B,I}
    relative_position_bias::B
    relative_position_index::I
end

Flux.@layer :expand RelativePositionEmbedding trainable=(relative_position_bias,)

function RelativePositionEmbedding(dim::Int, nheads::Int, window_size::Tuple; init=zeros32)
    @argcheck all(window_size .> 0)
    @argcheck dim > 0
    @argcheck nheads > 0
    num_positions = prod((2 .* window_size) .- 1)
    relative_position_bias = init((nheads, num_positions))
    relative_position_index = _relative_position_index(window_size)
    return RelativePositionEmbedding(relative_position_bias, relative_position_index)
end

function (m::RelativePositionEmbedding)(x)
    relative_position_bias = Flux.unsqueeze(m.relative_position_bias[:,m.relative_position_index]; dims=4)
    relative_position_bias = permutedims(relative_position_bias, (2,3,1,4)) # [Ww*Wh x Ww*Wh x nH x 1]
    return relative_position_bias .+ x
end

Flux.NNlib.apply_attn_bias(logits, bias::RelativePositionEmbedding) = bias(logits)

# 2D Relative Position Index Calculation
function _relative_position_index(window_size::NTuple{2,Int})
    # Generate coordinates
    Wy, Wx = window_size  # Window is (W x H) = (cols x rows) = (Y x X) in matrix coordinates
    coords = hcat(map(collect, Iterators.product(1:Wx, 1:Wy))...)  # Shape: 2, Wx*Wy

    # Compute relative coordinates
    relative_coords = Flux.unsqueeze(coords; dims=3) .- Flux.unsqueeze(coords; dims=2)  # Shape: 2, Wx*Wy, Wx*Wy

    # Shift to start from 0
    relative_coords[1, :, :] .+= Wx - 1
    relative_coords[2, :, :] .+= Wy - 1

    # Calculate relative position index
    relative_coords[2, :, :] .*= 2 * Wx - 1
    relative_position_index = dropdims(sum(relative_coords, dims=1), dims=1)
    return relative_position_index .+ 1  # adjust for 1-based indexing
end

# 3D Relative Position Index Calculation
function _relative_position_index(window_size::NTuple{3,Int})
    # Generate coordinates: shape (3, Wx*Wy*Wz)
    Wy, Wx, Wz = window_size # Window is (W x H x D) = (cols x rows x pages) = (Y x X x Z) in matrix coordinates
    coords = hcat(map(collect, Iterators.product(1:Wx, 1:Wy, 1:Wz))...)

    # Compute relative coordinates: shape (3, num_positions, num_positions)
    relative_coords = Flux.unsqueeze(coords; dims=3) .- Flux.unsqueeze(coords; dims=2)

    # Shift to start from 0
    relative_coords[1, :, :] .+= Wx - 1
    relative_coords[2, :, :] .+= Wy - 1
    relative_coords[3, :, :] .+= Wz - 1

    # Calculate relative position index
    relative_coords[2, :, :] .*= (2*Wx - 1)
    relative_coords[3, :, :] .*= (2*Wx - 1) * (2*Wy - 1)
    relative_position_index = dropdims(sum(relative_coords, dims=1), dims=1)
    return relative_position_index .+ 1  # adjust for 1-based indexing
end