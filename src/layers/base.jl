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
    return Flux.Chain(
        Flux.Dense(indims => hiddendims, act), 
        Flux.Dropout(dropout), 
        Flux.Dense(hiddendims => outdims), 
        Flux.Dropout(dropout)
    )
end

struct PositionEmbedding{E}
    embedding::E
end

Flux.@layer :expand PositionEmbedding

PositionEmbedding(dim::Int, seqlength::Int; init=zeros32) = PositionEmbedding(init((dim, seqlength, 1)))
function PositionEmbedding(dim::Int, imsize::Tuple, patchsize::Tuple; class_token=false, init=zeros32)
    @argcheck length(imsize) == length(patchsize)
    seqlength = prod(imsize .÷ patchsize) + (class_token ? 1 : 0)
    return PositionEmbedding(dim, seqlength; init)
end

(m::PositionEmbedding)(x::AbstractArray{<:Real,3}) = x .+ m.embedding
function (m::PositionEmbedding)(x::AbstractArray{<:Real,N}) where N
    return x .+ reshape(m.embedding, (:,size(x)[2:N-1]...,1))
end

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
    #relative_position_index = reshape(m.relative_position_index, :)
    #relative_position_bias = m.relative_position_bias[:,relative_position_index]
    #relative_position_bias = reshape(relative_position_bias, (:, size(x)[1:2]..., 1)) # [nH x Ww*Wh x Ww*Wh x 1]
    relative_position_bias = Flux.unsqueeze(m.relative_position_bias[:,m.relative_position_index]; dims=4)
    relative_position_bias = permutedims(relative_position_bias, (2,3,1,4)) # [Ww*Wh x Ww*Wh x nH x 1]
    return relative_position_bias .+ x
end

Flux.NNlib.apply_attn_bias(logits, bias::RelativePositionEmbedding) = bias(logits)

function _relative_position_index(window_size::NTuple{2,Int})
    # Generate coordinates
    Wy, Wx = window_size  # Window is (W x H) = (cols x rows) = (Y x X) in matrix coordinates
    coords = hcat(map(collect, Iterators.product(1:Wx, 1:Wy))...)  # Shape: 2, Wx*Wy

    # Compute relative coordinates
    relative_coords = Flux.unsqueeze(coords, 3) .- Flux.unsqueeze(coords, 2)  # Shape: 2, Wx*Wy, Wx*Wy

    # Shift to start from 0
    relative_coords[1, :, :] .+= Wx - 1
    relative_coords[2, :, :] .+= Wy - 1

    # Calculate relative position index
    relative_coords[2, :, :] .*= 2 * Wx - 1
    relative_position_index = dropdims(sum(relative_coords, dims=1), dims=1)
    return relative_position_index .+ 1  # adjust for 1-based indexing
end

function _relative_position_index(window_size::NTuple{3,Int})
    # Generate coordinates: shape (3, Wx*Wy*Wz)
    Wy, Wx, Wz = window_size # Window is (W x H x D) = (cols x rows x pages) = (Y x X x Z) in matrix coordinates
    coords = hcat(map(collect, Iterators.product(1:Wx, 1:Wy, 1:Wz))...)

    # Compute relative coordinates: shape (3, num_positions, num_positions)
    relative_coords = Flux.unsqueeze(coords, 3) .- Flux.unsqueeze(coords, 2)

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


struct Tokens{T}
    tokens::T
end

Flux.@layer :expand Tokens

Tokens(dim::Int, ntokens::Int; init=rand32) = Tokens(init((dim, ntokens, 1)))

function (m::Tokens)(x::AbstractArray{T,3}) where {T}
    tokens = m.tokens .* Flux.ones_like(x, T, (1, 1, size(x, 3)))
    return hcat(tokens, x)
end

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