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

PositionEmbedding(dim::Int, seqlength::Int) = PositionEmbedding(rand(Float32, (dim, seqlength, 1)))
function PositionEmbedding(dim::Int, imsize::Tuple, patchsize::Tuple)
    @argcheck length(imsize) == length(patchsize)
    return PositionEmbedding(dim, prod(imsize .÷ patchsize))
end

(m::PositionEmbedding)(x::AbstractArray{<:Real,3}) = x .+ m.embedding
function (m::PositionEmbedding)(x::AbstractArray{<:Real,N}) where N
    return x .+ reshape(m.embedding, (:,size(x)[2:N-1]...,1))
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