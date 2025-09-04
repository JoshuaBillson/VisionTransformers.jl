chunkseq(x, n::Int) = Flux.chunk(x, n; dims=1)

"""
    img2seq(x::AbstractArray{<:Any,4})

Take a collection of image tokens of size [W x H x C x N] and flatten them into a sequence
of size [C x L x N] where L = W * H.
"""
function img2seq(x::AbstractArray{<:Any,N}) where N
    C = size(x,N-1)
    B = size(x,N)
    permutedims(reshape(x, (:, C, B)), (2, 1, 3))
end

"""
    seq2img(x::AbstractArray{<:Any,3})

Take a sequence of image tokens of size [C x L x N] and reshape it into an image of size [W x H x C x N], where
W = H = sqrt(L).
"""
function seq2img(x::AbstractArray{<:Any,3}, imsize=nothing)
    C, _, N = size(x)
    imsize = isnothing(imsize) ? imsize(x) : imsize
    return reshape(permutedims(x, (2, 1, 3)), imsize..., C, N)
end
seq2img(x::AbstractArray{<:Any,4}) = permutedims(x, (2,3,1,4))
seq2img(x::AbstractArray{<:Any,5}) = permutedims(x, (2,3,4,1,5))

imsize(x::AbstractArray{<:Any,4}) = size(x)[1:2]
imsize(x::AbstractArray{<:Any,5}) = size(x)[1:3]
function imsize(x::AbstractArray{<:Any,3})
    S = isqrt(size(x,2))
    return (S,S)
end

set_imsize(l, imsize) = l
set_imsize(l::Flux.Chain, imsize) = Flux.Chain([set_imsize(layer, imsize) for layer in l.layers]...)

function windowed_dot_product_attention(
    q::AbstractArray{<:Real,N}, k::AbstractArray{<:Real,N}, v::AbstractArray{<:Real,N};
    bias=nothing, fdrop=identity, mask=nothing, nheads=1) where N
end

function dot_product_attention(
    q::AbstractArray{<:Real,N}, k::AbstractArray{<:Real,N}, v::AbstractArray{<:Real,N}; kw...) where N

    q_flat, k_flat, v_flat = map(_flatten_seq, (q, k, v))
    x, α = dot_product_attention(q_flat, k_flat, v_flat; kw...)
    return reshape(x, size(x, 1), size(q)[2:end]...), α
end

function dot_product_attention(
    q::AbstractArray{<:Real,3}, k::AbstractArray{<:Real,3}, v::AbstractArray{<:Real,3};
    bias=nothing, fdrop=identity, mask=nothing, nheads=1)
    return Flux.NNlib.dot_product_attention(q, k, v, bias; fdrop, mask, nheads)
end

_seq_size(x::AbstractArray{<:Any,N}) where N = size(x)[2:N-1]

_seq_length(x::AbstractArray) = _seq_size(x) |> prod

_flatten_seq(x::AbstractArray{<:Any,N}) where N = reshape(x, size(x,1), :, size(x,N))

seconddimmean(x) = dropdims(mean(x; dims=2); dims=2)