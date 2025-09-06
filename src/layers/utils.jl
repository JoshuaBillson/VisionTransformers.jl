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

seconddimmean(x) = dropdims(mean(x; dims=2); dims=2)

rand32(x::Tuple) = Flux.rand32(x...)

zeros32(x::Tuple) = Flux.zeros32(x...)

randn32(x::Tuple) = Flux.randn32(x...)