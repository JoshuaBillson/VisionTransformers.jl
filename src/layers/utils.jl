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

imsize(x::AbstractArray{<:Any,N}) where N = size(x)[1:N-2]

seconddimmean(x) = dropdims(mean(x; dims=2); dims=2)

rand32(x::Tuple) = rand32(x...)
rand32(xs::Vararg{I}) where I <: Integer = Flux.rand32(xs...)

zeros32(x::Tuple) = zeros32(x...)
zeros32(xs::Vararg{I}) where I <: Integer = Flux.zeros32(xs...)

randn32(x::Tuple) = randn32(x...)
randn32(xs::Vararg{I}) where I <: Integer = Flux.randn32(xs...)

trunc_normal(x::Tuple) = trunc_normal(x...)
trunc_normal(xs::Vararg{I}) where I <: Integer = Flux.truncated_normal(xs..., std=0.02)