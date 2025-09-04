function CVT(config::Symbol; kw...)
    @match config begin
        :B13 => CVT(64, [1,2,10], [1,3,6]; kw...)
        :B21 => CVT(64, [1,4,16], [1,3,6], kw...)
        :W24 => CVT(64, [2,2,20], [3,12,16], kw...)
    end
end

function CVT(dim::Int, depths, nheads; inchannels=3, mlp_ratio=4, dropout=0.1, drop_path=0.0, norm=:BN, nclasses=1000)
    @argcheck length(depths) == length(nheads) == 3
    Flux.Chain(
        # Encoder
        Flux.Chain(

            # Stage 1
            Flux.Chain(
                Flux.Conv((7,7), inchannels=>dim*nheads[1], stride=4, pad=Flux.SamePad()),
                Base.Fix2(permutedims, (3,1,2,4)),
                Flux.LayerNorm(dim*nheads[1]),
                Flux.Chain([CVTBlock(dim*nheads[1], nheads[1]; kv_stride=2, qkv_bias=true, mlp_ratio, dropout, drop_path, norm) for _ in 1:depths[1]]...),
                Base.Fix2(permutedims, (2,3,1,4)),
            ),

            # Stage 2
            Flux.Chain(
                Flux.Conv((3,3), dim*nheads[1]=>dim*nheads[2], stride=2, pad=Flux.SamePad()),
                Base.Fix2(permutedims, (3,1,2,4)),
                Flux.LayerNorm(dim*nheads[2]),
                Flux.Chain([CVTBlock(dim*nheads[2], nheads[2]; kv_stride=2, qkv_bias=true, mlp_ratio, dropout, drop_path, norm) for _ in 1:depths[2]]...),
                Base.Fix2(permutedims, (2,3,1,4)),
            ), 

            # Stage 3
            Flux.Chain(
                Flux.Conv((3,3), dim*nheads[2]=>dim*nheads[3], stride=2, pad=Flux.SamePad()),
                Base.Fix2(permutedims, (3,1,2,4)),
                Flux.LayerNorm(dim*nheads[3]),
                Flux.Chain([CVTBlock(dim*nheads[3], nheads[3]; kv_stride=2, qkv_bias=true, mlp_ratio, dropout, drop_path, norm) for _ in 1:depths[3]]...),
                seq2img,
            ),
        ), 

        # Classification Head
        Flux.Chain(
            Base.Fix2(permutedims, (3,1,2,4)),
            Flux.LayerNorm(dim*nheads[3]), 
            x -> dropdims(mean(x, dims=(2,3)), dims=(2,3)),
            Flux.Dense(dim*nheads[3], nclasses),
        )
    )
end

function CVTBlock(dim, nheads; kernel_size=3, mlp_ratio=4, qkv_bias=false, q_stride=1, kv_stride=1, dropout=0., drop_path=0., norm=:BN)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                ConvAttention(
                    dim; 
                    kernel=(kernel_size,kernel_size),
                    q_stride=(q_stride,q_stride),
                    kv_stride=(kv_stride,kv_stride),
                    attn_dropout_prob=dropout, 
                    proj_dropout_prob=dropout, 
                    nheads, 
                    norm, 
                    qkv_bias
                ),
                Flux.Dropout(drop_path, dims=4)
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                MLP(dim, dim*mlp_ratio, dim; dropout),
                Flux.Dropout(drop_path, dims=4)
            ), 
            +
        )
    )
end