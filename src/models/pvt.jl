function PVT(config::Symbol; kw...)
    @match config begin
        :tiny => PVT(
            [64,128,320,512], # Embed Dims
            [2,2,2,2],        # Block Depths
            [1,2,5,8],        # Number of Heads
            [8,8,4,4],        # MLP Ratio
            [8,4,2,1];        # SR Ratios
            kw...)
        :small => PVT(
            [64,128,320,512], # Embed Dims
            [3,4,6,3],        # Block Depths
            [1,2,5,8],        # Number of Heads
            [8,8,4,4],        # MLP Ratio
            [8,4,2,1];        # SR Ratios
            kw...)
        :medium => PVT(
            [64,128,320,512], # Embed Dims
            [3,4,18,3],       # Block Depths
            [1,2,5,8],        # Number of Heads
            [8,8,4,4],        # MLP Ratio
            [8,4,2,1];        # SR Ratios
            kw...)
        :large => PVT(
            [64,128,320,512], # Embed Dims
            [3,8,27,3],       # Block Depths
            [1,2,5,8],        # Number of Heads
            [8,8,4,4],        # MLP Ratio
            [8,4,2,1];        # SR Ratios
            kw...)
    end
end

function PVT(embed_dims, depths, nheads, mlp_ratios, sr_ratios; qkv_bias=true, dropout=0.1, drop_path=0.0, imsize=224, inchannels=3, nclasses=1000)
    # Validate Arguments
    @argcheck length(embed_dims) == length(depths) == length(nheads) == length(mlp_ratios) == length(sr_ratios)

    # Get Per-Block Path Drop Rate
    drop_path_linspace = LinRange(0, drop_path, sum(depths))
    drop_path_rates = map(zip(cumsum(vcat(0, depths)) .+ 1, cumsum(depths))) do i
        return collect(drop_path_linspace[i[1]:i[2]])
    end

    # Construct Stages
    stages = Any[]
    for i in eachindex(embed_dims)
        push!(
            stages, 
            PVTStage(
                (i == 1) ? inchannels : embed_dims[i-1],
                embed_dims[i];
                nheads=nheads[i], 
                depth=depths[i], 
                mlp_ratio=mlp_ratios[i],
                sr_ratio=sr_ratios[i], 
                patchsize=(i == 1) ? 4 : 2,
                qkv_bias, 
                dropout, 
                drop_path=drop_path_rates[i],
                imsize
            )
        )
        imsize = (i == 1) ? imsize ÷ 4 : imsize ÷ 2
    end

    # Return PVT
    Flux.Chain(
        # Encoder
        Flux.Chain(stages...),

        # Classification Head
        Flux.Chain(
            Base.Fix2(permutedims, (3,1,2,4)),
            Flux.LayerNorm(last(embed_dims)), 
            x -> dropdims(mean(x, dims=(2,3)), dims=(2,3)),
            Flux.Dense(last(embed_dims), nclasses)
        )
    )
end

function PVTStage(inplanes, outplanes; nheads=8, imsize=224, patchsize=4, depth=4, mlp_ratio=4, qkv_bias=false, dropout=0., drop_path=0., sr_ratio=1)
    drop_path = drop_path isa Number ? repeat([drop_path], depth) : drop_path
    Flux.Chain(
        Flux.Conv((patchsize,patchsize), inplanes=>outplanes; stride=patchsize), 
        Base.Fix2(permutedims, (3,1,2,4)),
        Flux.LayerNorm(outplanes),
        PositionEmbedding(outplanes, (imsize,imsize), (patchsize, patchsize)),
        Flux.Dropout(dropout), 
        [PVTBlock(outplanes, nheads; mlp_ratio, qkv_bias, sr_ratio, drop_path=drop_path[i], dropout) for i in 1:depth]..., 
        Base.Fix2(permutedims, (2,3,1,4)),
    )
end

function PVTBlock(dim, nheads; mlp_ratio=4, qkv_bias=false, sr_ratio=1, dropout=0., drop_path=0.)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                SRAttention(dim; nheads, qkv_bias, sr_ratio=(sr_ratio,sr_ratio), attn_dropout_prob=dropout, proj_dropout_prob=dropout),
                Flux.Dropout(drop_path, dims=4)  # Drop Path
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                MLP(dim, dim*mlp_ratio, dim; dropout),
                Flux.Dropout(drop_path, dims=4)  # Drop Path
            ), 
            +
        )
    )
end