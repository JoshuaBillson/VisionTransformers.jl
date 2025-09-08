"""
    PVT(config::Symbol; kw...)
    PVT(embed_dims, depths, nheads, mlp_ratios, sr_ratios;
        qkv_bias=true, dropout=0.1, drop_path=0.0,
        imsize=224, inchannels=3, nclasses=1000)

Construct a Pyramid Vision Transformer (PVT) model for image classification
or feature extraction. PVT builds a hierarchical representation by
progressively reducing spatial resolution while increasing the embedding
dimension. Spatial Reduction Attention (SRA) is used to make attention
efficient on high-resolution feature maps.

# Arguments
- `config`: One of `:tiny`, `:small`, `:medium`, or `:large`.
- `embed_dims`: Embedding dimension for each stage of the network.
- `depths`: Number of transformer blocks in each stage.
- `nheads`: Number of attention heads in each stage.
- `mlp_ratios`: Expansion ratio for the hidden dimension of the MLP in each stage.
- `sr_ratios`: Spatial reduction ratio for SRA in each stage.
- `qkv_bias`: Whether to add a bias to query, key, and value projections. Default is `true`.
- `dropout`: Dropout probability applied to MLP and attention outputs.
- `drop_path`: Probability for stochastic depth (drop-path) regularization.
- `imsize`: Input image size (assumed square). Default is `224`.
- `inchannels`: Number of input image channels. Default is `3`.
- `nclasses`: Number of output classes for classification. Default is `1000`.
"""
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
    drop_path_rates = _per_layer_drop_path(drop_path, depths)

    # Construct Stages
    stages = Any[]
    for i in eachindex(embed_dims)
        push!(
            stages, 
            PVTStage(
                (i == 1) ? inchannels : embed_dims[i-1],
                embed_dims[i], 
                depths[i];
                nheads=nheads[i], 
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

function PVTStage(inplanes, outplanes, depth; nheads=8, imsize=224, patchsize=4, mlp_ratio=4, qkv_bias=false, dropout=0., drop_path=0., sr_ratio=1)
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