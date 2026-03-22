"""
    Twins(config::Symbol; kw...)
    Twins(embed_dims, depths, nheads; 
          window_sizes=[7,7,7,7], sr_ratios=[8,4,2,1], mlp_ratio=4, 
          qkv_bias=true, dropout=0.1, attn_dropout_prob=0.1, 
          drop_path=0.0, inchannels=3, nclasses=1000)

Construct a Twins Vision Transformer model. The architecture consists of alternating local 
and global attention mechanisms to effectively capture both local and global features in images.

# Parameters
- `config`: Predefined configuration symbol. One of `:small`, `:base`, or `:large`.
- `embed_dims`: Embedding dimensions for each stage.
- `depths`: Number of transformer blocks in each stage.
- `nheads`: Number of attention heads in each stage.
- `window_sizes`: Spatial size of the attention window for each stage. Default is `[7,7,7,7]`.
- `sr_ratios`: Spatial reduction ratios for SRAttention in each stage. Default is `[8,4,2,1]`.
- `mlp_ratio`: Expansion ratio for the hidden dimension of the MLP relative to the embedding dimension. Default is `4`.
- `qkv_bias`: Whether to add bias to query, key, and value projections. Default is `true`.
- `dropout`: Dropout probability applied to embeddings, MLP, and attention outputs. Default is `0.1`.
- `attn_dropout`: Dropout probability applied to attention weights. Default is `0.1`.
- `drop_path`: Probability for stochastic depth (drop-path) regularization. Default is `0.0`.
- `inchannels`: Number of input image channels. Default is `3`.
- `nclasses`: Number of output classes for classification. Default is `1000`.
"""
function Twins(config::Symbol; kw...)
    @match config begin
        :small => Twins(
            [64,128,256,512], 
            [2,2,10,4], 
            [2,4,8,16];
            kw...)
        :base => Twins(
            [96,192,384,768], 
            [2,2,18,2], 
            [3,6,12,24];
            kw...)
        :large => Twins(
            [128,256,512,1024], 
            [2,2,18,2], 
            [4,8,16,32];
            kw...)
    end
end

function Twins(embed_dims, depths, nheads; window_sizes=[7,7,7,7], sr_ratios=[8,4,2,1], mlp_ratio=4, qkv_bias=true, dropout=0.1, attn_dropout=0.1, drop_path=0.0, inchannels=3, nclasses=1000)
    @argcheck length(embed_dims) == length(depths) == length(nheads) == length(window_sizes) == length(sr_ratios)
    @argcheck all(embed_dims .> 0)
    @argcheck all(depths .> 0)
    @argcheck all(nheads .> 0)
    @argcheck all(window_sizes .> 0)
    @argcheck all(sr_ratios .> 0)
    @argcheck mlp_ratio > 0
    @argcheck nclasses > 0
    @argcheck inchannels > 0
    @argcheck 0 <= dropout < 1
    @argcheck 0 <= attn_dropout < 1
    @argcheck 0 <= drop_path < 1

    # Get Per-Block Path Drop Rate
    drop_path_rates = _per_layer_drop_path(drop_path, depths)

    Flux.Chain(
        # Encoder
        Flux.Chain(
            [TwinsStage(
                i == 1 ? inchannels : embed_dims[i-1], embed_dims[i]; 
                patch_size=(i == 1 ? 4 : 2), 
                nheads=nheads[i], 
                mlp_ratio, 
                qkv_bias, 
                sr_ratio=sr_ratios[i], 
                window_size=window_sizes[i], 
                depth=depths[i], 
                dropout, 
                attn_dropout,
                drop_path=drop_path_rates[i]
            ) for i in eachindex(depths) ]...,
        ),

        # Classification Head
        Flux.Chain(
            Base.Fix2(permutedims, (3,1,2,4)),  # HWCN -> CHWN
            Flux.LayerNorm(last(embed_dims)), 
            x -> dropdims(mean(x, dims=(2,3)), dims=(2,3)),
            Flux.Dense(last(embed_dims), nclasses)
        )

    )
end

function TwinsStage(indim, outdim; patch_size=4, nheads=8, mlp_ratio=4, qkv_bias=false, sr_ratio=4, window_size=7, depth=4, dropout=0.0, attn_dropout=0.0, drop_path=0.0)
    drop_path = drop_path isa Number ? repeat([drop_path], depth) : drop_path
    Flux.Chain(
        # Patch Embedding
        Flux.Chain(
            Flux.Conv((patch_size,patch_size), indim=>outdim; stride=patch_size),
            Base.Fix2(permutedims, (3,1,2,4)),  # HWCN -> CHWN
            Flux.LayerNorm(outdim),
            Flux.Dropout(dropout),
        ), 

        # First Transformer Block
        TwinsBlock(outdim; nheads, mlp_ratio, qkv_bias, sr_ratio, window_size, dropout, attn_dropout, drop_path=drop_path[1], variant=:LSA),

        # PEG Embedding
        Flux.Chain(
            Base.Fix2(permutedims, (2,3,1,4)),  # CHWN -> HWCN
            PEG(outdim),
            Base.Fix2(permutedims, (3,1,2,4)),  # HWCN -> CHWN
        ),

        # Remaining Transformer Blocks
        [TwinsBlock(outdim; nheads, mlp_ratio, qkv_bias, sr_ratio, window_size, dropout, attn_dropout, drop_path=drop_path[i], variant=(i % 2 == 0 ? :GSA : :LSA)) for i in 2:depth]...,
        Flux.LayerNorm(outdim),
        Base.Fix2(permutedims, (2,3,1,4)),  # CHWN -> HWCN
    )
end

function TwinsBlock(dim; nheads=8, mlp_ratio=4, qkv_bias=false, sr_ratio=4, window_size=7, dropout=0.0, attn_dropout=0.0, drop_path=0.0, variant=:LSA)
    # Construct Attention Layer
    @argcheck variant in (:LSA, :GSA)
    attention = @match variant begin
        :LSA => WindowedAttention(dim; nheads, window_size=(window_size,window_size), qkv_bias, attn_dropout_prob=attn_dropout, proj_dropout_prob=dropout)
        :GSA => SRAttention(dim; nheads, qkv_bias, sr_ratio=(sr_ratio,sr_ratio), attn_dropout_prob=attn_dropout, proj_dropout_prob=dropout)
    end

    # Build Block
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                attention,
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