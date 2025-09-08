"""
    SWIN(config::Symbol; kw...)
    SWIN(embed_dims, depths, nheads;
         window_size=7, position_embedding=true, mlp_ratio=4,
         qkv_bias=true, dropout=0.1, drop_path=0.0,
         inchannels=3, nclasses=1000)

Construct a Swin Transformer model for image classification or feature
extraction. The architecture is composed of multiple stages of
Shifted Windowed Multi-Head Self-Attention (SW-MSA) and MLP blocks.
The model progressively merges patches while increasing channel
dimension, enabling hierarchical feature learning.

# Arguments
- `config`: One of `:tiny`, `:small`, `:base`, or `:large`.
- `embed_dims`: Embedding dimension for each stage of the network.
- `depths`: Number of transformer blocks in each stage.
- `nheads`: Number of attention heads in each stage.
- `window_size`: Spatial size of the attention window. Default is `7`.
- `position_embedding`: Whether to include learnable relative position embeddings in attention.
- `mlp_ratio`: Expansion ratio for the hidden dimension of the MLP relative to the embedding dimension. Default is `4`.
- `qkv_bias`: Whether to add bias to query, key, and value projections.
- `dropout`: Dropout probability applied to MLP and attention outputs.
- `drop_path`: Probability for stochastic depth (drop-path) regularization.
- `inchannels`: Number of input image channels. Default is `3`.
- `nclasses`: Number of output classes for classification. Default is `1000`.
"""
function SWIN(config::Symbol; kw...)
    @match config begin
        :tiny => SWIN(
            [96,192,384,768], 
            [2,2,6,2], 
            [3,6,12,24];
            kw...)
        :small => SWIN(
            [96,192,384,768], 
            [2,2,18,2], 
            [3,6,12,24];
            kw...)
        :base => SWIN(
            [128,256,512,1024], 
            [2,2,18,2], 
            [4,8,16,32];
            kw...)
        :large => SWIN(
            [192,384,768,1536], 
            [2,2,18,2], 
            [6,12,24,48];
            kw...)
    end
end

function SWIN(embed_dims, depths, nheads; window_size=7, position_embedding=true, mlp_ratio=4, qkv_bias=true, dropout=0.1, drop_path=0.0, inchannels=3, nclasses=1000)
    @argcheck length(embed_dims) == length(depths) == length(nheads)
    return Flux.Chain(
        # Patch Embedding
        Flux.Conv((4,4), inchannels=>embed_dims[1], stride=(4,4)),
        Base.Fix2(permutedims, (3,1,2,4)),
        Flux.LayerNorm(embed_dims[1]),
        Flux.Dropout(dropout),

        # Encoder
        Flux.Chain(
            [SWINStage(
                embed_dims[i]; 
                depth=depths[i], 
                nheads=nheads[i], 
                merge_patches=(i>1),
                window_size,
                position_embedding,
                mlp_ratio, 
                qkv_bias, 
                dropout, 
                drop_path
            ) for i in eachindex(embed_dims)]...
        ),

        # Classification Head
        Flux.Chain(
            Flux.LayerNorm(last(embed_dims)), 
            x -> dropdims(mean(x, dims=(2,3)), dims=(2,3)),
            Flux.Dense(last(embed_dims), nclasses)
        )
    )
end

function SWINStage(dims; nheads=8, window_size=7, position_embedding=true, depth=4, mlp_ratio=4, qkv_bias=false, dropout=0., drop_path=0., merge_patches=true)
    Flux.Chain(
        merge_patches ? PatchMerging(dims÷2) : identity,
        [SWINBlock(dims; window_size, position_embedding, nheads, mlp_ratio, qkv_bias, window_shift=((i%2)==0), dropout, drop_path) for i in 1:depth]..., 
    )
end

function SWINBlock(dim::Int; window_size=7, position_embedding=true, nheads=8, mlp_ratio=4, qkv_bias=false, window_shift=false, dropout=0.0, drop_path=0.0)
    window_size = (window_size,window_size)
    shift_size = window_shift ? (window_size .÷ 2) : (0,0)
    #shift_size = (0,0)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                WindowedAttention(dim; nheads, window_size, shift_size, position_embedding, qkv_bias, attn_dropout_prob=dropout, proj_dropout_prob=dropout),
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