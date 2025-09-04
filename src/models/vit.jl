function ViT(config::Symbol; kw...)
    @match config begin
        :tiny => ViT(192, 3, 12; kw...)
        :small => ViT(384, 6, 12; kw...)
        :base => ViT(768, 12, 12; kw...)
        :large => ViT(1024, 16, 24; kw...)
        :huge => ViT(1280, 16, 32; kw...)
    end
end

function ViT(dim::Integer, nheads::Integer, depth::Integer; imsize=(224,224), patchsize=(16,16), inchannels=3, nclasses=1000, mlp_ratio=4, qkv_bias=true, dropout=0.1, drop_path=0.0)
    # Get Per-Block Path Drop Rate
    drop_path_rates = LinRange(0, drop_path, depth) |> collect

    # Construct ViT
    Flux.Chain(

        # Encoder
        Flux.Chain(

            # Patch Embedding
            Flux.Conv(patchsize, inchannels=>dim, stride=patchsize), 

            # Position Embedding
            Flux.Chain(
                img2seq,
                Flux.LayerNorm(dim),
                PositionEmbedding(dim, imsize, patchsize),
                Flux.Dropout(dropout),
            ),

            # Encoder Blocks
            Flux.Chain([ViTBlock(dim; nheads, mlp_ratio, qkv_bias, dropout, drop_path=drop_path_rates[i]) for i in 1:depth]...),
        ),

        # Classification Head
        Flux.Chain(
            seconddimmean, 
            Flux.LayerNorm(dim), 
            Flux.Dense(dim, nclasses)
        ),
    )
end

function ViTBlock(dim::Integer; nheads=8, mlp_ratio=4, qkv_bias=true, dropout=0.0, drop_path=0.0)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                MultiHeadAttention(dim; nheads, qkv_bias, attn_dropout_prob=dropout, proj_dropout_prob=dropout),
                Flux.Dropout(drop_path, dims=3)
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                MLP(dim, dim*mlp_ratio, dim; dropout),
                Flux.Dropout(drop_path, dims=3)
            ), 
            +
        )
    )
end