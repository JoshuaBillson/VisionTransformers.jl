"""
    ViT(config::Symbol; kw...)
    ViT(dim::Integer, nheads::Integer, depth::Integer;
        imsize=(224,224), patchsize=(16,16), inchannels=3,
        nclasses=1000, mlp_ratio=4, qkv_bias=true,
        dropout=0.1, attn_dropout=0.1, drop_path=0.0, 
        class_token=false, register_tokens=0)

Construct a Vision Transformer (ViT) model for image classification.  
The input image is split into non-overlapping patches, each patch is
linearly embedded into a `dim`-dimensional vector, and a sequence of
Transformer blocks is applied. Optionally, a learnable class token may
be prepended to the patch sequence for classification.

# Arguments
- `config`: One of `:tiny`, `:small`, `:base`, `:large`, or `:huge`.
- `dim`: Embedding dimension of the patch tokens.
- `nheads`: Number of attention heads in each Transformer block.
- `depth`: Number of Transformer blocks in the encoder.
- `imsize`: Input image size `(height, width)`. Default `(224,224)`.
- `patchsize`: Patch size `(height, width)`. Default `(16,16)`.
- `inchannels`: Number of input image channels. Default is `3`.
- `nclasses`: Number of output classes for classification. Default is `1000`.
- `mlp_ratio`: Expansion ratio for the hidden dimension of the MLP relative to `dim`. Default is `4`.
- `qkv_bias`: Whether to add a bias to query, key, and value projections. Default is `true`.
- `dropout`: Dropout probability applied to embeddings, MLP, and attention outputs. Default is `0.1`.
- `attn_dropout`: Dropout probability applied to attention weights. Default is `0.1`.
- `drop_path`: Probability for stochastic depth (drop-path) regularization. Default is `0.0`.
- `class_token`: Whether to prepend a learnable class token to the patch sequence for classification. If `false`, global average pooling is applied instead.
- `register_tokens`: Number of additional learnable tokens to prepend to the patch sequence. Default is `0`.
"""
function ViT(config::Symbol; kw...)
    @match config begin
        :tiny => ViT(192, 3, 12; kw...)
        :small => ViT(384, 6, 12; kw...)
        :base => ViT(768, 12, 12; kw...)
        :large => ViT(1024, 16, 24; kw...)
        :huge => ViT(1280, 16, 32; kw...)
    end
end

function ViT(dim::Integer, nheads::Integer, depth::Integer; imsize=(224,224), patchsize=(16,16), inchannels=3, nclasses=1000, mlp_ratio=4, qkv_bias=true, dropout=0.1, attn_dropout=0.1, drop_path=0.0, class_token=false, register_tokens=0)
    @argcheck dim > 0
    @argcheck nheads > 0
    @argcheck depth > 0
    @argcheck all(imsize .> 0)
    @argcheck all(patchsize .> 0)
    @argcheck inchannels > 0
    @argcheck nclasses > 0
    @argcheck mlp_ratio > 0
    @argcheck 0 <= dropout < 1
    @argcheck 0 <= attn_dropout < 1
    @argcheck 0 <= drop_path < 1
    @argcheck register_tokens >= 0
    @argcheck all(imsize .% patchsize .== 0)

    # Get Per-Block Path Drop Rate
    drop_path_rates = _per_layer_drop_path(drop_path, depth)
    extra_tokens = (class_token ? 1 : 0) + register_tokens

    # Construct ViT
    Flux.Chain(

        # Encoder
        Flux.Chain(

            # Patch Embedding
            Flux.Chain(
                Flux.Conv(patchsize, inchannels=>dim, stride=patchsize), 
                img2seq,
                extra_tokens > 0 ? Tokens(dim, extra_tokens; init=trunc_normal) : identity,
                AbsolutePositionEmbedding(dim, imsize, patchsize; init=randn32, extra_tokens),
                Flux.Dropout(dropout),
            ),

            # Encoder Blocks
            Flux.Chain(
                [ViTBlock(dim; nheads, mlp_ratio, qkv_bias, dropout, attn_dropout, drop_path=drop_path_rates[i]) for i in 1:depth]...
            ),
        ),

        # Classification Head
        Flux.Chain(
            class_token ? (x -> x[:, 1, :]) : seconddimmean,
            Flux.LayerNorm(dim), 
            Flux.Dense(dim, nclasses)
        ),
    )
end

function ViTBlock(dim::Integer; nheads=8, mlp_ratio=4, qkv_bias=true, dropout=0.0, attn_dropout=0.0, drop_path=0.0)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                MultiHeadAttention(dim; nheads, qkv_bias, attn_dropout_prob=attn_dropout, proj_dropout_prob=dropout),
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